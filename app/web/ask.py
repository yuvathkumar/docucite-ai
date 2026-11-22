"""Question endpoints: ask a question, and review the question log (admins)."""

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import get_session
from app.llm import ChatModel, Embedder, GenerationError
from app.models import Question
from app.rag.answer import answer_question
from app.web.auth import AdminDep, UserDep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])

MAX_QUESTION_LENGTH = 1000
MIN_QUESTION_LENGTH = 3
MAX_LOGGED_QUESTIONS = 200

SessionDep = Annotated[Session, Depends(get_session)]


def get_embedder(request: Request) -> Embedder:
    """Provide the app's embedding provider (a FastAPI dependency)."""
    embedder: Embedder = request.app.state.embedder
    return embedder


def get_chat_model(request: Request) -> ChatModel:
    """Provide the app's chat model (a FastAPI dependency)."""
    chat: ChatModel = request.app.state.chat
    return chat


class AskRequest(BaseModel):
    """Body of ``POST /ask``.

    Attributes:
        question: The question in plain language.
    """

    question: str = Field(min_length=MIN_QUESTION_LENGTH, max_length=MAX_QUESTION_LENGTH)


class CitationOut(BaseModel):
    """One cited source.

    Attributes:
        number: The ``[n]`` marker in the answer text.
        document_id: Id of the cited document.
        filename: File name of the cited document.
        page_number: Page of the passage.
        passage: The exact passage text.
    """

    number: int
    document_id: int
    filename: str
    page_number: int
    passage: str


class AskResponse(BaseModel):
    """Answer to a question.

    Attributes:
        question_id: Id of the logged question.
        answer: The answer text, with ``[n]`` citation markers.
        found: False if the documents did not contain the answer.
        citations: The sources behind the answer.
    """

    question_id: int
    answer: str
    found: bool
    citations: list[CitationOut]


class QuestionOut(BaseModel):
    """A logged question, for admin review.

    Attributes:
        id: Question id.
        username: Who asked.
        question: The question text.
        answer: The answer that was shown.
        found: False if no answer was found in the documents.
        created_at: When it was asked (UTC).
    """

    id: int
    username: str
    question: str
    answer: str
    found: bool
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def _as_utc(cls, value: datetime) -> datetime:
        """Mark the time as UTC; SQLite stores it without a timezone."""
        return value if value.tzinfo else value.replace(tzinfo=UTC)


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={status.HTTP_502_BAD_GATEWAY: {"description": "The language model failed."}},
)
def ask(
    body: AskRequest,
    request: Request,
    user: UserDep,
    session: SessionDep,
    embedder: Annotated[Embedder, Depends(get_embedder)],
    chat: Annotated[ChatModel, Depends(get_chat_model)],
) -> AskResponse:
    """Answer a question from the uploaded documents and log it.

    Every question is logged, including ones the documents cannot answer, so
    admins can see what people need that is missing.

    Args:
        body: The question.
        request: Gives access to the application settings.
        user: The logged-in user.
        session: Database session.
        embedder: Provider used to embed the question.
        chat: Chat model that writes the answer.

    Returns:
        The answer with its citations.

    Raises:
        HTTPException: 502 if the language model fails.
    """
    settings: Settings = request.app.state.settings
    try:
        result = answer_question(
            session, embedder, chat, body.question.strip(), settings.answer_top_k
        )
    except GenerationError as error:
        logger.warning("Answer generation failed: %s", error)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error

    logged = Question(
        user_id=user.id,
        question=body.question.strip(),
        answer=result.text,
        found=result.found,
        cited_chunk_ids=",".join(str(citation.chunk_id) for citation in result.citations),
    )
    session.add(logged)
    session.commit()
    return AskResponse(
        question_id=logged.id,
        answer=result.text,
        found=result.found,
        citations=[
            CitationOut(
                number=citation.number,
                document_id=citation.document_id,
                filename=citation.filename,
                page_number=citation.page_number,
                passage=citation.passage,
            )
            for citation in result.citations
        ],
    )


@router.get("/questions", response_model=list[QuestionOut])
def list_questions(_admin: AdminDep, session: SessionDep) -> list[QuestionOut]:
    """Show the most recent questions and answers (admins only).

    Args:
        _admin: Ensures the caller is an admin.
        session: Database session.

    Returns:
        Recent questions, newest first.
    """
    rows = session.scalars(
        select(Question).order_by(Question.id.desc()).limit(MAX_LOGGED_QUESTIONS)
    )
    return [
        QuestionOut(
            id=row.id,
            username=row.user.username,
            question=row.question,
            answer=row.answer,
            found=row.found,
            created_at=row.created_at,
        )
        for row in rows
    ]
