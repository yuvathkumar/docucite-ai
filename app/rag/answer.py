"""Answer a question from the stored documents, with citations or an honest "not found"."""

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.llm.base import ChatModel, Embedder
from app.rag.prompts import (
    NOT_FOUND_TOKEN,
    SYSTEM_PROMPT,
    build_user_prompt,
    extract_citation_numbers,
    remove_invalid_citations,
)
from app.rag.search import SearchHit, hybrid_search

NOT_FOUND_MESSAGE = "I could not find the answer to that in the uploaded documents."


@dataclass(frozen=True)
class Citation:
    """A source the answer relies on.

    Attributes:
        number: The ``[n]`` marker used in the answer text.
        chunk_id: Id of the cited chunk.
        document_id: Id of the cited document.
        filename: File name of the cited document.
        page_number: Page of the cited passage.
        passage: The exact passage text.
    """

    number: int
    chunk_id: int
    document_id: int
    filename: str
    page_number: int
    passage: str


@dataclass(frozen=True)
class Answer:
    """The result of asking a question.

    Attributes:
        text: The answer shown to the user.
        found: False when the documents did not contain the answer.
        citations: Sources behind the answer; empty when ``found`` is False.
    """

    text: str
    found: bool
    citations: list[Citation] = field(default_factory=list)


def _not_found() -> Answer:
    """Build the standard "could not find it" answer."""
    return Answer(text=NOT_FOUND_MESSAGE, found=False)


def _to_citation(number: int, hit: SearchHit) -> Citation:
    """Convert a search hit into a citation with the given number."""
    return Citation(
        number=number,
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        filename=hit.filename,
        page_number=hit.page_number,
        passage=hit.text,
    )


def answer_question(
    session: Session,
    embedder: Embedder,
    chat: ChatModel,
    question: str,
    top_k: int,
) -> Answer:
    """Retrieve passages and have the model answer from them, citing its sources.

    An answer is only returned if it cites at least one real passage. A reply
    with no valid citation is discarded, because showing uncited claims would
    break the product's promise. The model is not called at all when search
    finds nothing.

    Args:
        session: Database session.
        embedder: Provider used to embed the question.
        chat: Chat model that writes the answer.
        question: The user's question.
        top_k: Number of passages to give the model.

    Returns:
        A cited answer, or a "not found" answer.

    Raises:
        GenerationError: If the chat model fails.
    """
    hits = hybrid_search(session, embedder, question, top_k)
    if not hits:
        return _not_found()

    reply = chat.generate(SYSTEM_PROMPT, build_user_prompt(question, hits)).strip()
    if reply.upper().startswith(NOT_FOUND_TOKEN):
        return _not_found()
    numbers = extract_citation_numbers(reply, len(hits))
    if not numbers:
        return _not_found()
    citations = [_to_citation(number, hits[number - 1]) for number in numbers]
    return Answer(text=remove_invalid_citations(reply, len(hits)), found=True, citations=citations)
