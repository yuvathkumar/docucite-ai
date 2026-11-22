"""Document endpoints: upload a PDF, list documents and (re)create their embeddings."""

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import get_session
from app.llm import Embedder, EmbeddingError
from app.models import Document
from app.rag.embeddings import count_embedded_chunks, embed_missing_chunks
from app.rag.ingest import (
    BYTES_PER_MB,
    DuplicateDocumentError,
    EmptyPdfError,
    FileTooLargeError,
    InvalidPdfError,
    ingest_pdf,
)
from app.web.ask import get_embedder
from app.web.auth import AdminDep, UserDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


SessionDep = Annotated[Session, Depends(get_session)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]


class DocumentOut(BaseModel):
    """Public description of an ingested document.

    Attributes:
        id: Document id.
        filename: Original file name.
        page_count: Number of pages in the PDF.
        chunk_count: Number of text chunks stored for search.
        embedded_chunk_count: How many of those chunks have an embedding vector.
            Lower than ``chunk_count`` means embedding failed or is still pending;
            keyword search works regardless.
        uploaded_at: When the document was ingested (UTC).
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    page_count: int
    chunk_count: int
    embedded_chunk_count: int = 0
    uploaded_at: datetime

    @field_validator("uploaded_at")
    @classmethod
    def _as_utc(cls, value: datetime) -> datetime:
        """Mark the time as UTC; SQLite stores it without a timezone."""
        return value if value.tzinfo else value.replace(tzinfo=UTC)


def _to_out(document: Document, embedded_chunk_count: int) -> DocumentOut:
    """Convert a stored document to its API representation."""
    return DocumentOut.model_validate(document).model_copy(
        update={"embedded_chunk_count": embedded_chunk_count}
    )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=DocumentOut,
    responses={
        status.HTTP_409_CONFLICT: {"description": "The same file was already uploaded."},
        status.HTTP_413_CONTENT_TOO_LARGE: {"description": "File exceeds the upload limit."},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Not a readable PDF with text."},
    },
)
def upload_document(
    request: Request,
    session: SessionDep,
    embedder: EmbedderDep,
    _admin: AdminDep,
    file: UploadFile,
) -> DocumentOut:
    """Ingest an uploaded PDF so its pages become searchable.

    The document is saved and keyword-indexed first. Embedding is a second,
    best-effort step: if the provider is down the upload still succeeds and
    ``embedded_chunk_count`` shows what is missing (retry via the embeddings endpoint).

    There is no access control yet: only admins should be able to upload, which
    arrives with the roles feature.

    Args:
        request: Gives access to the application settings.
        session: Database session for this request.
        embedder: Provider used to create embedding vectors.
        _admin: Ensures the caller is an admin.
        file: The uploaded PDF.

    Returns:
        The stored document.

    Raises:
        HTTPException: 409 for a duplicate, 413 for an oversized file and 422 for
            a file that is not a readable PDF with extractable text.
    """
    settings: Settings = request.app.state.settings
    max_bytes = settings.max_upload_mb * BYTES_PER_MB
    # Read one byte past the limit: enough to detect "too large" without loading it all.
    data = file.file.read(max_bytes + 1)
    try:
        document = ingest_pdf(
            session,
            file.filename or "",
            data,
            max_bytes=max_bytes,
            max_chars=settings.chunk_max_chars,
            overlap=settings.chunk_overlap,
        )
    except FileTooLargeError as error:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(error)) from error
    except DuplicateDocumentError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
    except (InvalidPdfError, EmptyPdfError) as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error

    try:
        embed_missing_chunks(session, embedder, document.id)
    except EmbeddingError as error:
        logger.warning("Document %d saved without embeddings: %s", document.id, error)
    return _to_out(document, count_embedded_chunks(session).get(document.id, 0))


@router.post(
    "/{document_id}/embeddings",
    response_model=DocumentOut,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "No such document."},
        status.HTTP_502_BAD_GATEWAY: {"description": "The embedding provider failed."},
    },
)
def embed_document(
    document_id: int, session: SessionDep, embedder: EmbedderDep, _admin: AdminDep
) -> DocumentOut:
    """Create embeddings for the chunks of a document that lack one.

    Use this to retry after a provider failure, or to embed documents uploaded
    before embeddings existed. Chunks that already have a vector are skipped.

    Args:
        document_id: The document to embed.
        session: Database session for this request.
        embedder: Provider used to create embedding vectors.

    Returns:
        The document with its updated ``embedded_chunk_count``.

    Raises:
        HTTPException: 404 if the document does not exist, 502 if the provider fails.
    """
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    try:
        embed_missing_chunks(session, embedder, document_id)
    except EmbeddingError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
    return _to_out(document, count_embedded_chunks(session).get(document_id, 0))


@router.get("", response_model=list[DocumentOut])
def list_documents(session: SessionDep, _user: UserDep) -> list[DocumentOut]:
    """List all ingested documents, newest first.

    Args:
        session: Database session for this request.

    Returns:
        The documents.
    """
    embedded = count_embedded_chunks(session)
    documents = session.scalars(select(Document).order_by(Document.id.desc()))
    return [_to_out(document, embedded.get(document.id, 0)) for document in documents]
