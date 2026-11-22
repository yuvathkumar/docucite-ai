"""Document endpoints: upload a PDF and list the ingested documents."""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import get_session
from app.models import Document
from app.rag.ingest import (
    BYTES_PER_MB,
    DuplicateDocumentError,
    EmptyPdfError,
    FileTooLargeError,
    InvalidPdfError,
    ingest_pdf,
)

router = APIRouter(prefix="/documents", tags=["documents"])

SessionDep = Annotated[Session, Depends(get_session)]


class DocumentOut(BaseModel):
    """Public description of an ingested document.

    Attributes:
        id: Document id.
        filename: Original file name.
        page_count: Number of pages in the PDF.
        chunk_count: Number of text chunks stored for search.
        uploaded_at: When the document was ingested (UTC).
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    page_count: int
    chunk_count: int
    uploaded_at: datetime

    @field_validator("uploaded_at")
    @classmethod
    def _as_utc(cls, value: datetime) -> datetime:
        """Mark the time as UTC; SQLite stores it without a timezone."""
        return value if value.tzinfo else value.replace(tzinfo=UTC)


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
def upload_document(request: Request, session: SessionDep, file: UploadFile) -> Document:
    """Ingest an uploaded PDF so its pages become searchable.

    There is no access control yet: only admins should be able to upload, which
    arrives with the roles feature.

    Args:
        request: Gives access to the application settings.
        session: Database session for this request.
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
        return ingest_pdf(
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


@router.get("", response_model=list[DocumentOut])
def list_documents(session: SessionDep) -> list[Document]:
    """List all ingested documents, newest first.

    Args:
        session: Database session for this request.

    Returns:
        The documents.
    """
    return list(session.scalars(select(Document).order_by(Document.id.desc())))
