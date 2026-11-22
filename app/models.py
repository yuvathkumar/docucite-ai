"""SQLAlchemy table definitions: documents and their text chunks."""

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

MAX_FILENAME_LENGTH = 255
SHA256_HEX_LENGTH = 64


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Common base class that collects all table definitions."""


class Document(Base):
    """One uploaded PDF.

    Only the extracted text is stored (as chunks), not the file itself: citations
    need the page and the passage, not the original bytes.

    Attributes:
        id: Primary key.
        filename: Original file name, kept as a display label only.
        content_hash: SHA-256 of the file bytes; unique, so re-uploading the same
            PDF is detected instead of duplicating its chunks.
        page_count: Number of pages in the PDF.
        chunk_count: Number of chunks stored (kept here to avoid a COUNT query
            when listing documents).
        uploaded_at: When the document was ingested (UTC).
        chunks: The text chunks of this document.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(MAX_FILENAME_LENGTH))
    content_hash: Mapped[str] = mapped_column(String(SHA256_HEX_LENGTH), unique=True)
    page_count: Mapped[int]
    chunk_count: Mapped[int]
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """A passage of text from one page of a document.

    Every chunk belongs to exactly one page, so a citation can always name the
    exact document and page.

    Attributes:
        id: Primary key; also the ``chunk_id`` stored in the FTS5 index.
        document_id: The document this chunk came from.
        page_number: 1-based page number inside the PDF.
        chunk_index: Position of the chunk within the whole document.
        text: The chunk text.
        document: The owning document.
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    page_number: Mapped[int]
    chunk_index: Mapped[int]
    text: Mapped[str] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="chunks")
