"""SQLAlchemy table definitions: documents and their text chunks."""

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

MAX_FILENAME_LENGTH = 255
MAX_MODEL_NAME_LENGTH = 100
MAX_USERNAME_LENGTH = 50
PASSWORD_HASH_LENGTH = 255
ROLE_LENGTH = 10
SHA256_HEX_LENGTH = 64


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Common base class that collects all table definitions."""


class User(Base):
    """A person who can log in.

    Attributes:
        id: Primary key.
        username: Unique login name.
        password_hash: Salted PBKDF2 hash; the password itself is never stored.
        role: ``"admin"`` (manages documents and users) or ``"user"`` (asks questions).
        created_at: When the account was created (UTC).
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(MAX_USERNAME_LENGTH), unique=True)
    password_hash: Mapped[str] = mapped_column(String(PASSWORD_HASH_LENGTH))
    role: Mapped[str] = mapped_column(String(ROLE_LENGTH))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class Question(Base):
    """One question asked and the answer given, kept for review.

    Attributes:
        id: Primary key.
        user_id: Who asked.
        question: The question text.
        answer: The answer shown to the user.
        found: False if the documents did not contain the answer.
        cited_chunk_ids: Comma-separated ids of the chunks cited in the answer.
        created_at: When it was asked (UTC).
        user: The asking user.
    """

    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    found: Mapped[bool]
    cited_chunk_ids: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    user: Mapped[User] = relationship()


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
    embedding: Mapped["ChunkEmbedding | None"] = relationship(
        back_populates="chunk", cascade="all, delete-orphan"
    )


class ChunkEmbedding(Base):
    """The embedding vector of one chunk.

    Kept in its own table so a chunk is searchable by keyword before it has a
    vector, and so existing databases need no migration.

    Attributes:
        chunk_id: The chunk this vector belongs to (also the primary key).
        model: Embedding model that produced the vector. Vectors from different
            models cannot be compared, so this lets us detect a model change.
        dimensions: Number of values in the vector.
        vector: The values packed as 32-bit floats (see ``app.rag.embeddings``).
        chunk: The owning chunk.
    """

    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id"), primary_key=True)
    model: Mapped[str] = mapped_column(String(MAX_MODEL_NAME_LENGTH))
    dimensions: Mapped[int]
    vector: Mapped[bytes] = mapped_column(LargeBinary)

    chunk: Mapped[Chunk] = relationship(back_populates="embedding")
