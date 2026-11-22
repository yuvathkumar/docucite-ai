"""Ingest a PDF: validate it, extract text per page, chunk it and store it for search."""

import hashlib
import io
import logging
import re
from pathlib import Path

from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.db import index_chunks
from app.models import MAX_FILENAME_LENGTH, Chunk, Document
from app.rag.chunking import split_into_chunks

logger = logging.getLogger(__name__)

PDF_MAGIC_BYTES = b"%PDF-"
BYTES_PER_MB = 1024 * 1024
DEFAULT_FILENAME = "document.pdf"
JUNK_CHARACTERS = re.compile("[-�\U0001f800-\U0001f8ff]")


class IngestError(Exception):
    """Base class for every reason an upload can be rejected."""


class InvalidPdfError(IngestError):
    """The file is not a readable, unencrypted PDF."""


class EmptyPdfError(IngestError):
    """The PDF contains no extractable text (for example a scanned document)."""


class FileTooLargeError(IngestError):
    """The file is larger than the configured upload limit."""


class DuplicateDocumentError(IngestError):
    """A document with identical content was already ingested.

    Attributes:
        document_id: Id of the existing document, if known.
    """

    def __init__(self, document_id: int | None) -> None:
        """Remember which document already exists.

        Args:
            document_id: Id of the existing document, or None if unknown.
        """
        super().__init__("This document has already been uploaded.")
        self.document_id = document_id


def validate_pdf_bytes(data: bytes, max_bytes: int) -> None:
    """Check size and file signature before spending time parsing.

    The signature check looks at the content, not the file name or the
    client-supplied content type, both of which an uploader controls.

    Args:
        data: The uploaded file content.
        max_bytes: Largest accepted size.

    Raises:
        FileTooLargeError: If the file is bigger than ``max_bytes``.
        InvalidPdfError: If the content does not start with the PDF signature.
    """
    if len(data) > max_bytes:
        raise FileTooLargeError(f"The file is larger than {max_bytes // BYTES_PER_MB} MB.")
    if not data.startswith(PDF_MAGIC_BYTES):
        raise InvalidPdfError("The file is not a PDF.")


def clean_page_text(text: str) -> str:
    """Replace junk characters that PDF text extraction produces with spaces.

    Symbol fonts (bullets, arrows) often come out as private-use or replacement
    characters that show as empty boxes in answers and carry no meaning.

    Args:
        text: Raw text of one page.

    Returns:
        The text with those characters turned into spaces.

    Example:
        >>> clean_page_text("Skills  Python � SQL")
        'Skills   Python   SQL'
    """
    return JUNK_CHARACTERS.sub(" ", text)


def extract_pages(data: bytes) -> list[str]:
    """Extract the text of every page of a PDF.

    Args:
        data: The PDF file content.

    Returns:
        One string per page, in page order (empty if a page has no text).

    Raises:
        InvalidPdfError: If the PDF is encrypted, corrupt or cannot be parsed.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise InvalidPdfError("Encrypted PDFs are not supported.")
        return [clean_page_text(page.extract_text() or "") for page in reader.pages]
    except InvalidPdfError:
        raise
    except Exception as error:
        # pypdf raises many different exception types for malformed input, and the
        # file is untrusted, so any parsing failure means "reject this file".
        logger.warning("Could not parse an uploaded PDF: %s", type(error).__name__)
        raise InvalidPdfError("The file could not be read as a PDF.") from error


def build_chunks(pages: list[str], max_chars: int, overlap: int) -> list[Chunk]:
    """Chunk every page and number the chunks across the whole document.

    Chunking happens per page so each chunk keeps one exact page number.

    Args:
        pages: Page texts in page order.
        max_chars: Maximum characters per chunk.
        overlap: Characters repeated between neighbouring chunks.

    Returns:
        Unsaved ``Chunk`` objects with page number and chunk index filled in.
    """
    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        for piece in split_into_chunks(page_text, max_chars=max_chars, overlap=overlap):
            chunks.append(Chunk(page_number=page_number, chunk_index=len(chunks), text=piece))
    return chunks


def clean_filename(filename: str) -> str:
    """Reduce an uploaded file name to a safe display label.

    The name is never used as a path; this only removes any directory part and
    caps the length so it fits the column.

    Args:
        filename: The name sent by the client.

    Returns:
        The bare file name, or a default if nothing usable is left.
    """
    name = Path(filename.replace("\\", "/")).name
    return name[:MAX_FILENAME_LENGTH] or DEFAULT_FILENAME


def ingest_pdf(
    session: Session,
    filename: str,
    data: bytes,
    *,
    max_bytes: int,
    max_chars: int,
    overlap: int,
) -> Document:
    """Validate, parse, chunk and store one PDF, and add it to the search index.

    The document, its chunks and their FTS5 entries are written in one
    transaction, so a failure never leaves a half-ingested document behind.

    Args:
        session: Database session to write with.
        filename: Original file name (display label only).
        data: The PDF file content.
        max_bytes: Largest accepted file size.
        max_chars: Maximum characters per chunk.
        overlap: Characters repeated between neighbouring chunks.

    Returns:
        The saved document.

    Raises:
        FileTooLargeError: If the file exceeds ``max_bytes``.
        InvalidPdfError: If the file is not a readable, unencrypted PDF.
        EmptyPdfError: If no page contains extractable text.
        DuplicateDocumentError: If the same file was ingested before.
    """
    validate_pdf_bytes(data, max_bytes)
    content_hash = hashlib.sha256(data).hexdigest()
    existing = session.scalar(select(Document).where(Document.content_hash == content_hash))
    if existing is not None:
        raise DuplicateDocumentError(existing.id)

    pages = extract_pages(data)
    chunks = build_chunks(pages, max_chars, overlap)
    if not chunks:
        raise EmptyPdfError("No text could be extracted. Scanned PDFs (images) are not supported.")

    document = Document(
        filename=clean_filename(filename),
        content_hash=content_hash,
        page_count=len(pages),
        chunk_count=len(chunks),
        chunks=chunks,
    )
    session.add(document)
    try:
        session.flush()
        index_chunks(session, [(chunk.id, chunk.text) for chunk in document.chunks])
        session.commit()
    except IntegrityError as error:
        # Two identical uploads raced past the check above; the unique hash caught it.
        session.rollback()
        raise DuplicateDocumentError(None) from error
    except SQLAlchemyError:
        session.rollback()
        raise
    return document
