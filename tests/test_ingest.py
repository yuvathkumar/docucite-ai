"""Tests for PDF ingestion: extraction, storage, FTS5 indexing and rejection rules."""

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.llm.fake import FakeEmbedder
from app.models import Chunk, ChunkEmbedding, Document
from app.rag.embeddings import embed_missing_chunks
from app.rag.ingest import (
    DuplicateDocumentError,
    EmptyPdfError,
    FileTooLargeError,
    InvalidPdfError,
    clean_filename,
    clean_page_text,
    delete_document,
    ingest_pdf,
)
from tests.pdf_factory import make_pdf

MAX_BYTES = 1024 * 1024
MAX_CHARS = 200
OVERLAP = 20


def ingest(session: Session, data: bytes, filename: str = "policy.pdf") -> Document:
    """Ingest ``data`` with the test limits."""
    return ingest_pdf(
        session, filename, data, max_bytes=MAX_BYTES, max_chars=MAX_CHARS, overlap=OVERLAP
    )


def test_ingest_stores_document_and_chunks_with_page_numbers(db_session: Session) -> None:
    """Save one chunk per page with the correct 1-based page number."""
    pdf = make_pdf(["Vacation policy for all staff.", "Expense reports are due monthly."])

    document = ingest(db_session, pdf)

    assert document.page_count == 2
    assert document.chunk_count == 2
    chunks = db_session.scalars(select(Chunk).order_by(Chunk.chunk_index)).all()
    assert [(chunk.page_number, chunk.text) for chunk in chunks] == [
        (1, "Vacation policy for all staff."),
        (2, "Expense reports are due monthly."),
    ]


def test_ingested_text_is_searchable_with_fts5(db_session: Session) -> None:
    """Find the right chunk through the FTS5 keyword index."""
    ingest(db_session, make_pdf(["Vacation policy for all staff.", "Expense reports monthly."]))

    rows = db_session.execute(
        text("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH :query"),
        {"query": "expense"},
    ).all()

    assert len(rows) == 1
    chunk = db_session.get(Chunk, rows[0][0])
    assert chunk is not None
    assert chunk.page_number == 2


def test_long_page_is_split_into_several_chunks_on_the_same_page(db_session: Session) -> None:
    """Keep every chunk of a long page tied to that page."""
    long_page = " ".join(f"word{number}" for number in range(120))

    document = ingest(db_session, make_pdf([long_page]))

    assert document.chunk_count > 1
    assert {chunk.page_number for chunk in document.chunks} == {1}


def test_duplicate_upload_is_rejected_and_adds_nothing(db_session: Session) -> None:
    """Refuse a second upload of identical bytes."""
    pdf = make_pdf(["Same content."])
    first = ingest(db_session, pdf)

    with pytest.raises(DuplicateDocumentError) as error:
        ingest(db_session, pdf, filename="renamed.pdf")

    assert error.value.document_id == first.id
    assert len(db_session.scalars(select(Document)).all()) == 1


def test_clean_page_text_removes_symbol_font_junk() -> None:
    """Turn private-use, replacement and arrow-symbol characters into spaces."""
    assert clean_page_text("AB�C\U0001f83fD") == "A B C D"
    assert clean_page_text("Plain text stays.") == "Plain text stays."


def test_delete_document_removes_chunks_embeddings_and_index(db_session: Session) -> None:
    """Leave no trace of a deleted document anywhere, including the FTS5 index."""
    document = ingest(db_session, make_pdf(["Quokka handling rules."]))
    embed_missing_chunks(db_session, FakeEmbedder(), document.id)

    assert delete_document(db_session, document.id) is True

    assert db_session.scalars(select(Document)).all() == []
    assert db_session.scalars(select(Chunk)).all() == []
    assert db_session.scalars(select(ChunkEmbedding)).all() == []
    hits = db_session.execute(
        text("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'quokka'")
    ).all()
    assert hits == []


def test_delete_missing_document_returns_false(db_session: Session) -> None:
    """Report that there was nothing to delete."""
    assert delete_document(db_session, 12345) is False


def test_non_pdf_content_is_rejected(db_session: Session) -> None:
    """Judge the file by its content, not by its name."""
    with pytest.raises(InvalidPdfError, match="not a PDF"):
        ingest(db_session, b"just some text", filename="fake.pdf")


def test_corrupt_pdf_is_rejected(db_session: Session) -> None:
    """Reject a file with a PDF signature but a broken body."""
    with pytest.raises(InvalidPdfError):
        ingest(db_session, b"%PDF-1.4\nthis is not really a pdf")


def test_oversized_file_is_rejected(db_session: Session) -> None:
    """Enforce the upload size limit."""
    with pytest.raises(FileTooLargeError):
        ingest_pdf(
            db_session,
            "big.pdf",
            make_pdf(["text"]),
            max_bytes=10,
            max_chars=MAX_CHARS,
            overlap=OVERLAP,
        )


def test_pdf_without_text_is_rejected(db_session: Session) -> None:
    """Explain that scanned (image-only) PDFs are not supported."""
    with pytest.raises(EmptyPdfError, match="Scanned"):
        ingest(db_session, make_pdf(["", ""]))
    assert db_session.scalars(select(Document)).all() == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\scan.pdf", "scan.pdf"),
        ("", "document.pdf"),
        ("a" * 400 + ".pdf", "a" * 255),
    ],
)
def test_clean_filename_removes_directories_and_limits_length(raw: str, expected: str) -> None:
    """Keep only a short, bare file name as the display label."""
    assert clean_filename(raw) == expected
