"""Tests for the ``/documents`` endpoints."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from app import create_app
from app.config import Settings
from app.llm import UnavailableEmbedder
from app.llm.fake import FakeEmbedder
from tests.pdf_factory import make_pdf

PDF_TYPE = "application/pdf"
BYTES_PER_MB = 1024 * 1024


def upload(client: TestClient, data: bytes, filename: str = "policy.pdf") -> Response:
    """POST ``data`` to ``/documents`` as a multipart file."""
    return client.post("/documents", files={"file": (filename, data, PDF_TYPE)})


def test_upload_returns_201_and_document_details(client: TestClient) -> None:
    """Create a document and describe it in the response."""
    response = upload(client, make_pdf(["Vacation policy.", "Expense policy."]))

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "policy.pdf"
    assert body["page_count"] == 2
    assert body["chunk_count"] == 2
    assert "id" in body


def test_uploaded_document_appears_in_the_list(client: TestClient) -> None:
    """Return uploaded documents from ``GET /documents``."""
    upload(client, make_pdf(["First."]), filename="first.pdf")
    upload(client, make_pdf(["Second."]), filename="second.pdf")

    response = client.get("/documents")

    assert response.status_code == 200
    assert [item["filename"] for item in response.json()] == ["second.pdf", "first.pdf"]


def test_upload_time_is_reported_in_utc_everywhere(client: TestClient) -> None:
    """Show the same UTC timestamp format on upload and in the list."""
    created = upload(client, make_pdf(["Timestamps."])).json()["uploaded_at"]
    listed = client.get("/documents").json()[0]["uploaded_at"]

    assert created.endswith("Z")
    assert listed == created


def test_list_is_empty_before_any_upload(client: TestClient) -> None:
    """Return an empty list, not an error, when nothing was uploaded."""
    assert client.get("/documents").json() == []


def test_duplicate_upload_returns_409(client: TestClient) -> None:
    """Tell the user when the same file is uploaded twice."""
    pdf = make_pdf(["Same content."])
    upload(client, pdf)

    response = upload(client, pdf, filename="again.pdf")

    assert response.status_code == 409


def test_non_pdf_upload_returns_422(client: TestClient) -> None:
    """Reject a text file even if it claims to be a PDF."""
    response = upload(client, b"hello world", filename="notes.pdf")

    assert response.status_code == 422
    assert "not a PDF" in response.json()["detail"]


def test_pdf_without_text_returns_422(client: TestClient) -> None:
    """Reject a PDF whose pages contain no extractable text."""
    response = upload(client, make_pdf([""]))

    assert response.status_code == 422


def test_missing_file_returns_422(client: TestClient) -> None:
    """Let FastAPI reject a request that has no file part."""
    assert client.post("/documents").status_code == 422


def test_oversized_upload_returns_413() -> None:
    """Refuse files above ``MAX_UPLOAD_MB``."""
    settings = Settings(database_url="sqlite://", max_upload_mb=1, _env_file=None)
    too_big = b"%PDF-" + b"0" * BYTES_PER_MB

    with TestClient(create_app(settings, embedder=FakeEmbedder())) as client:
        response = upload(client, too_big)

    assert response.status_code == 413


def test_upload_creates_embeddings_for_every_chunk(client: TestClient) -> None:
    """Report all chunks as embedded when the provider works."""
    body = upload(client, make_pdf(["Vacation policy.", "Expense policy."])).json()

    assert body["chunk_count"] == 2
    assert body["embedded_chunk_count"] == 2
    assert client.get("/documents").json()[0]["embedded_chunk_count"] == 2


def test_upload_succeeds_when_the_embedding_provider_is_down() -> None:
    """Keep the document (and keyword search) even if embedding fails."""
    settings = Settings(database_url="sqlite://", _env_file=None)
    down = UnavailableEmbedder("provider down")

    with TestClient(create_app(settings, embedder=down)) as client:
        response = upload(client, make_pdf(["Still searchable by keyword."]))

    assert response.status_code == 201
    assert response.json()["chunk_count"] == 1
    assert response.json()["embedded_chunk_count"] == 0


def test_embeddings_endpoint_retries_after_a_failure(app: FastAPI, client: TestClient) -> None:
    """Fill in missing vectors once the provider is back."""
    app.state.embedder = UnavailableEmbedder("provider down")
    document_id = upload(client, make_pdf(["Retry me."])).json()["id"]
    assert client.get("/documents").json()[0]["embedded_chunk_count"] == 0

    app.state.embedder = FakeEmbedder()
    response = client.post(f"/documents/{document_id}/embeddings")

    assert response.status_code == 200
    assert response.json()["embedded_chunk_count"] == 1


def test_embeddings_endpoint_returns_404_for_unknown_document(client: TestClient) -> None:
    """Report a missing document instead of failing silently."""
    assert client.post("/documents/999/embeddings").status_code == 404


def test_embeddings_endpoint_returns_502_when_provider_fails(
    app: FastAPI, client: TestClient
) -> None:
    """Tell the caller the upstream provider failed, not that they sent a bad request."""
    app.state.embedder = UnavailableEmbedder("provider down")
    document_id = upload(client, make_pdf(["Needs a vector."])).json()["id"]

    response = client.post(f"/documents/{document_id}/embeddings")

    assert response.status_code == 502
    assert "provider down" in response.json()["detail"]
