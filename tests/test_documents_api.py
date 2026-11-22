"""Tests for the ``/documents`` endpoints."""

from fastapi.testclient import TestClient
from httpx import Response

from app import create_app
from app.config import Settings
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

    with TestClient(create_app(settings)) as client:
        response = upload(client, too_big)

    assert response.status_code == 413
