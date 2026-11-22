"""Tests that the single-page interface is served and contains its help content."""

from fastapi.testclient import TestClient


def test_interface_is_served_without_login(anonymous_client: TestClient) -> None:
    """Serve the page itself publicly; the API behind it still requires login."""
    response = anonymous_client.get("/")

    assert response.status_code == 200
    assert "DocuCite" in response.text


def test_interface_explains_the_tool(anonymous_client: TestClient) -> None:
    """Include the About page and the question tips, so new users are not left guessing."""
    page = anonymous_client.get("/").text

    assert 'id="view-about"' in page
    assert "Who is it for?" in page
    assert "Tips for good answers" in page
    assert "AI can make mistakes" in page


def test_static_assets_are_served(anonymous_client: TestClient) -> None:
    """Serve the stylesheet and script the page depends on."""
    assert anonymous_client.get("/app.css").status_code == 200
    assert anonymous_client.get("/app.js").status_code == 200
