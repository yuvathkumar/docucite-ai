"""Tests for login, logout, roles and user management."""

from fastapi.testclient import TestClient

from app.web.auth import hash_password, verify_password
from tests.conftest import ADMIN_PASSWORD, ADMIN_USERNAME, log_in
from tests.pdf_factory import make_pdf

NEW_PASSWORD = "correct-horse"


def make_user(client: TestClient, username: str = "alice", role: str = "user") -> None:
    """Create a user through the API (the client must be logged in as an admin)."""
    response = client.post(
        "/users", json={"username": username, "password": NEW_PASSWORD, "role": role}
    )
    assert response.status_code == 201, response.text


def test_password_hash_verifies_and_is_salted() -> None:
    """Accept the right password, reject others, and never repeat a hash."""
    first = hash_password("secret", 1)
    second = hash_password("secret", 1)

    assert verify_password("secret", first)
    assert not verify_password("wrong", first)
    assert first != second
    assert "secret" not in first


def test_malformed_stored_hash_is_rejected() -> None:
    """Treat a corrupt hash as a failed login instead of crashing."""
    assert not verify_password("x", "not-a-hash")
    assert not verify_password("x", "pbkdf2_sha256$abc$zz$zz")
    assert not verify_password("x", "md5$1$00$00")


def test_first_admin_is_created_from_settings(anonymous_client: TestClient) -> None:
    """Allow logging in as the admin defined in the settings."""
    response = anonymous_client.post(
        "/auth/login", json={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD}
    )

    assert response.status_code == 200
    assert response.json() == {"id": 1, "username": ADMIN_USERNAME, "role": "admin"}


def test_wrong_password_and_unknown_user_get_the_same_error(anonymous_client: TestClient) -> None:
    """Do not reveal which usernames exist."""
    wrong = anonymous_client.post(
        "/auth/login", json={"username": ADMIN_USERNAME, "password": "nope"}
    )
    unknown = anonymous_client.post("/auth/login", json={"username": "ghost", "password": "nope"})

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_me_requires_login_and_logout_ends_the_session(client: TestClient) -> None:
    """Return the current user while logged in and 401 after logging out."""
    assert client.get("/auth/me").json()["username"] == ADMIN_USERNAME

    assert client.post("/auth/logout").status_code == 204

    assert client.get("/auth/me").status_code == 401


def test_endpoints_require_login(anonymous_client: TestClient) -> None:
    """Return 401 for every protected endpoint when not logged in."""
    assert anonymous_client.get("/documents").status_code == 401
    assert anonymous_client.post("/ask", json={"question": "anything"}).status_code == 401
    assert anonymous_client.get("/questions").status_code == 401
    assert anonymous_client.get("/users").status_code == 401


def test_health_stays_public(anonymous_client: TestClient) -> None:
    """Let monitors call ``/health`` without logging in."""
    assert anonymous_client.get("/health").status_code == 200


def test_normal_user_can_read_but_not_manage(client: TestClient) -> None:
    """Allow asking and listing, but forbid uploads, embedding, users and the log."""
    make_user(client, "alice")
    client.post("/auth/logout")
    log_in(client, "alice", NEW_PASSWORD)
    files = {"file": ("a.pdf", make_pdf(["Some text."]), "application/pdf")}

    assert client.get("/documents").status_code == 200
    assert client.post("/documents", files=files).status_code == 403
    assert client.post("/documents/1/embeddings").status_code == 403
    assert client.get("/users").status_code == 403
    assert client.get("/questions").status_code == 403
    assert (
        client.post("/users", json={"username": "bob", "password": NEW_PASSWORD}).status_code == 403
    )


def test_admin_can_create_and_list_users(client: TestClient) -> None:
    """Show new users without any password information."""
    make_user(client, "alice")

    users = client.get("/users").json()

    assert [user["username"] for user in users] == [ADMIN_USERNAME, "alice"]
    assert all(set(user) == {"id", "username", "role"} for user in users)


def test_duplicate_username_returns_409(client: TestClient) -> None:
    """Refuse a second account with the same name."""
    make_user(client, "alice")

    response = client.post("/users", json={"username": "alice", "password": NEW_PASSWORD})

    assert response.status_code == 409


def test_weak_or_invalid_user_input_is_rejected(client: TestClient) -> None:
    """Validate username format, password length and role."""
    assert (
        client.post("/users", json={"username": "ab", "password": NEW_PASSWORD}).status_code == 422
    )
    assert client.post("/users", json={"username": "bob", "password": "short"}).status_code == 422
    bad_role = {"username": "bob", "password": NEW_PASSWORD, "role": "root"}
    assert client.post("/users", json=bad_role).status_code == 422
