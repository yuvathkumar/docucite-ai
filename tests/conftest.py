"""Shared pytest fixtures: an app, fakes and logged-in clients on an in-memory SQLite database."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.orm import Session

from app import create_app
from app.config import Settings
from app.db import create_db_engine, create_session_factory, init_db
from app.llm.fake import FakeChatModel, FakeEmbedder

TEST_DATABASE_URL = "sqlite://"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin-password"
LOW_ITERATIONS = 1  # real hashing is deliberately slow; tests do not need that


def make_settings(**overrides: object) -> Settings:
    """Build test settings: in-memory database, a known admin, cheap password hashing."""
    values: dict[str, object] = {
        "database_url": TEST_DATABASE_URL,
        "admin_username": ADMIN_USERNAME,
        "admin_password": SecretStr(ADMIN_PASSWORD),
        "secret_key": SecretStr("test-secret"),
        "password_iterations": LOW_ITERATIONS,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def log_in(client: TestClient, username: str, password: str) -> None:
    """Log ``client`` in; the session cookie then stays on the client."""
    response = client.post("/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text


@pytest.fixture
def settings() -> Settings:
    """Return settings that point at an in-memory database, never the dev file."""
    return make_settings()


@pytest.fixture
def db_session() -> Iterator[Session]:
    """Yield a session on a fresh in-memory database with all tables and FTS5 ready."""
    engine = create_db_engine(TEST_DATABASE_URL)
    init_db(engine)
    with create_session_factory(engine)() as session:
        yield session
    engine.dispose()


@pytest.fixture
def chat() -> FakeChatModel:
    """Return a fake chat model that cites passage [1] by default."""
    return FakeChatModel()


@pytest.fixture
def app(settings: Settings, chat: FakeChatModel) -> FastAPI:
    """Build a fresh application for each test, with fake providers (no network)."""
    return create_app(settings, embedder=FakeEmbedder(), chat=chat)


@pytest.fixture
def anonymous_client(app: FastAPI) -> Iterator[TestClient]:
    """Yield a client that is not logged in; ``with`` runs startup and shutdown."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client(anonymous_client: TestClient) -> TestClient:
    """Return the same client, logged in as the admin."""
    log_in(anonymous_client, ADMIN_USERNAME, ADMIN_PASSWORD)
    return anonymous_client
