"""Shared pytest fixtures: an app and client backed by an in-memory SQLite database."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app import create_app
from app.config import Settings
from app.db import create_db_engine, create_session_factory, init_db
from app.llm.fake import FakeEmbedder

TEST_DATABASE_URL = "sqlite://"


@pytest.fixture
def settings() -> Settings:
    """Return settings that point at an in-memory database, never the dev file."""
    return Settings(database_url=TEST_DATABASE_URL, _env_file=None)


@pytest.fixture
def db_session() -> Iterator[Session]:
    """Yield a session on a fresh in-memory database with all tables and FTS5 ready."""
    engine = create_db_engine(TEST_DATABASE_URL)
    init_db(engine)
    with create_session_factory(engine)() as session:
        yield session
    engine.dispose()


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    """Build a fresh application for each test, with a fake embedder (no network)."""
    return create_app(settings, embedder=FakeEmbedder())


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Yield a test client; the ``with`` block runs startup and shutdown (the lifespan)."""
    with TestClient(app) as test_client:
        yield test_client
