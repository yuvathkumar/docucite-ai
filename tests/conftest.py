"""Shared pytest fixtures: an app and client backed by an in-memory SQLite database."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import create_app
from app.config import Settings

TEST_DATABASE_URL = "sqlite://"


@pytest.fixture
def settings() -> Settings:
    """Return settings that point at an in-memory database, never the dev file."""
    return Settings(database_url=TEST_DATABASE_URL, _env_file=None)


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    """Build a fresh application for each test."""
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Yield a test client; the ``with`` block runs startup and shutdown (the lifespan)."""
    with TestClient(app) as test_client:
        yield test_client
