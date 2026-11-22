"""Tests for the ``GET /health`` endpoint: healthy, database down, FTS5 missing."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.db import create_db_engine

UNREACHABLE_DATABASE_URL = "sqlite:////this-folder-does-not-exist/docucite.db"


def test_health_returns_200_when_everything_works(client: TestClient) -> None:
    """Report HTTP 200 and all checks "ok" when the database and FTS5 work."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok", "fts5": "ok"}


def test_health_returns_503_when_database_is_unreachable(app: FastAPI, client: TestClient) -> None:
    """Report HTTP 503 with every check failed when the database cannot be reached."""
    # Swap in an engine whose file can never be opened, instead of touching a real database.
    app.state.engine = create_engine(UNREACHABLE_DATABASE_URL)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "error", "database": "error", "fts5": "error"}


def test_health_returns_503_when_fts5_table_is_missing(app: FastAPI, client: TestClient) -> None:
    """Report HTTP 503 with only ``fts5`` failed when the keyword index table is absent."""
    # A healthy but empty database: SELECT 1 works, the FTS5 table was never created.
    app.state.engine = create_db_engine("sqlite://")

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "error", "database": "ok", "fts5": "error"}
