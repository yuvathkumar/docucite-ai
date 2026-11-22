"""Database engine, sessions and SQLite FTS5 setup."""

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from fastapi import Request
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base

logger = logging.getLogger(__name__)

FTS_TABLE_NAME = "chunks_fts"
IN_MEMORY_DATABASE = ":memory:"


def create_db_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine for a SQLite URL.

    FastAPI runs sync endpoints in a thread pool, so SQLite's "same thread
    only" check is turned off. An in-memory database exists per connection,
    so it gets a single shared connection (``StaticPool``); without that,
    every new connection would see an empty database.

    Args:
        database_url: A ``sqlite:///...`` URL.

    Returns:
        A ready-to-use engine. The parent folder of a file database is created.
    """
    url = make_url(database_url)
    in_memory = url.database in (None, "", IN_MEMORY_DATABASE)
    if not in_memory and url.database:
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool if in_memory else None,
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a factory that opens database sessions bound to ``engine``.

    Args:
        engine: The engine sessions should use.

    Returns:
        A ``sessionmaker``; call it to get a new ``Session``.
    """
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_fts(engine: Engine) -> None:
    """Create the FTS5 keyword-search table if it does not exist yet.

    FTS5 is SQLite's built-in full-text index. It is a virtual table, so
    SQLAlchemy models cannot describe it and we create it with plain SQL.
    ``chunk_id`` is stored but not indexed: it links a hit back to its chunk row.

    Args:
        engine: Engine of the database to prepare.
    """
    with engine.begin() as connection:
        connection.execute(
            text(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE_NAME} "
                "USING fts5(text, chunk_id UNINDEXED)"
            )
        )


def init_db(engine: Engine) -> None:
    """Create all tables and the FTS5 index if they do not exist yet.

    Args:
        engine: Engine of the database to prepare.
    """
    Base.metadata.create_all(engine)
    init_fts(engine)


def get_session(request: Request) -> Iterator[Session]:
    """Provide one database session per request (a FastAPI dependency).

    The ``with`` block closes the session even if the endpoint raises, so
    connections are never leaked.

    Args:
        request: The current request; carries the app's session factory.

    Yields:
        An open session.
    """
    factory: sessionmaker[Session] = request.app.state.session_factory
    with factory() as session:
        yield session


def index_chunks(session: Session, chunks: Iterable[tuple[int, str]]) -> None:
    """Add chunks to the FTS5 keyword index.

    The caller commits, so the chunk rows and their index entries are saved
    together or not at all.

    Args:
        session: Session whose transaction the inserts join.
        chunks: Pairs of ``(chunk_id, text)``.
    """
    session.execute(
        text(f"INSERT INTO {FTS_TABLE_NAME} (text, chunk_id) VALUES (:text, :chunk_id)"),
        [{"text": chunk_text, "chunk_id": chunk_id} for chunk_id, chunk_text in chunks],
    )


def check_database(engine: Engine) -> bool:
    """Check that the database accepts a trivial query.

    Args:
        engine: Engine to test.

    Returns:
        True if ``SELECT 1`` succeeds, False on any database error.
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        logger.exception("Database check failed")
        return False
    return True


def check_fts5(engine: Engine) -> bool:
    """Check that the FTS5 table exists and can be searched.

    Args:
        engine: Engine to test.

    Returns:
        True if a search on the FTS5 table succeeds, False on any database error.
    """
    try:
        with engine.connect() as connection:
            connection.execute(
                text(f"SELECT count(*) FROM {FTS_TABLE_NAME} WHERE {FTS_TABLE_NAME} MATCH 'x'")
            )
    except SQLAlchemyError:
        logger.exception("FTS5 check failed")
        return False
    return True
