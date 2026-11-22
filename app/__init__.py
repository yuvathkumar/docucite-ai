"""DocuCite AI application package; the app factory lives here."""

import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.sessions import SessionMiddleware

from app.config import Settings, get_settings
from app.db import create_db_engine, create_session_factory, init_db
from app.llm import ChatModel, Embedder, create_chat_model, create_embedder
from app.web.ask import router as ask_router
from app.web.auth import ensure_admin
from app.web.auth import router as auth_router
from app.web.documents import router as documents_router
from app.web.health import router as health_router

logger = logging.getLogger(__name__)

STATIC_DIRECTORY = Path(__file__).parent / "web" / "static"
SESSION_COOKIE_NAME = "docucite_session"
SESSION_MAX_AGE_SECONDS = 14 * 24 * 60 * 60
SECRET_KEY_BYTES = 32


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    chat: ChatModel | None = None,
) -> FastAPI:
    """Build and configure the FastAPI application.

    A factory (instead of a module-level ``app``) lets tests build an app with
    their own settings, database and providers, and keeps import free of side effects.

    Args:
        settings: Settings to use. Defaults to those read from the environment.
        embedder: Embedding provider. Defaults to the one chosen by ``LLM_PROVIDER``.
        chat: Chat model. Defaults to the one chosen by ``LLM_PROVIDER``. Tests pass
            fakes so no real API is ever called.

    Returns:
        The configured application.
    """
    settings = settings or get_settings()
    engine = create_db_engine(settings.database_url)
    session_factory = create_session_factory(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Prepare the database and first admin on startup; release the engine on shutdown."""
        try:
            init_db(engine)
            ensure_admin(session_factory, settings)
        except SQLAlchemyError:
            # Stay up so /health can report the failure with a 503.
            logger.exception("Could not initialise the database at startup")
        yield
        engine.dispose()

    app = FastAPI(title="DocuCite AI", lifespan=lifespan)
    app.state.settings = settings
    app.state.embedder = embedder or create_embedder(settings)
    app.state.chat = chat or create_chat_model(settings)
    app.state.engine = engine
    app.state.session_factory = session_factory

    secret_key = settings.secret_key.get_secret_value()
    if not secret_key:
        logger.warning("SECRET_KEY is empty: using a random key, logins reset on every restart.")
        secret_key = secrets.token_urlsafe(SECRET_KEY_BYTES)
    app.add_middleware(
        SessionMiddleware,
        secret_key=secret_key,
        session_cookie=SESSION_COOKIE_NAME,
        max_age=SESSION_MAX_AGE_SECONDS,
        same_site="lax",
    )

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(documents_router)
    app.include_router(ask_router)
    # Mounted last: it only receives paths that no API route matched.
    app.mount("/", StaticFiles(directory=STATIC_DIRECTORY, html=True), name="ui")
    return app
