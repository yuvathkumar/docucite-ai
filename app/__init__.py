"""DocuCite AI application package; the app factory lives here."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.exc import SQLAlchemyError

from app.config import Settings, get_settings
from app.db import create_db_engine, create_session_factory, init_db
from app.web.documents import router as documents_router
from app.web.health import router as health_router

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    A factory (instead of a module-level ``app``) lets tests build an app with
    their own settings and database, and keeps import free of side effects.

    Args:
        settings: Settings to use. Defaults to those read from the environment.

    Returns:
        The configured application.
    """
    settings = settings or get_settings()
    engine = create_db_engine(settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Prepare the database on startup and release it on shutdown."""
        try:
            init_db(engine)
        except SQLAlchemyError:
            # Stay up so /health can report the failure with a 503.
            logger.exception("Could not initialise the database at startup")
        yield
        engine.dispose()

    app = FastAPI(title="DocuCite AI", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    app.include_router(health_router)
    app.include_router(documents_router)
    return app
