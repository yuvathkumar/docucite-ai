"""Health endpoint: reports whether the app, database and FTS5 are working."""

from typing import Literal

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import Engine

from app.db import check_database, check_fts5

router = APIRouter()

CheckResult = Literal["ok", "error"]


class HealthStatus(BaseModel):
    """Response body of ``GET /health``.

    Attributes:
        status: ``"ok"`` only if every individual check passed.
        database: Result of a ``SELECT 1`` against the database.
        fts5: Result of a search on the FTS5 keyword index.
    """

    status: CheckResult
    database: CheckResult
    fts5: CheckResult


def _to_result(passed: bool) -> CheckResult:
    """Translate a boolean check into the ``"ok"``/``"error"`` wording of the API."""
    return "ok" if passed else "error"


@router.get(
    "/health",
    response_model=HealthStatus,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthStatus}},
)
def health(request: Request, response: Response) -> HealthStatus:
    """Report the health of the app and its dependencies.

    Load balancers and monitors call this to decide whether to send traffic,
    so a broken database must produce HTTP 503, not a 200 with bad news.

    Args:
        request: Gives access to the engine stored on the app.
        response: Used to switch the status code to 503 on failure.

    Returns:
        The per-check results.
    """
    engine: Engine = request.app.state.engine
    database_ok = check_database(engine)
    fts5_ok = database_ok and check_fts5(engine)
    if not (database_ok and fts5_ok):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthStatus(
        status=_to_result(database_ok and fts5_ok),
        database=_to_result(database_ok),
        fts5=_to_result(fts5_ok),
    )
