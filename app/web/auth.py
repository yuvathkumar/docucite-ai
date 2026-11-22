"""Authentication and roles: password hashing, login/logout, and user management."""

import hashlib
import hmac
import logging
import secrets
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db import get_session
from app.models import User

logger = logging.getLogger(__name__)

HASH_SCHEME = "pbkdf2_sha256"
SALT_BYTES = 16
SESSION_USER_KEY = "user_id"
MIN_PASSWORD_LENGTH = 8
USERNAME_PATTERN = r"^[A-Za-z0-9_.-]{3,50}$"
ADMIN_ROLE = "admin"
USER_ROLE = "user"

router = APIRouter(tags=["auth"])


def hash_password(password: str, iterations: int) -> str:
    """Hash a password with a random salt using PBKDF2-HMAC-SHA256.

    A slow, salted hash means a stolen database does not reveal passwords and
    identical passwords do not produce identical hashes.

    Args:
        password: The plain-text password.
        iterations: PBKDF2 work factor.

    Returns:
        A string ``pbkdf2_sha256$iterations$salt$hash`` safe to store.

    Example:
        >>> verify_password("secret", hash_password("secret", 1))
        True
    """
    salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"{HASH_SCHEME}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check a password against a stored hash in constant time.

    Args:
        password: The plain-text password to check.
        stored: A value produced by :func:`hash_password`.

    Returns:
        True if the password matches; False for a wrong password or a malformed hash.
    """
    try:
        scheme, iterations, salt_hex, digest_hex = stored.split("$")
        if scheme != HASH_SCHEME:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(digest, bytes.fromhex(digest_hex))
    except ValueError:
        return False


def ensure_admin(session_factory: sessionmaker[Session], settings: Settings) -> None:
    """Create the first admin from settings when no users exist yet.

    Without this there would be no way to log in to a fresh install. The
    password comes from ``.env``, never from the code.

    Args:
        session_factory: Factory for database sessions.
        settings: Application settings with the admin username and password.
    """
    password = settings.admin_password.get_secret_value()
    with session_factory() as session:
        if session.scalar(select(func.count(User.id))):
            return
        if not password:
            logger.warning("No users exist and ADMIN_PASSWORD is empty: nobody can log in.")
            return
        session.add(
            User(
                username=settings.admin_username,
                password_hash=hash_password(password, settings.password_iterations),
                role=ADMIN_ROLE,
            )
        )
        session.commit()
        logger.info("Created the first admin user %r", settings.admin_username)


SessionDep = Annotated[Session, Depends(get_session)]


def current_user(request: Request, session: SessionDep) -> User:
    """Return the logged-in user (a FastAPI dependency).

    Args:
        request: The request, whose signed cookie holds the user id.
        session: Database session.

    Returns:
        The user.

    Raises:
        HTTPException: 401 if nobody is logged in.
    """
    user_id = request.session.get(SESSION_USER_KEY)
    user = session.get(User, user_id) if user_id is not None else None
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Please log in.")
    return user


UserDep = Annotated[User, Depends(current_user)]


def require_admin(user: UserDep) -> User:
    """Return the logged-in user only if they are an admin (a FastAPI dependency).

    Args:
        user: The logged-in user.

    Returns:
        The admin user.

    Raises:
        HTTPException: 403 if the user is not an admin.
    """
    if user.role != ADMIN_ROLE:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admins only.")
    return user


AdminDep = Annotated[User, Depends(require_admin)]


class LoginRequest(BaseModel):
    """Body of ``POST /auth/login``.

    Attributes:
        username: Login name.
        password: Plain-text password (sent over the connection, never stored).
    """

    username: str = Field(max_length=100)
    password: str = Field(max_length=200)


class UserOut(BaseModel):
    """Public description of a user (never includes the password hash).

    Attributes:
        id: User id.
        username: Login name.
        role: ``"admin"`` or ``"user"``.
    """

    id: int
    username: str
    role: str


class NewUser(BaseModel):
    """Body of ``POST /users``.

    Attributes:
        username: Login name (3-50 letters, digits, ``_``, ``.`` or ``-``).
        password: At least 8 characters.
        role: ``"admin"`` or ``"user"``.
    """

    username: str = Field(pattern=USERNAME_PATTERN)
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=200)
    role: Literal["admin", "user"] = USER_ROLE


@router.post("/auth/login", response_model=UserOut)
def login(body: LoginRequest, request: Request, session: SessionDep) -> User:
    """Log in and start a session.

    The error message is the same for an unknown user and a wrong password, and
    the hash is computed either way, so an attacker cannot tell which usernames exist.

    Args:
        body: Username and password.
        request: The request, used to store the session.
        session: Database session.

    Returns:
        The logged-in user.

    Raises:
        HTTPException: 401 for wrong credentials.
    """
    settings: Settings = request.app.state.settings
    user = session.scalar(select(User).where(User.username == body.username))
    if user is None:
        hash_password(body.password, settings.password_iterations)
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong username or password.")
    request.session.clear()
    request.session[SESSION_USER_KEY] = user.id
    return user


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request) -> None:
    """End the session.

    Args:
        request: The request whose session is cleared.
    """
    request.session.clear()


@router.get("/auth/me", response_model=UserOut)
def me(user: UserDep) -> User:
    """Return the logged-in user, so the interface knows who and what role.

    Args:
        user: The logged-in user.

    Returns:
        The user.
    """
    return user


@router.get("/users", response_model=list[UserOut])
def list_users(_admin: AdminDep, session: SessionDep) -> list[User]:
    """List all users (admins only).

    Args:
        _admin: Ensures the caller is an admin.
        session: Database session.

    Returns:
        All users.
    """
    return list(session.scalars(select(User).order_by(User.id)))


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(body: NewUser, request: Request, _admin: AdminDep, session: SessionDep) -> User:
    """Create a user (admins only).

    Args:
        body: The new user's details.
        request: The request, used to read the password settings.
        _admin: Ensures the caller is an admin.
        session: Database session.

    Returns:
        The created user.

    Raises:
        HTTPException: 409 if the username is taken.
    """
    settings: Settings = request.app.state.settings
    user = User(
        username=body.username,
        password_hash=hash_password(body.password, settings.password_iterations),
        role=body.role,
    )
    session.add(user)
    try:
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "That username is taken.") from error
    return user
