"""Application settings, read from environment variables or the ``.env`` file."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for DocuCite AI in one validated place.

    Values come from environment variables first, then from ``.env``, then
    from the defaults below. Keeping this in one class means a typo in a
    variable name or a wrong type fails at startup, not in the middle of a request.

    Attributes:
        database_url: SQLAlchemy URL of the SQLite database. The file lives in
            ``instance/`` so it is git-ignored and separate from the code.
        llm_provider: Which provider answers questions: ``"gemini"`` (hosted
            free tier, the default) or ``"ollama"`` (local, free).
        ollama_url: Base URL of the local Ollama server (only for ``ollama``).
        ollama_model: Ollama chat model used to write answers.
        embed_model: Model that turns text into embedding vectors. It must
            belong to the chosen provider (a Gemini model name by default).
        gemini_api_key: Gemini API key. A ``SecretStr`` so it is masked if the
            settings object is ever printed or logged. Only needed for Gemini.
        gemini_model: Gemini model used to write answers.
        max_upload_mb: Largest PDF accepted, in megabytes. Uploads are untrusted,
            so an upper bound protects memory and disk.
        chunk_max_chars: Maximum characters per text chunk.
        chunk_overlap: Characters repeated between neighbouring chunks of a page.
            Must be smaller than ``chunk_max_chars``.
        answer_top_k: How many passages are given to the model to answer a question.
        secret_key: Signs the login cookie. If empty, a random key is generated at
            startup (everyone is logged out on each restart), so set it in ``.env``.
        admin_username: Username of the first admin, created at startup when no
            users exist yet.
        admin_password: Password of that first admin. Leave empty to skip creating it.
        password_iterations: PBKDF2 work factor for password hashing. High on
            purpose: it makes stolen password hashes expensive to crack.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite:///instance/docucite.db"
    llm_provider: Literal["ollama", "gemini"] = "gemini"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    embed_model: str = "gemini-embedding-001"
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-3.5-flash"
    max_upload_mb: int = Field(default=20, ge=1)
    chunk_max_chars: int = Field(default=1200, ge=1)
    chunk_overlap: int = Field(default=200, ge=0)
    answer_top_k: int = Field(default=5, ge=1, le=20)
    secret_key: SecretStr = SecretStr("")
    admin_username: str = "admin"
    admin_password: SecretStr = SecretStr("")
    password_iterations: int = Field(default=600_000, ge=1)

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        """Reject an overlap that would stop the chunker from making progress."""
        if self.chunk_overlap >= self.chunk_max_chars:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_MAX_CHARS")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, reading them only once.

    Returns:
        The shared ``Settings`` instance.
    """
    return Settings()
