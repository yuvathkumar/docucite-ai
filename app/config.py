"""Application settings, read from environment variables or the ``.env`` file."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
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
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite:///instance/docucite.db"
    llm_provider: Literal["ollama", "gemini"] = "gemini"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    embed_model: str = "gemini-embedding-001"
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-3.5-flash"


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, reading them only once.

    Returns:
        The shared ``Settings`` instance.
    """
    return Settings()
