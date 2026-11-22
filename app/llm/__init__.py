"""LLM and embedding providers behind one interface (Gemini, fake)."""

from collections.abc import Sequence

from app.config import Settings
from app.llm.base import ChatModel, Embedder, EmbeddingError, GenerationError
from app.llm.gemini import GeminiChat, GeminiEmbedder

__all__ = [
    "ChatModel",
    "Embedder",
    "EmbeddingError",
    "GenerationError",
    "UnavailableChat",
    "UnavailableEmbedder",
    "create_chat_model",
    "create_embedder",
]


class UnavailableEmbedder:
    """Stand-in used when the selected provider has no embedding support.

    It fails only when something tries to embed, so the rest of the app
    (uploads, keyword search, ``/health``) keeps working.

    Attributes:
        model_name: Placeholder identifier.
    """

    model_name = "unavailable"

    def __init__(self, reason: str) -> None:
        """Remember why embeddings cannot be produced.

        Args:
            reason: Human-readable explanation shown in the error.
        """
        self._reason = reason

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Always fail with the stored reason.

        Args:
            texts: Ignored.

        Raises:
            EmbeddingError: Always.
        """
        raise EmbeddingError(self._reason)

    def embed_query(self, text: str) -> list[float]:
        """Always fail with the stored reason.

        Args:
            text: Ignored.

        Raises:
            EmbeddingError: Always.
        """
        raise EmbeddingError(self._reason)


class UnavailableChat:
    """Stand-in used when the selected provider has no chat support.

    Attributes:
        model_name: Placeholder identifier.
    """

    model_name = "unavailable"

    def __init__(self, reason: str) -> None:
        """Remember why answers cannot be generated.

        Args:
            reason: Human-readable explanation shown in the error.
        """
        self._reason = reason

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Always fail with the stored reason.

        Args:
            system_prompt: Ignored.
            user_prompt: Ignored.

        Raises:
            GenerationError: Always.
        """
        raise GenerationError(self._reason)


def create_embedder(settings: Settings) -> Embedder:
    """Build the embedder selected by ``LLM_PROVIDER``.

    Creating it never contacts the network, so the app still starts (and
    ``/health`` works) when the key is missing or the provider is down.

    Args:
        settings: Application settings.

    Returns:
        The configured embedder.
    """
    if settings.llm_provider == "gemini":
        return GeminiEmbedder(
            api_key=settings.gemini_api_key.get_secret_value(),
            model_name=settings.embed_model,
        )
    return UnavailableEmbedder(
        f"Embeddings are not implemented for {settings.llm_provider!r}. Set LLM_PROVIDER=gemini."
    )


def create_chat_model(settings: Settings) -> ChatModel:
    """Build the chat model selected by ``LLM_PROVIDER``.

    Args:
        settings: Application settings.

    Returns:
        The configured chat model.
    """
    if settings.llm_provider == "gemini":
        return GeminiChat(
            api_key=settings.gemini_api_key.get_secret_value(),
            model_name=settings.gemini_model,
        )
    return UnavailableChat(
        f"Chat is not implemented for {settings.llm_provider!r}. Set LLM_PROVIDER=gemini."
    )
