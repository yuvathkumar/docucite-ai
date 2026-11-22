"""LLM and embedding providers behind one interface (Gemini, fake)."""

from collections.abc import Sequence

from app.config import Settings
from app.llm.base import Embedder, EmbeddingError
from app.llm.gemini import GeminiEmbedder

__all__ = ["Embedder", "EmbeddingError", "UnavailableEmbedder", "create_embedder"]


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


def create_embedder(settings: Settings) -> Embedder:
    """Build the embedder selected by ``LLM_PROVIDER``.

    Creating the embedder never contacts the network, so the app still starts
    (and ``/health`` works) when the key is missing or the provider is down.

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
