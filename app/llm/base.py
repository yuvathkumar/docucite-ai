"""The interface every embedding provider implements, so the app never depends on one vendor."""

from collections.abc import Sequence
from typing import Protocol


class EmbeddingError(Exception):
    """An embedding provider could not produce vectors (network, quota or bad response)."""


class Embedder(Protocol):
    """Turns text into embedding vectors.

    The app depends on this protocol only. Tests plug in a fake, and switching
    from Gemini to another provider means writing one new class.

    Attributes:
        model_name: Identifies the embedding model. It is stored with every
            vector, because vectors from different models are not comparable.
    """

    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages that will be stored and searched later.

        Args:
            texts: The passages to embed.

        Returns:
            One vector per input text, in the same order.

        Raises:
            EmbeddingError: If the provider fails or returns an unexpected result.
        """
        ...
