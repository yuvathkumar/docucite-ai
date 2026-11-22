"""The interfaces every provider implements, so the app never depends on one vendor."""

from collections.abc import Sequence
from typing import Protocol


class EmbeddingError(Exception):
    """An embedding provider could not produce vectors (network, quota or bad response)."""


class GenerationError(Exception):
    """A chat model could not produce an answer (network, quota, block or bad response)."""


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

    def embed_query(self, text: str) -> list[float]:
        """Embed a user's question so it can be compared with stored passages.

        Providers optimise document and query vectors differently, which is why
        this is a separate method.

        Args:
            text: The question.

        Returns:
            The question's vector.

        Raises:
            EmbeddingError: If the provider fails or returns an unexpected result.
        """
        ...


class ChatModel(Protocol):
    """Generates text from a system prompt and a user prompt.

    Attributes:
        model_name: Identifies the chat model.
    """

    model_name: str

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a reply.

        Args:
            system_prompt: Fixed rules for the model.
            user_prompt: The question together with its context.

        Returns:
            The model's text. It is untrusted: validate it, never execute it.

        Raises:
            GenerationError: If the provider fails or returns no text.
        """
        ...
