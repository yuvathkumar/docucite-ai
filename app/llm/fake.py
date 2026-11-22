"""Deterministic fakes for tests: no network, no API key, repeatable results."""

import hashlib
import math
import re
from collections.abc import Callable, Sequence

FAKE_DIMENSIONS = 64
WORD_PATTERN = re.compile(r"\w+")


class FakeEmbedder:
    """Embed text by hashing its words into a fixed number of buckets.

    Texts that share words get similar vectors, so search tests can check that
    relevant passages rank higher, while everything stays deterministic.

    Attributes:
        model_name: Identifier stored with each vector.
        dimensions: Length of every vector.
    """

    model_name = "fake-hash-embedding"

    def __init__(self, dimensions: int = FAKE_DIMENSIONS) -> None:
        """Create the embedder.

        Args:
            dimensions: Length of every vector.
        """
        self.dimensions = dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed each text as a length-normalised bag-of-words vector.

        Args:
            texts: The passages to embed.

        Returns:
            One vector per text, in order.
        """
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a question the same way as a passage.

        Args:
            text: The question.

        Returns:
            The question's vector.
        """
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        """Hash each word into a bucket, count it, and scale the vector to length 1."""
        vector = [0.0] * self.dimensions
        for word in WORD_PATTERN.findall(text.lower()):
            bucket = int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dimensions
            vector[bucket] += 1.0
        length = math.sqrt(sum(value * value for value in vector))
        return [value / length for value in vector] if length else vector


class FakeChatModel:
    """A scripted chat model that records the prompts it receives.

    Attributes:
        model_name: Identifier of the fake.
        reply: The scripted reply; tests may change it between calls.
        calls: `(system_prompt, user_prompt)` pairs, one per `generate` call.
    """

    model_name = "fake-chat"

    def __init__(
        self, reply: str | Callable[[str, str], str] = "Answer from the docs. [1]"
    ) -> None:
        """Create the fake.

        Args:
            reply: A fixed reply, or a function ``(system, user) -> reply``.
        """
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Record the prompts and return the scripted reply.

        Args:
            system_prompt: Fixed rules for the model.
            user_prompt: The question together with its context.

        Returns:
            The scripted reply.
        """
        self.calls.append((system_prompt, user_prompt))
        if callable(self.reply):
            return self.reply(system_prompt, user_prompt)
        return self.reply
