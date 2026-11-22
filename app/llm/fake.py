"""A deterministic embedder for tests: no network, no API key, repeatable results."""

import hashlib
import math
import re
from collections.abc import Sequence

FAKE_DIMENSIONS = 64
WORD_PATTERN = re.compile(r"\w+")


class FakeEmbedder:
    """Embed text by hashing its words into a fixed number of buckets.

    Texts that share words get similar vectors, so later search tests can check
    that relevant passages rank higher, while everything stays deterministic.

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

    def _embed_one(self, text: str) -> list[float]:
        """Hash each word into a bucket, count it, and scale the vector to length 1."""
        vector = [0.0] * self.dimensions
        for word in WORD_PATTERN.findall(text.lower()):
            bucket = int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dimensions
            vector[bucket] += 1.0
        length = math.sqrt(sum(value * value for value in vector))
        return [value / length for value in vector] if length else vector
