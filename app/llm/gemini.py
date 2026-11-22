"""Gemini embedding provider, called over plain HTTP with httpx."""

import logging
import time
from collections.abc import Callable, Sequence

import httpx

from app.llm.base import EmbeddingError

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
BATCH_SIZE = 100
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.0
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"


class GeminiEmbedder:
    """Embed passages with a Gemini embedding model.

    Attributes:
        model_name: The Gemini embedding model, for example ``gemini-embedding-001``.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create the embedder.

        Args:
            api_key: Gemini API key. It is sent in a header and never logged.
            model_name: Gemini embedding model name.
            client: HTTP client; tests pass one with a mock transport.
            sleep: Function used to wait between retries; tests pass a no-op.
        """
        self._api_key = api_key
        self.model_name = model_name
        self._client = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
        self._sleep = sleep

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages, splitting them into batches the API accepts.

        Args:
            texts: The passages to embed.

        Returns:
            One vector per text, in order.

        Raises:
            EmbeddingError: If no API key is set, the request keeps failing, or the
                response does not contain one vector per text.
        """
        if not self._api_key:
            raise EmbeddingError("GEMINI_API_KEY is empty. Put your key in .env.")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            vectors.extend(self._embed_batch(list(texts[start : start + BATCH_SIZE])))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Send one ``batchEmbedContents`` request and parse the vectors."""
        model_path = f"models/{self.model_name}"
        body = {
            "requests": [
                {
                    "model": model_path,
                    "content": {"parts": [{"text": text}]},
                    "taskType": DOCUMENT_TASK_TYPE,
                }
                for text in batch
            ]
        }
        response = self._post_with_retries(
            f"{GEMINI_BASE_URL}/{model_path}:batchEmbedContents", body
        )
        try:
            vectors = [item["values"] for item in response.json()["embeddings"]]
        except (KeyError, TypeError, ValueError) as error:
            raise EmbeddingError("Gemini returned an unexpected response.") from error
        if len(vectors) != len(batch):
            raise EmbeddingError("Gemini returned a different number of vectors than texts.")
        return vectors

    def _post_with_retries(self, url: str, body: dict[str, object]) -> httpx.Response:
        """POST JSON, retrying temporary failures (rate limits, overload, network).

        Gemini answers 503 when a model is busy; a short wait usually fixes it.

        Raises:
            EmbeddingError: If every attempt fails or the error is not temporary.
        """
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._client.post(url, headers=headers, json=body)
            except httpx.TransportError as error:
                reason = type(error).__name__
            else:
                if response.status_code < httpx.codes.BAD_REQUEST:
                    return response
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    raise EmbeddingError(f"Gemini returned HTTP {response.status_code}.")
                reason = f"HTTP {response.status_code}"
            logger.warning(
                "Gemini embedding attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, reason
            )
            if attempt < MAX_ATTEMPTS:
                self._sleep(RETRY_BASE_DELAY_SECONDS * 2 ** (attempt - 1))
        raise EmbeddingError(f"Gemini is unavailable after {MAX_ATTEMPTS} attempts ({reason}).")
