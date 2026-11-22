"""Gemini providers (embeddings and chat), called over plain HTTP with httpx."""

import logging
import time
from collections.abc import Callable, Sequence

import httpx

from app.llm.base import EmbeddingError, GenerationError

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
BATCH_SIZE = 100
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.0
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"
GENERATION_TEMPERATURE = 0.2  # low: answers should stick to the passages, not improvise


class GeminiHttp:
    """Shared HTTP plumbing: auth header, retries with backoff, safe error messages."""

    def __init__(
        self,
        api_key: str,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create the helper.

        Args:
            api_key: Gemini API key. Sent in a header and never logged.
            client: HTTP client; tests pass one with a mock transport.
            sleep: Function used to wait between retries; tests pass a no-op.
        """
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
        self._sleep = sleep

    def post(
        self, url: str, body: dict[str, object], error_type: type[Exception]
    ) -> httpx.Response:
        """POST JSON, retrying temporary failures (rate limits, overload, network).

        Gemini answers 503 when a model is busy; a short wait usually fixes it.

        Args:
            url: Full endpoint URL.
            body: JSON request body.
            error_type: Exception class to raise on failure (embedding vs generation).

        Returns:
            The successful response.

        Raises:
            Exception: An ``error_type`` instance if the key is missing, the error is
                not temporary, or every attempt fails.
        """
        if not self._api_key:
            raise error_type("GEMINI_API_KEY is empty. Put your key in .env.")
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}
        reason = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._client.post(url, headers=headers, json=body)
            except httpx.TransportError as error:
                reason = type(error).__name__
            else:
                if response.status_code < httpx.codes.BAD_REQUEST:
                    return response
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    raise error_type(f"Gemini returned HTTP {response.status_code}.")
                reason = f"HTTP {response.status_code}"
            logger.warning("Gemini attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, reason)
            if attempt < MAX_ATTEMPTS:
                self._sleep(RETRY_BASE_DELAY_SECONDS * 2 ** (attempt - 1))
        raise error_type(f"Gemini is unavailable after {MAX_ATTEMPTS} attempts ({reason}).")


class GeminiEmbedder:
    """Embed text with a Gemini embedding model.

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
            api_key: Gemini API key.
            model_name: Gemini embedding model name.
            client: HTTP client; tests pass one with a mock transport.
            sleep: Function used to wait between retries; tests pass a no-op.
        """
        self.model_name = model_name
        self._http = GeminiHttp(api_key, client, sleep)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages, splitting them into batches the API accepts.

        Args:
            texts: The passages to embed.

        Returns:
            One vector per text, in order.

        Raises:
            EmbeddingError: If the request keeps failing or the response does not
                contain one vector per text.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = list(texts[start : start + BATCH_SIZE])
            vectors.extend(self._embed_batch(batch, DOCUMENT_TASK_TYPE))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a question with the query task type.

        Args:
            text: The question.

        Returns:
            The question's vector.

        Raises:
            EmbeddingError: If the request fails or the response is unexpected.
        """
        return self._embed_batch([text], QUERY_TASK_TYPE)[0]

    def _embed_batch(self, batch: list[str], task_type: str) -> list[list[float]]:
        """Send one ``batchEmbedContents`` request and parse the vectors."""
        model_path = f"models/{self.model_name}"
        body: dict[str, object] = {
            "requests": [
                {
                    "model": model_path,
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                }
                for text in batch
            ]
        }
        url = f"{GEMINI_BASE_URL}/{model_path}:batchEmbedContents"
        response = self._http.post(url, body, EmbeddingError)
        try:
            vectors = [item["values"] for item in response.json()["embeddings"]]
        except (KeyError, TypeError, ValueError) as error:
            raise EmbeddingError("Gemini returned an unexpected response.") from error
        if len(vectors) != len(batch):
            raise EmbeddingError("Gemini returned a different number of vectors than texts.")
        return vectors


class GeminiChat:
    """Generate answers with a Gemini chat model.

    Attributes:
        model_name: The Gemini model, for example ``gemini-3.5-flash``.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create the chat model.

        Args:
            api_key: Gemini API key.
            model_name: Gemini chat model name.
            client: HTTP client; tests pass one with a mock transport.
            sleep: Function used to wait between retries; tests pass a no-op.
        """
        self.model_name = model_name
        self._http = GeminiHttp(api_key, client, sleep)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a reply.

        Args:
            system_prompt: Fixed rules for the model.
            user_prompt: The question together with its context.

        Returns:
            The model's text.

        Raises:
            GenerationError: If the request fails or the reply contains no text
                (for example when the answer is blocked).
        """
        body: dict[str, object] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": GENERATION_TEMPERATURE},
        }
        url = f"{GEMINI_BASE_URL}/models/{self.model_name}:generateContent"
        response = self._http.post(url, body, GenerationError)
        try:
            parts = response.json()["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts).strip()
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise GenerationError("Gemini returned no answer.") from error
        if not text:
            raise GenerationError("Gemini returned an empty answer.")
        return text
