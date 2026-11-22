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
MAX_RETRY_DELAY_SECONDS = 40.0
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"
GENERATION_TEMPERATURE = 0.2  # low: answers should stick to the passages, not improvise


def retry_delay_seconds(response: httpx.Response | None, attempt: int) -> float:
    """Decide how long to wait before retrying a failed request.

    When Gemini rate-limits us (HTTP 429) it says how long to wait, either in a
    ``Retry-After`` header or as ``retryDelay`` (for example ``"34s"``) in the
    error body. Obeying it works far better than guessing: free-tier limits are
    per minute, so a 1-2 second wait just fails again. The wait is capped so one
    request never blocks for minutes.

    Args:
        response: The failed response, or None after a network error.
        attempt: Which attempt just failed, starting at 1.

    Returns:
        Seconds to wait.

    Example:
        >>> retry_delay_seconds(None, 2)
        2.0
        >>> retry_delay_seconds(httpx.Response(429, headers={"Retry-After": "7"}), 1)
        7.0
    """
    default = RETRY_BASE_DELAY_SECONDS * float(2 ** (attempt - 1))
    asked = server_retry_delay(response) if response is not None else None
    if asked is None:
        return default
    return min(max(asked, default), MAX_RETRY_DELAY_SECONDS)


def server_retry_delay(response: httpx.Response) -> float | None:
    """Read how long the server asked us to wait, without any cap.

    Args:
        response: A failed response.

    Returns:
        Seconds from the ``Retry-After`` header or the error body's ``retryDelay``,
        or None if the server gave no hint.

    Example:
        >>> server_retry_delay(httpx.Response(429, headers={"Retry-After": "7"}))
        7.0
        >>> server_retry_delay(httpx.Response(429)) is None
        True
    """
    header = response.headers.get("retry-after", "")
    if header.replace(".", "", 1).isdigit():
        return float(header)
    try:
        for detail in response.json()["error"]["details"]:
            delay = detail.get("retryDelay")
            if isinstance(delay, str) and delay.endswith("s"):
                return float(delay[:-1])
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    return None


def describe_wait(seconds: float) -> str:
    """Say a waiting time in plain words for an error message.

    Args:
        seconds: The wait in seconds.

    Returns:
        For example ``"about a minute"``, ``"about 20 minutes"`` or ``"about 18 hours"``.

    Example:
        >>> describe_wait(45)
        'about a minute'
        >>> describe_wait(1500)
        'about 25 minutes'
        >>> describe_wait(66273)
        'about 18 hours'
    """
    if seconds < SECONDS_PER_MINUTE * 1.5:
        return "about a minute"
    if seconds < SECONDS_PER_HOUR:
        return f"about {round(seconds / SECONDS_PER_MINUTE)} minutes"
    return f"about {round(seconds / SECONDS_PER_HOUR)} hours"


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
            failed_response: httpx.Response | None = None
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
                failed_response = response
                self._fail_fast_if_quota_used_up(response, error_type)
            logger.warning("Gemini attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, reason)
            if attempt < MAX_ATTEMPTS:
                self._sleep(retry_delay_seconds(failed_response, attempt))
        raise error_type(f"Gemini is unavailable after {MAX_ATTEMPTS} attempts ({reason}).")

    @staticmethod
    def _fail_fast_if_quota_used_up(response: httpx.Response, error_type: type[Exception]) -> None:
        """Stop at once when Gemini says the limit will not reset for a long time.

        A free-tier daily quota resets in hours. Waiting 40 seconds and retrying
        cannot help, and only delays the clear message the user needs.

        Args:
            response: The failed response.
            error_type: Exception class to raise.

        Raises:
            Exception: An ``error_type`` instance if this is a 429 whose reset time
                is longer than the longest wait we are willing to do.
        """
        if response.status_code != httpx.codes.TOO_MANY_REQUESTS:
            return
        asked = server_retry_delay(response)
        if asked is not None and asked > MAX_RETRY_DELAY_SECONDS:
            raise error_type(
                f"Gemini's free-tier limit is used up for now; try again in {describe_wait(asked)}."
            )


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
