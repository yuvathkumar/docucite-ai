"""Tests for the Gemini embedder, using a mock HTTP transport (no network)."""

import json
from collections.abc import Callable

import httpx
import pytest

from app.llm.base import EmbeddingError
from app.llm.gemini import BATCH_SIZE, MAX_ATTEMPTS, GeminiEmbedder

API_KEY = "test-key-123"
MODEL = "gemini-embedding-001"

Handler = Callable[[httpx.Request], httpx.Response]


def embeddings_response(request: httpx.Request) -> httpx.Response:
    """Answer a batch request with a 2-value vector per requested text."""
    count = len(json.loads(request.content)["requests"])
    return httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2]}] * count})


def make_embedder(handler: Handler, sleeps: list[float] | None = None) -> GeminiEmbedder:
    """Build an embedder whose HTTP calls go to ``handler`` and whose waits are recorded."""
    waits = sleeps if sleeps is not None else []
    return GeminiEmbedder(
        api_key=API_KEY,
        model_name=MODEL,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=waits.append,
    )


def test_embeds_texts_and_returns_one_vector_each() -> None:
    """Return the vectors in the same order and number as the texts."""
    embedder = make_embedder(embeddings_response)

    assert embedder.embed_documents(["a", "b", "c"]) == [[0.1, 0.2]] * 3


def test_request_uses_document_task_type_and_header_key() -> None:
    """Send the key in a header (not the URL) and ask for document embeddings."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return embeddings_response(request)

    make_embedder(handler).embed_documents(["hello"])

    request = seen[0]
    assert request.headers["x-goog-api-key"] == API_KEY
    assert API_KEY not in str(request.url)
    body = json.loads(request.content)["requests"][0]
    assert body["taskType"] == "RETRIEVAL_DOCUMENT"
    assert body["model"] == f"models/{MODEL}"
    assert body["content"]["parts"][0]["text"] == "hello"


def test_large_inputs_are_split_into_batches() -> None:
    """Never send more texts per request than the API accepts."""
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sizes.append(len(json.loads(request.content)["requests"]))
        return embeddings_response(request)

    vectors = make_embedder(handler).embed_documents(["t"] * (BATCH_SIZE * 2 + 5))

    assert sizes == [BATCH_SIZE, BATCH_SIZE, 5]
    assert len(vectors) == BATCH_SIZE * 2 + 5


def test_temporary_error_is_retried_with_growing_delay() -> None:
    """Recover from a 503 ("model busy") by waiting and trying again."""
    calls = {"count": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 3:
            return httpx.Response(503, json={"error": {"status": "UNAVAILABLE"}})
        return embeddings_response(request)

    result = make_embedder(handler, sleeps).embed_documents(["x"])

    assert result == [[0.1, 0.2]]
    assert calls["count"] == 3
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_max_attempts() -> None:
    """Raise a clear error when the provider stays unavailable."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(503)

    with pytest.raises(EmbeddingError, match="unavailable"):
        make_embedder(handler).embed_documents(["x"])

    assert calls["count"] == MAX_ATTEMPTS


def test_network_failure_is_retried_then_reported() -> None:
    """Treat a dropped connection like a temporary error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(EmbeddingError, match="ConnectError"):
        make_embedder(handler).embed_documents(["x"])


def test_client_error_is_not_retried_and_hides_the_key() -> None:
    """Fail fast on a bad request or key, without leaking the key in the message."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, json={"error": {"message": "API key not valid"}})

    with pytest.raises(EmbeddingError) as error:
        make_embedder(handler).embed_documents(["x"])

    assert calls["count"] == 1
    assert "400" in str(error.value)
    assert API_KEY not in str(error.value)


def test_unexpected_response_shape_raises() -> None:
    """Reject a 200 response that has no ``embeddings`` list."""
    embedder = make_embedder(lambda request: httpx.Response(200, json={"oops": True}))

    with pytest.raises(EmbeddingError, match="unexpected"):
        embedder.embed_documents(["x"])


def test_wrong_number_of_vectors_raises() -> None:
    """Reject a response that returns fewer vectors than texts."""
    embedder = make_embedder(
        lambda request: httpx.Response(200, json={"embeddings": [{"values": [1.0]}]})
    )

    with pytest.raises(EmbeddingError, match="number of vectors"):
        embedder.embed_documents(["a", "b"])


def test_missing_api_key_raises_before_any_request() -> None:
    """Explain how to fix an empty key instead of sending a doomed request."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be sent")

    embedder = GeminiEmbedder(
        api_key="",
        model_name=MODEL,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
        embedder.embed_documents(["x"])
