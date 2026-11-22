"""Smoke-test the configured LLM provider before the app depends on it.

Sends one short prompt and one embedding request, and prints the reply, the
embedding size and how long each call took. Settings (including the API key)
come from ``.env`` through ``app.config``; the key is never printed.

Run with: ``python scripts/check_llm.py``
"""

import sys
import time

import httpx

from app.config import Settings, get_settings

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
PROMPT = "Reply with the word OK"
EMBED_SAMPLE_TEXT = "DocuCite AI answers questions with citations."
REQUEST_TIMEOUT_SECONDS = 60.0


def gemini_headers(settings: Settings) -> dict[str, str]:
    """Build request headers carrying the API key.

    The key goes in a header, not the URL, because URLs end up in logs and
    error messages while headers do not.

    Args:
        settings: Application settings holding the Gemini key.

    Returns:
        Headers for a Gemini JSON request.

    Raises:
        SystemExit: If no API key is configured.
    """
    api_key = settings.gemini_api_key.get_secret_value()
    if not api_key:
        sys.exit("GEMINI_API_KEY is empty. Put your key in .env (see .env.example).")
    return {"x-goog-api-key": api_key, "Content-Type": "application/json"}


def ask_gemini(client: httpx.Client, settings: Settings) -> str:
    """Send the prompt to the Gemini chat model and return its text reply.

    Args:
        client: HTTP client to use.
        settings: Application settings (model name and key).

    Returns:
        The model's reply text.
    """
    response = client.post(
        f"{GEMINI_BASE_URL}/models/{settings.gemini_model}:generateContent",
        headers=gemini_headers(settings),
        json={"contents": [{"parts": [{"text": PROMPT}]}]},
    )
    response.raise_for_status()
    parts = response.json()["candidates"][0]["content"]["parts"]
    return "".join(part.get("text", "") for part in parts).strip()


def embed_gemini(client: httpx.Client, settings: Settings) -> list[float]:
    """Request one embedding vector from Gemini.

    Args:
        client: HTTP client to use.
        settings: Application settings (embedding model name and key).

    Returns:
        The embedding as a list of floats.
    """
    response = client.post(
        f"{GEMINI_BASE_URL}/models/{settings.embed_model}:embedContent",
        headers=gemini_headers(settings),
        json={"content": {"parts": [{"text": EMBED_SAMPLE_TEXT}]}},
    )
    response.raise_for_status()
    values: list[float] = response.json()["embedding"]["values"]
    return values


def main() -> None:
    """Run the chat and embedding checks and print the results."""
    settings = get_settings()
    if settings.llm_provider != "gemini":
        sys.exit(f"This check only supports gemini; LLM_PROVIDER is {settings.llm_provider!r}.")

    print(f"Provider: gemini | chat model: {settings.gemini_model} | embed: {settings.embed_model}")
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        started = time.perf_counter()
        reply = ask_gemini(client, settings)
        print(f"Reply: {reply!r} ({time.perf_counter() - started:.2f}s)")

        started = time.perf_counter()
        vector = embed_gemini(client, settings)
        print(f"Embedding length: {len(vector)} ({time.perf_counter() - started:.2f}s)")


if __name__ == "__main__":
    try:
        main()
    except httpx.HTTPStatusError as error:
        sys.exit(f"Gemini returned HTTP {error.response.status_code}: {error.response.text[:300]}")
    except httpx.HTTPError as error:
        sys.exit(f"Network problem talking to Gemini: {type(error).__name__}")
