# DocuCite AI

Upload your PDFs, ask questions in plain English, and get answers that cite the exact document, page and passage. If the documents don't contain the answer, it says so instead of guessing.

## Screenshots

_Add screenshots of the chat, documents and login pages here._

## Features

- **Cited answers:** ask a question in the chat; the answer carries numbered citations. Click one to
  see the exact passage, document and page. An answer that cites nothing is never shown; if the
  documents don't contain the answer you are told so.
- **Hybrid search:** SQLite FTS5 keyword ranking (BM25) plus vector similarity (cosine, pure
  Python), merged with Reciprocal Rank Fusion. If the embedding provider is down, keyword search
  still answers.
- **PDF ingestion:** page-aware chunking with overlap, duplicate detection (SHA-256), size and
  signature checks. Scanned (image-only) PDFs are not supported.
- **Roles:** admins upload and delete documents, manage users and review the **question log**; users
  ask questions. Passwords are salted PBKDF2 hashes; sessions use signed cookies.
- **Web interface:** single-page app (no build step), light/dark theme, mobile friendly.
- **Provider-agnostic:** Gemini sits behind `Embedder` / `ChatModel` interfaces; tests use fakes and
  never touch the network.

> **Privacy:** passage text and questions are sent to the AI provider (Google Gemini). On the free
> tier Google may use them to improve its products, so do not upload confidential documents.

## Tech stack

- **Language:** Python 3 (virtual environment `docucite-ai`)
- **Web:** FastAPI + Uvicorn, Pydantic settings, vanilla JS front end
- **Database:** SQLite via SQLAlchemy 2.0, with FTS5 for keyword search
- **PDF parsing:** pypdf
- **LLM and embeddings:** Google Gemini API (free tier), called over HTTP with httpx
- **Quality:** pytest, ruff, mypy, GitHub Actions

## Run locally (Windows, PowerShell)

1. **Create and activate the virtual environment**
   ```powershell
   python -m venv docucite-ai
   .\docucite-ai\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"
   ```

2. **Configure the environment**
   ```powershell
   Copy-Item .env.example .env
   ```
   Open `.env` and set:
   - `GEMINI_API_KEY` (free key: https://aistudio.google.com/apikey)
   - `SECRET_KEY` (a long random string; signs the login cookie)
   - `ADMIN_USERNAME` / `ADMIN_PASSWORD` (the first admin, created on first start)

   Never commit `.env`.

   On Windows with Smart App Control, also set once (then reopen the terminal):
   ```powershell
   [Environment]::SetEnvironmentVariable("DISABLE_SQLALCHEMY_CEXT_RUNTIME", "1", "User")
   ```

3. **Check the LLM connection**
   ```powershell
   python scripts/check_llm.py
   ```

4. **Run the app**
   ```powershell
   python -m uvicorn app:create_app --factory --reload
   ```
   Open http://127.0.0.1:8000 and sign in with the admin from `.env`. API docs are at `/docs`,
   health at `/health`.

5. **Run the tests and checks**
   ```powershell
   python -m pytest -v
   python -m ruff format --check .
   python -m ruff check .
   python -m mypy app
   ```
   `mypy` may be blocked by Smart App Control on Windows; it always runs in CI.

## Project structure

See `CLAUDE.md` section 7.

## Sample documents

Only openly licensed or self-written documents are used for demos and screenshots, for example
NIST SP 800-63B and IRS Publication 1 (US government works, public domain). See `data/SOURCES.md`.
Never upload personal or confidential documents while using the Gemini free tier.

## Known limits

- Only Gemini is implemented as the LLM and embedding provider (Ollama is not built yet).
- Scanned (image-only) PDFs are not supported (no OCR).
- Login attempts are not rate-limited yet.
