# DocuCite AI

Upload your PDFs, ask questions in plain English, and get answers that cite the exact document, page and passage. If the documents don't contain the answer, it says so instead of guessing.

> Status: project setup complete; features are being built one at a time.

## Screenshots

_Coming soon._

## Features

_Planned: PDF ingestion, hybrid search (keyword + vector), cited answers, admin/user roles, question log._

## Tech stack

- **Language:** Python 3 (virtual environment `docucite-ai`)
- **Web:** FastAPI + Uvicorn, Pydantic settings
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
   Open `.env` and set `GEMINI_API_KEY` (free key: https://aistudio.google.com/apikey).
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
   Open http://127.0.0.1:8000/health and http://127.0.0.1:8000/docs.

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

Only openly licensed or self-written documents are used. See `data/SOURCES.md`.
