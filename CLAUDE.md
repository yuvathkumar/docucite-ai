# CLAUDE.md — DocuCite AI

Claude Code reads this file at the start of every session. These rules apply to all work in this repository.

## 1. About the project

**DocuCite AI** lets a team upload their documents (policies, manuals, guides, reports as PDFs) and ask questions in plain English. **Every answer cites the exact document, page and passage.** If the documents do not contain the answer, it says so instead of guessing.

- **Pattern:** RAG (Retrieval-Augmented Generation): ingest PDFs → split into chunks → create embeddings → store them → retrieve the most relevant chunks → generate a cited answer.
- **Search:** hybrid. Keyword search with SQLite **FTS5** (built into Python), plus vector similarity search written in **pure Python** (cosine similarity), combined into one ranking.
- **Roles:** admins upload and manage documents; users ask questions. Every question and answer is logged for review.

### Stack (Windows-native, nothing else to install)
- Python 3 in a virtual environment named `docucite-ai` (created with `python -m venv docucite-ai`; activate with `.\docucite-ai\Scripts\Activate.ps1`)
- **Web framework:** FastAPI if it passes the compatibility check in the setup prompt, otherwise Flask. Once chosen, it is recorded here.
  - Chosen framework: **FastAPI** (compatibility check passed on 2026-11-22 with Python 3.15.2; Flask was uninstalled)
  - SQLAlchemy is pinned to `>=2.0,<2.1`: version 2.1 loads compiled modules that Smart App Control blocks, while 2.0.x honours `DISABLE_SQLALCHEMY_CEXT_RUNTIME=1`.
  - **ruff runs locally. mypy is CI-only:** it needs the compiled `librt` module, which Application Control blocks and which has no pure-Python version.
- SQLite (via SQLAlchemy 2.0), pypdf, pytest
- LLM and embeddings: **Ollama** (free, Windows installer) by default, or the **Gemini API** free tier, called over HTTP with the standard library
- GitHub Actions for CI. It runs in the cloud, so nothing is installed locally.

## 2. Who I am and how to work with me

I'm a Python developer building this portfolio project **to learn**. I must be able to explain every part in a job interview.

- **Teach like a teacher.** For every file or decision, explain **why** it exists, not only what it does.
- **Plan first.** Before building a feature, show a short plan (files, approach, trade-offs) and **wait for my approval**.
- **One feature at a time.** Do not build several features in one step.
- **After each feature, give me:**
  1. the files changed, with one line each on why
  2. how to run and test it
  3. **3 quiz questions** about what was built
  4. a suggested commit message (Conventional Commits, e.g. `feat: add pdf ingestion`)
- **Git:** do **not** commit or push yourself. Give me the commands and I run them, because I'm learning Git.
- Before writing the core logic (chunking, vector search, ranking, prompt building), ask whether I want to write it myself and have you review it.

## 3. Environment rules (very important)

- **Windows only.** Do **not** suggest or use WSL, Linux, Docker, Dev Containers or virtual machines.
- **All packages go into the project virtual environment `docucite-ai`.** Never install globally or with `--user`.
- Always run tools as modules: `python -m pip`, `python -m pytest`, `python -m flask` / `python -m uvicorn`. Do not call `.exe` launchers such as `pytest.exe`.
- Commands are for **PowerShell**. Set environment variables with `$env:NAME = "value"`.
- **Smart App Control** on this laptop blocks unsigned compiled files (`.pyd`, `.exe`) from pip packages. Rules:
  - Prefer **pure-Python** packages. Before adding any new dependency, check whether it needs compiled code and tell me.
  - If something fails with an "Application Control" or DLL-blocked error, **never** try to disable security. Explain it, then choose a pure-Python alternative or move that tool to CI only.
  - Already known: SQLAlchemy needs `DISABLE_SQLALCHEMY_CEXT_RUNTIME=1` (already set as a user environment variable).
- Dependencies are declared in `pyproject.toml`, with versions pinned to a compatible range.
- Only free and open-source tools.

## 4. Coding standards

### Style
- Follow **PEP 8**: formatting and linting with **ruff**, line length 100. If ruff is blocked locally, it runs in CI only, and you keep the code PEP 8 clean yourself.
- Imports grouped: standard library, then third-party, then local.
- Names: `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants. Use clear names: `chunk_text`, not `ct`.

### Type hints
- Every function has type hints for its parameters and return value.
- Type checking with **mypy** (locally if it runs, otherwise in CI).

### Docstrings (Google style)
Every **module**, **class** and **public function** has a docstring. The format:

```python
def split_into_chunks(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    """Split a long text into overlapping chunks for embedding.

    Overlap keeps a sentence that crosses a chunk boundary readable in
    at least one chunk, which improves retrieval quality.

    Args:
        text: The full text of one document page or section.
        max_chars: Maximum characters per chunk.
        overlap: Characters repeated between neighbouring chunks.

    Returns:
        The chunks in their original order. Returns an empty list for empty text.

    Raises:
        ValueError: If ``overlap`` is not smaller than ``max_chars``.

    Example:
        >>> split_into_chunks("abc", max_chars=2, overlap=1)
        ['ab', 'bc']
    """
```

- The first line is a one-sentence summary in the imperative ("Split…", "Return…").
- Explain **why** where it isn't obvious.
- `Args`, `Returns` and `Raises` whenever they apply. `Example` for core utilities.
- Module docstring at the top of every file: what the module is responsible for.

### Comments
- Comments explain **why**, not what. Do not comment obvious lines.
- No commented-out code. Delete it; Git keeps the history.

### Design
- Small functions with one responsibility each.
- No magic numbers: use named constants or settings.
- All configuration goes through `app/config.py`, read from environment variables or `.env`.
- Use `logging`, never `print`, in application code.
- Raise specific exceptions and turn them into clear HTTP errors at the web layer.
- The LLM and embedding providers sit behind an interface, so tests can use fakes.

## 5. Security rules

- **Secrets only in `.env`.** `.env` is in `.gitignore`; `.env.example` (no real values) is committed. Never print, log or hard-code an API key.
- Treat all LLM output and all uploaded files as **untrusted**: validate file type and size, and never execute model output.
- Sample documents must be openly licensed (public domain or Creative Commons) or written by me. Record each source and its licence in `data/SOURCES.md`.
- **Nothing from my employer:** no code, data, documents or copies of internal tools.
- Use parameterised SQL only (SQLAlchemy). Never build SQL by string formatting.

## 6. Testing rules

- Every feature comes with **pytest** tests in the same step.
- Tests never call a real LLM or the internet. Use the fake LLM and fake embeddings.
- Tests use a temporary or in-memory SQLite database, never the dev database.
- **Before saying a step is done, run the checks and show me the result:**
  ```
  python -m pytest -v
  python -m ruff format --check .   # if ruff runs locally
  python -m ruff check .            # if ruff runs locally
  python -m mypy app                # if mypy runs locally
  ```
- CI (GitHub Actions) always runs all four on every push.

## 7. Project structure

```
docucite-ai/
├── .github/workflows/ci.yml
├── app/
│   ├── __init__.py         # App factory
│   ├── config.py           # Settings from env / .env
│   ├── db.py               # SQLAlchemy engine, session, FTS5 setup
│   ├── models.py           # Tables: users, documents, chunks, questions
│   ├── web/                # Routes / API endpoints and templates
│   ├── rag/                # ingest, chunking, embeddings, search, prompts
│   └── llm/                # Provider interface + Ollama, Gemini, Fake
├── data/
│   ├── samples/            # Openly licensed sample PDFs
│   └── SOURCES.md
├── scripts/                # check_env.py, check_llm.py
├── tests/
├── pyproject.toml
├── .env.example
├── .gitignore
├── CLAUDE.md
└── README.md
```

Do not create files outside this structure without asking.
