"""Check which packages import on this machine and whether SQLite FTS5 works.

Smart App Control can block unsigned compiled files (``.pyd``/``.dll``) that
pip packages ship. Running this once tells us which web framework and tools
are safe to depend on, before we build anything on top of them.
"""

import importlib
import importlib.util
import sqlite3
import sys
from pathlib import Path

COMPILED_SUFFIXES = (".pyd", ".dll", ".so")

# (label shown in the table, importable module name)
CANDIDATES: list[tuple[str, str]] = [
    ("fastapi", "fastapi"),
    # The package itself imports lazily; the compiled module is the part that can be blocked.
    ("pydantic_core", "pydantic_core._pydantic_core"),
    ("uvicorn", "uvicorn"),
    ("flask", "flask"),
    ("sqlalchemy", "sqlalchemy"),
    ("pypdf", "pypdf"),
    ("pytest", "pytest"),
    ("ruff", "ruff"),
    ("mypy", "mypy"),
]


def has_compiled_code(module_name: str) -> bool | None:
    """Report whether a package ships compiled extension files.

    Args:
        module_name: Importable top-level module name.

    Returns:
        True if compiled files are found, False if the package is pure Python,
        None if the package is not installed.
    """
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    if spec.submodule_search_locations:
        roots = [Path(location) for location in spec.submodule_search_locations]
    elif spec.origin:
        roots = [Path(spec.origin)]
    else:
        return False
    for root in roots:
        files = [root] if root.is_file() else root.rglob("*")
        if any(f.suffix in COMPILED_SUFFIXES for f in files):
            return True
    return False


def try_import(module_name: str) -> str:
    """Try to import a module and describe the outcome.

    Args:
        module_name: Importable top-level module name.

    Returns:
        ``"OK"``, ``"not installed"`` or ``"BLOCKED: <reason>"``.
    """
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        return "not installed"
    except Exception as error:  # noqa: BLE001 - any failure means "unusable here"
        return f"BLOCKED: {str(error)[:60]}"
    return "OK"


def check_fts5() -> str:
    """Create an in-memory FTS5 table and run one search against it.

    Returns:
        ``"OK"`` if the search finds the row, otherwise a short failure reason.
    """
    try:
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE VIRTUAL TABLE docs USING fts5(body)")
        connection.execute("INSERT INTO docs(body) VALUES ('vacation policy for staff')")
        rows = connection.execute("SELECT body FROM docs WHERE docs MATCH 'vacation'").fetchall()
        connection.close()
    except sqlite3.Error as error:
        return f"FAILED: {error}"
    return "OK" if rows else "FAILED: no match returned"


def main() -> None:
    """Print the compatibility table and the FTS5 result."""
    print(f"Python {sys.version.split()[0]} at {sys.executable}\n")
    print(f"{'package':<15}{'import':<28}compiled code")
    print("-" * 60)
    for label, module_name in CANDIDATES:
        compiled = has_compiled_code(module_name)
        compiled_text = {True: "yes", False: "no", None: "-"}[compiled]
        print(f"{label:<15}{try_import(module_name):<28}{compiled_text}")
    print(f"\nSQLite {sqlite3.sqlite_version} FTS5: {check_fts5()}")


if __name__ == "__main__":
    main()
