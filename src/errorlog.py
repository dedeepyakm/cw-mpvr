"""Append development errors to ERROR_LOG.md in one consistent format.

Usage in a notebook, right after a cell fails:

    log_error(
        where="02_generate_paths / step 3",
        cause="Batch of 16 paths did not fit in GPU memory",
        fix="Lowered batch size to 4 and used fp16",
    )

The error message is taken from the last failed cell automatically.
"""

import re
import sys
from datetime import date

from .utils import REPO_ROOT

LOG_PATH = REPO_ROOT / "ERROR_LOG.md"


def _last_error() -> str:
    """Short 'Type: message' text of the most recent uncaught exception."""
    etype, value = getattr(sys, "last_type", None), getattr(sys, "last_value", None)
    if etype is None:
        return "(no recent error found: pass error=... by hand)"
    msg = str(value).strip().splitlines()[0] if str(value).strip() else ""
    return f"{etype.__name__}: {msg[:200]}"


def _next_number() -> int:
    text = LOG_PATH.read_text() if LOG_PATH.exists() else ""
    return len(re.findall(r"^### E\d+", text, flags=re.M)) + 1


def log_error(where: str, cause: str, fix: str, error: str | None = None,
              lesson: str | None = None) -> None:
    """Add one numbered entry to ERROR_LOG.md and print it."""
    n = _next_number()
    entry = [
        f"\n### E{n:02d} ({date.today().isoformat()})",
        f"- **Where**: {where}",
        f"- **Error**: `{error or _last_error()}`",
        f"- **Cause**: {cause}",
        f"- **Fix**: {fix}",
    ]
    if lesson:
        entry.append(f"- **Lesson**: {lesson}")
    text = "\n".join(entry) + "\n"
    with open(LOG_PATH, "a") as f:
        f.write(text)
    print(f"Logged as E{n:02d}:{text}")
