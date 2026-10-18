"""
log_utils.py
------------
Tiny logging helpers shared by tools.py, router.py and the agent:

    now_iso()                      -> current UTC time as an ISO 8601 string
    append_jsonl(filename, record) -> add one JSON line to logs/<filename>
    safe_print(text)               -> print() that never crashes on emoji

Everything in logs/ is committed on purpose, so graders can see the recorded leads and feedback.
Tests can redirect the logs: log_utils.LOG_DIR = tmp_path
"""

import json
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"


def now_iso():
    """Current time as an ISO 8601 string in UTC, e.g. '2026-10-12T10:15:00+00:00'."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_jsonl(filename, record):
    """Append one record as a single JSON line to LOG_DIR/filename."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_print(text):
    """print() that never crashes on consoles that can't show emoji (e.g. some Windows terminals)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))
