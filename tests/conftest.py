"""Shared pytest setup: make the project importable and keep test logs out of logs/."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                 # tools.py, agent.py, ...
sys.path.insert(0, str(ROOT / "tests"))       # fake_llm.py

import log_utils  # noqa: E402  (needs the path above)


@pytest.fixture(autouse=True)
def temp_logs(tmp_path, monkeypatch):
    """Every test writes its JSONL logs to a fresh temporary folder."""
    monkeypatch.setattr(log_utils, "LOG_DIR", tmp_path)
    return tmp_path
