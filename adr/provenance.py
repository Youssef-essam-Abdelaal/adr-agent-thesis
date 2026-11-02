"""
Which code produced a result. Every run directory gets this next to its config,
so any number in the thesis can be traced to an exact commit.
"""
from __future__ import annotations

import os
import platform
import subprocess
import time

from .config import ROOT


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def code_version() -> dict:
    return {
        "git_commit": _git("rev-parse", "HEAD") or "not a git repository",
        "git_describe": _git("describe", "--tags", "--always", "--dirty"),
        "uncommitted_changes": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "fake_llm_dry_run": os.environ.get("ADR_FAKE_LLM") == "1",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
