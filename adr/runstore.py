"""
Where run results live, and how they survive interruptions.

Every run folder is named `<run name>__<fingerprint>`. The fingerprint is a short
hash of everything that can change a result: the code that runs the experiment,
the taxonomy file and every setting (except ones that cannot change a result, such
as the number of parallel workers). So:

- same code and settings  -> same folder -> a rerun resumes it
- any implementation change -> a new folder -> old and new results never mix

Results are written atomically (temporary file, then rename), so an interrupted
write can never leave a half file that looks finished.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from .config import ROOT

# code that decides what happens in a live episode
LIVE_FILES = ["adr/agent.py", "adr/config.py", "adr/llm.py", "adr/fake_llm.py", "adr/vision.py", "adr/monitor.py",
              "adr/diagnostician.py", "adr/strategist.py", "adr/wrapper.py", "adr/taxonomy.py",
              "adr/env/miniwob.py", "adr/env/faults.py", "adr/experiment.py",
              # the UI-TARS adapter, OSWorld and ADR v3
              "adr/agent_uitars.py", "adr/env/osworld.py", "adr/v3/monitor.py", "adr/v3/diagnostician.py",
              "adr/v3/strategist.py", "adr/v3/wrapper.py"]
# code that decides an offline RQ1 result
RQ1_FILES = ["adr/config.py", "adr/llm.py", "adr/fake_llm.py", "adr/vision.py", "adr/diagnostician.py",
             "adr/taxonomy.py", "adr/rq1_offline.py", "adr/rq1_agentic.py"]
# settings that cannot change a result
VOLATILE = {"workers"}


def fingerprint(files: Iterable[str], settings: dict, extra: Optional[dict] = None) -> str:
    h = hashlib.sha256()
    for rel in sorted(files):
        p = ROOT / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.exists() else b"missing")
    tax = settings.get("taxonomy")
    if tax:
        tp = ROOT / tax
        h.update(tp.read_bytes() if tp.exists() else b"missing")
    clean = {k: v for k, v in settings.items() if k not in VOLATILE and not k.startswith("_")}
    h.update(json.dumps(clean, sort_keys=True, default=str).encode())
    h.update(json.dumps(extra or {}, sort_keys=True, default=str).encode())
    h.update(b"fake" if os.environ.get("ADR_FAKE_LLM") == "1" else b"real")
    return h.hexdigest()[:8]


def run_folder(base: Path, name: str, fp: str) -> Path:
    return base / f"{name}__{fp}"


def other_versions(base: Path, name: str, fp: str):
    """Folders of earlier versions of the same run name."""
    return sorted(p for p in base.glob(f"{name}__*") if p.is_dir() and p.name != f"{name}__{fp}")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_result(path: Path) -> Optional[dict]:
    """A saved result, or None if it is missing or unreadable (then it is redone)."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def keep_awake() -> None:
    """macOS: keep the machine awake while this process runs (no-op elsewhere)."""
    if shutil.which("caffeinate"):
        try:
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


class CaseStore:
    """
    One file per finished case for offline runs (RQ1). A rerun skips saved cases;
    cases that failed (for example an API error) are not saved, so they are retried.
    """

    def __init__(self, folder: Path):
        self.dir = Path(folder) / "cases"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, cid: str) -> Path:
        return self.dir / (hashlib.sha256(cid.encode()).hexdigest()[:16] + ".json")

    def has(self, cid: str) -> bool:
        return read_result(self._path(cid)) is not None

    def save(self, cid: str, record: dict) -> None:
        atomic_write(self._path(cid), json.dumps(record, indent=1, default=str))

    def all(self) -> list:
        return [r for r in (read_result(p) for p in sorted(self.dir.glob("*.json"))) if r is not None]


def resolve(base: Path, name: str) -> Optional[Path]:
    """A run by name: the exact folder if it exists, else the newest `<name>__<fingerprint>` folder."""
    p = Path(base) / name
    if p.exists():
        return p
    found = sorted(Path(base).glob(f"{name}__*"), key=lambda q: q.stat().st_mtime)
    return found[-1] if found else None
