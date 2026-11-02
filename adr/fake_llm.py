"""
Fake model replies for dry runs: ADR_FAKE_LLM=1.

Exercises the whole pipeline (runner, browser, faults, every condition, analysis)
without calling any API or reading the key. Replies are random but valid JSON of
the right shape for each caller, seeded by the prompt so a dry run repeats
exactly. The numbers a dry run produces mean nothing; only crashes matter.
"""
from __future__ import annotations

import hashlib
import random
import re

TYPES = ["I", "II", "III", "IV", "V", "VI", "N"]
SYMPTOMS = ["none", "no_effect", "wrong_effect", "blocked", "unexpected_state"]


def _size(system: str, default=(480, 630)):
    m = re.search(r"(\d+)x(\d+) pixels", system)
    return (int(m.group(1)), int(m.group(2))) if m else default


def reply(system: str, text: str, tag: str) -> dict:
    rng = random.Random(hashlib.sha256((system[:200] + text).encode()).hexdigest())
    if tag == "agent":
        w, h = _size(system)
        r = rng.random()
        if r < 0.08:
            return {"thought": "fake: the task looks complete", "action": {"type": "done"}}
        if r < 0.35:
            return {"thought": "fake: type into the field", "action": {"type": "type", "x": rng.randrange(w),
                                                                        "y": rng.randrange(60, h), "text": "abc"}}
        return {"thought": "fake: click Submit", "action": {"type": "click", "x": rng.randrange(w), "y": rng.randrange(60, h)}}
    if tag == "judge":
        if "says its task is complete" in system:
            return {"criteria": [{"criterion": "fake", "met": False}], "complete": rng.random() < 0.5,
                    "confidence": 0.9, "evidence": "fake"}
        if "finalizes the task" in system:
            return {"is_final": rng.random() < 0.5, "ready": rng.random() < 0.7, "problem": "fake problem",
                    "fix": "fake fix", "confidence": 0.9}
        ok = rng.random() < 0.6
        return {"change": "fake", "expected": "fake", "ok": ok, "symptom": "none" if ok else rng.choice(SYMPTOMS[1:]),
                "confidence": round(rng.uniform(0.4, 1.0), 2), "evidence": "fake evidence"}
    if tag == "diag":
        if "what_went_wrong" in system and "FAILURE TYPES" not in system:
            return {"what_went_wrong": "fake", "advice": "fake advice", "no_failure": rng.random() < 0.1,
                    "confidence": 0.8}
        return {"evidence": "fake", "type": rng.choice(TYPES), "subtype": "dismissable_overlay",
                "awareness": "unaware", "behaviour": rng.choice(["progressing", "looping"]),
                "recoverability": rng.choice(["recoverable"] * 9 + ["irreversible"]),
                "intended_target": "the Submit button", "correct_value": "abc", "dismiss_target": "Accept",
                "correct_target": "the right option", "undo_by_reclick": rng.random() < 0.5,
                "screen_truth": "fake", "confidence": round(rng.uniform(0.3, 1.0), 2)}
    if tag == "ground":
        w, h = _size(system)
        return {"found": rng.random() < 0.9, "x": rng.randrange(w), "y": rng.randrange(h), "why": "fake"}
    return {}


def reply_text(messages: list, tag: str) -> str:
    """A UI-TARS-style text reply for dry runs (adr/agent_uitars.py)."""
    rng = random.Random(hashlib.sha256(repr([m.get("role") for m in messages]).encode()
                                       + str(len(messages)).encode()).hexdigest())
    r = rng.random()
    if r < 0.08:
        return "Thought: fake: the task looks complete.\nAction: finished(content='done')"
    if r < 0.3:
        return "Thought: fake: type into the focused field.\nAction: type(content='abc\\n')"
    if r < 0.4:
        return "Thought: fake: scroll.\nAction: scroll(start_box='(200,300)', direction='down')"
    return f"Thought: fake: click.\nAction: click(start_box='({rng.randrange(476)},{rng.randrange(60, 616)})')"
