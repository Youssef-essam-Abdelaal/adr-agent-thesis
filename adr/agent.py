"""
The base agent being wrapped: a plain vision-only GUI agent.

It sees the current screenshot, the task and a text list of its own previous
actions, and returns one action. It does not check its own results. The
wrapper never changes it; the only channel the wrapper has is an optional
feedback message added to the next prompt, which any agent that takes text
input can receive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image

from .config import Config
from .llm import LLM

SYSTEM = """You are a GUI agent that completes tasks on a web page by looking at screenshots.
The screenshot is {w}x{h} pixels; every coordinate you give is a pixel position in it.
Reply with ONE JSON object and nothing else:
{{"thought": "<what you see and what you do next, 1-2 short sentences>",
  "action": <one action>}}
Actions:
  {{"type": "click", "x": <int>, "y": <int>}}
  {{"type": "type", "x": <int>, "y": <int>, "text": "<text>"}}   clicks the field at x,y and replaces its content
  {{"type": "key", "key": "Enter"}}                              also Tab, Backspace, ArrowDown, ArrowUp, Space
  {{"type": "scroll", "x": <int>, "y": <int>, "dy": <pixels, positive = down>}}
  {{"type": "wait"}}
  {{"type": "done"}}                                             only when the task is fully complete
Aim at the centre of the element you want. One action per reply."""

ALLOWED = {"click", "type", "key", "scroll", "wait", "done"}


@dataclass
class Proposal:
    thought: str
    action: dict
    raw: dict = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.action.get("type") == "done"


def describe(action: dict) -> str:
    t = action.get("type")
    if t == "click":
        return f"click({action.get('x')}, {action.get('y')})"
    if t == "type":
        return f"type({action.get('x')}, {action.get('y')}, {action.get('text', '')!r})"
    if t == "key":
        return f"key({action.get('key')})"
    if t == "scroll":
        return f"scroll({action.get('x')}, {action.get('y')}, dy={action.get('dy')})"
    return str(t)


class Agent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.llm = LLM.for_role(cfg, "agent")

    def act(self, shot: Image.Image, goal: str, history: List[dict],
            feedback: Optional[str] = None) -> Proposal:
        w, h = shot.size
        lines = [f"Task: {goal}", ""]
        if history:
            lines.append("Your previous actions:")
            for i, s in enumerate(history[-8:], start=max(1, len(history) - 7)):
                lines.append(f"{i}. {describe(s['action'])}  ({s.get('thought', '')[:140]})")
        else:
            lines.append("No previous actions.")
        if feedback:
            lines += ["", f"Message from your supervisor about the last step: {feedback}"]
        lines += ["", "What is the next action?"]
        out = self.llm.ask(SYSTEM.format(w=w, h=h), "\n".join(lines), [shot], tag="agent")
        return self._parse(out, w, h)

    @staticmethod
    def _parse(out: dict, w: int, h: int) -> Proposal:
        a = out.get("action") if isinstance(out.get("action"), dict) else {}
        t = a.get("type")
        if t not in ALLOWED:
            return Proposal(out.get("thought", "unparseable reply"), {"type": "wait"}, out)
        act = {"type": t}
        if t in ("click", "type", "scroll") and "x" in a and "y" in a:
            try:
                act["x"] = int(min(max(float(a["x"]), 0), w - 1))
                act["y"] = int(min(max(float(a["y"]), 0), h - 1))
            except (TypeError, ValueError):
                return Proposal(out.get("thought", ""), {"type": "wait"}, out)
        elif t in ("click", "scroll"):
            return Proposal(out.get("thought", ""), {"type": "wait"}, out)
        if t == "type":
            act["text"] = str(a.get("text", ""))
        if t == "key":
            act["key"] = str(a.get("key", "Enter"))
        if t == "scroll":
            act["dy"] = int(a.get("dy", 100) or 100)
        return Proposal(str(out.get("thought", "")), act, out)


def make_agent(cfg: Config):
    """The base agent named by cfg.agent_kind: json (this file) or uitars (adr/agent_uitars.py)."""
    if cfg.agent_kind == "uitars":
        from .agent_uitars import UITarsAgent
        return UITarsAgent(cfg)
    return Agent(cfg)
