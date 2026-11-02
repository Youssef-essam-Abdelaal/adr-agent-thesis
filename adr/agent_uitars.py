"""
UI-TARS-1.5-7B as the base agent (an open model, served by vLLM).

UI-TARS has its own prompt, conversation format and action language, so it gets this small adapter; the wrapper
around it does not change. The prompt is UI-TARS's computer-use prompt in the start_box form used by the official
OSWorld runs of UI-TARS-1.5-7B. The model sees its last few screenshots and replies, and answers

    Thought: ...
    Action: click(start_box='(x,y)')

with coordinates on the image it saw. Qwen2.5-VL's processor resizes every screenshot to multiples of 28 pixels
(smart_resize), so coordinates are scaled back to the screenshot (checked against the OSWorld logs: model (1134,184)
on a 1920x1080 screen = pyautogui (1126.96, 181.98)).

Mapping to our actions: click, left_double (double click), right_single (a click marked button=right: a right-click
in OSWorld, a plain click in MiniWoB), type (into the field clicked just before, else the focused one; a trailing
\\n also presses Enter), hotkey, scroll up/down, wait, finished (done), call_user (gives up: done, marked
infeasible). drag and sideways scrolls have no counterpart in our environment and become wait.
"""
from __future__ import annotations

import base64
import math
import re
from typing import List, Optional, Tuple

from PIL import Image

from .agent import Proposal
from .config import Config
from .llm import LLM, to_png_bytes

PROMPT = r"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

FACTOR, MIN_PIXELS, MAX_PIXELS = 28, 3136, 12845056     # UI-TARS-1.5-7B preprocessor_config.json
HISTORY_IMAGES = 4                                     # earlier screenshots kept; with the current one = 5 per call
KEYS = {"enter": "Enter", "return": "Enter", "tab": "Tab", "backspace": "Backspace", "space": "Space",
        "esc": "Escape", "escape": "Escape", "delete": "Delete", "up": "ArrowUp", "down": "ArrowDown",
        "left": "ArrowLeft", "right": "ArrowRight", "arrowup": "ArrowUp", "arrowdown": "ArrowDown",
        "arrowleft": "ArrowLeft", "arrowright": "ArrowRight", "pageup": "PageUp", "pagedown": "PageDown",
        "home": "Home", "end": "End", "ctrl": "Control", "control": "Control", "alt": "Alt", "shift": "Shift",
        "cmd": "Meta", "command": "Meta", "meta": "Meta", "win": "Meta"}


def smart_resize(h: int, w: int) -> Tuple[int, int]:
    """Qwen2.5-VL's resize: both sides multiples of 28, total pixels within [MIN_PIXELS, MAX_PIXELS]."""
    hb, wb = max(FACTOR, round(h / FACTOR) * FACTOR), max(FACTOR, round(w / FACTOR) * FACTOR)
    if hb * wb > MAX_PIXELS:
        beta = math.sqrt(h * w / MAX_PIXELS)
        hb, wb = math.floor(h / beta / FACTOR) * FACTOR, math.floor(w / beta / FACTOR) * FACTOR
    elif hb * wb < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (h * w))
        hb, wb = math.ceil(h * beta / FACTOR) * FACTOR, math.ceil(w * beta / FACTOR) * FACTOR
    return hb, wb


def _point(s: str) -> Optional[Tuple[float, float]]:
    """The first point in an argument: (x,y), (x1,y1,x2,y2) or [x1,y1,x2,y2] (box centre), or <point>x y</point>."""
    m = re.search(r"<point>\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*</point>", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"[\(\[]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*(?:,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*)?[\)\]]", s)
    if not m:
        return None
    x, y = float(m.group(1)), float(m.group(2))
    if m.group(3) is not None:
        x, y = (x + float(m.group(3))) / 2, (y + float(m.group(4))) / 2
    return x, y


def _arg(call: str, name: str) -> Optional[str]:
    m = re.search(name + r"\s*=\s*'((?:\\.|[^'\\])*)'", call, re.S) or re.search(name + r'\s*=\s*"((?:\\.|[^"\\])*)"', call, re.S)
    if not m:
        return None
    return m.group(1).replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")


def parse(text: str, w: int, h: int) -> Tuple[str, dict]:
    """UI-TARS reply -> (thought, action in our format) for a w x h screenshot."""
    m = re.search(r"Thought:(.*?)(?:\n\s*Action:|$)", text or "", re.S)
    thought = (m.group(1) if m else "").strip()
    m = re.search(r"Action:(.*)", text or "", re.S)
    call = (m.group(1) if m else text or "").strip()
    name = (re.match(r"\s*(\w+)", call) or re.match("", "")).group(0).strip().lower()
    rh, rw = smart_resize(h, w)

    def at(arg_names=("start_box", "point", "start_point")) -> Optional[dict]:
        for a in arg_names:
            v = _arg(call, a)
            p = _point(v) if v is not None else None
            if p:
                return {"x": int(min(max(p[0] * w / rw, 0), w - 1)), "y": int(min(max(p[1] * h / rh, 0), h - 1))}
        p = _point(call)
        return {"x": int(min(max(p[0] * w / rw, 0), w - 1)), "y": int(min(max(p[1] * h / rh, 0), h - 1))} if p else None

    if name in ("click", "left_single", "right_single", "left_double"):
        xy = at()
        if not xy:
            return thought, {"type": "wait"}
        act = {"type": "click", **xy}
        if name == "left_double":
            act["double"] = True
        if name == "right_single":
            act["button"] = "right"
        return thought, act
    if name == "type":
        content = _arg(call, "content") or ""
        act = {"type": "type", "text": content.rstrip("\n")}
        if content.endswith("\n"):
            act["submit"] = True
        return thought, act
    if name in ("hotkey", "press", "keydown"):
        keys = (_arg(call, "key") or _arg(call, "hotkey") or "").strip().lower().replace("+", " ").split()
        if not keys:
            return thought, {"type": "wait"}
        return thought, {"type": "key", "key": "+".join(KEYS.get(k, k if len(k) == 1 else k.capitalize()) for k in keys)}
    if name == "scroll":
        direction = (_arg(call, "direction") or "down").lower()
        xy = at() or {"x": w // 2, "y": h // 2}
        if direction not in ("down", "up"):
            return thought, {"type": "wait"}
        return thought, {"type": "scroll", **xy, "dy": (h // 2) * (1 if direction == "down" else -1)}
    if name == "finished":
        return thought, {"type": "done"}
    if name == "call_user":                   # gives up: OSWorld's FAIL (scored as right on infeasible tasks)
        return thought, {"type": "done", "infeasible": True}
    return thought, {"type": "wait"}          # wait(), drag, or anything unparseable


def _url(img: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(to_png_bytes(img)).decode()


class UITarsAgent:
    """Same interface as adr.agent.Agent: act(shot, goal, history, feedback) -> Proposal."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.llm = LLM.for_role(cfg, "agent")
        self.turns: List[Tuple[Image.Image, str]] = []    # (screenshot seen, raw reply) of this episode

    def act(self, shot: Image.Image, goal: str, history: List[dict],
            feedback: Optional[str] = None) -> Proposal:
        if not history:                     # a new episode
            self.turns = []
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "text", "text": PROMPT.format(instruction=goal)}]}]
        for i, (im, reply) in enumerate(self.turns):
            if i >= len(self.turns) - HISTORY_IMAGES:
                msgs.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": _url(im)}}]})
            msgs.append({"role": "assistant", "content": reply})
        last = [{"type": "image_url", "image_url": {"url": _url(shot)}}]
        if feedback:
            last.insert(0, {"type": "text", "text": f"Message from your supervisor about the last step: {feedback}"})
        msgs.append({"role": "user", "content": last})
        reply = self.llm.chat(msgs, tag="agent")
        self.turns.append((shot, reply))
        w, h = shot.size
        thought, action = parse(reply, w, h)
        prev = history[-1]["action"] if history else {}
        if action["type"] == "type" and prev.get("type") == "click" and "x" in prev:
            # UI-TARS clicks a field, then types. Our type action names the field: MiniWoB clicks it and replaces its
            # content (as for the JSON agent), OSWorld types where the focus already is, and in both the judge can
            # mark the field and ADR can retype a wrong value there.
            action.update(x=prev["x"], y=prev["y"])
        return Proposal(thought or reply[:200], action, {"text": reply})
