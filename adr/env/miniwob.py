"""
Live MiniWoB++ environment in a real Chrome, driven by Playwright.

Implements the three-method environment contract the wrapper relies on:
reset(task, seed, fault) -> goal, screenshot() -> image, execute(action) -> info,
plus status(final=False) for the benchmark's own success check (final=True once, at the end). Coordinates are in
screenshot pixels, the way a vision agent sees the page.
"""
from __future__ import annotations

import io
import sys
import time
from typing import Optional

from PIL import Image

from ..config import Config
from .faults import (LOADING_JS, OVERLAY_JS, OVERLAY_RANDOM_JS, SETUP_JS, SHIFT_JS, STAGNATE_JS, Fault,
                     corrupt, corrupt_random, randomize)


class MiniWoBEnv:
    def __init__(self, cfg: Config, browser):
        self.cfg = cfg
        self.browser = browser
        self.ctx = None
        self.page = None
        self.fault = Fault()
        self.goal = ""
        self.agent_actions = 0
        self.eligible_seen = 0      # eligible actions seen for the fault
        self.rect = (0.0, 0.0, 160.0, 210.0)

    def reset(self, task: str, seed: int, fault: Optional[Fault] = None) -> str:
        self.close()
        s = self.cfg.scale
        self.ctx = self.browser.new_context(viewport={"width": 360, "height": 320}, device_scale_factor=s)
        self.page = self.ctx.new_page()
        url = "file://" + str(self.cfg.path(self.cfg.miniwob_dir) / f"{task}.html")
        self.page.goto(url)
        self.page.wait_for_function("typeof core !== 'undefined' && core.cover_div !== null", timeout=15000)
        self.page.evaluate(f"""() => {{
            Math.seedrandom('{task}-{seed}');
            core.EPISODE_MAX_TIME = 1000000000;
            core.startEpisodeReal();
        }}""")
        self.page.evaluate(SETUP_JS)
        self.page.wait_for_timeout(300)
        goal = self.page.evaluate("core.getUtterance()")
        # a few tasks return {"utterance": ..., "fields": ...}; the agent only gets the sentence
        self.goal = goal.get("utterance", "") if isinstance(goal, dict) else str(goal)
        r = self.page.evaluate("(() => { const b = document.getElementById('wrap').getBoundingClientRect();"
                               " return [b.left, b.top, b.width, b.height]; })()")
        self.rect = tuple(float(v) for v in r)
        self.fault = fault or Fault()
        if self.cfg.fault_randomize and self.fault.kind != "none" and not self.fault.params:
            self.fault.params = randomize(self.fault.kind, task, seed)
        self.agent_actions = 0
        self.eligible_seen = 0
        return self.goal

    def close(self) -> None:
        if self.ctx is not None:
            try:
                self.ctx.close()
            except Exception:
                pass
        self.ctx = self.page = None

    def screenshot(self) -> Image.Image:
        x, y, w, h = self.rect
        png = self.page.screenshot(clip={"x": x, "y": y, "width": w, "height": h})
        return Image.open(io.BytesIO(png)).convert("RGB")

    def size(self):
        return int(self.rect[2] * self.cfg.scale), int(self.rect[3] * self.cfg.scale)

    def status(self, final: bool = False) -> dict:
        d = self.page.evaluate("(() => ({done: WOB_DONE_GLOBAL, raw: WOB_RAW_REWARD_GLOBAL,"
                               " reason: (typeof WOB_REWARD_REASON === 'undefined') ? null : WOB_REWARD_REASON}))()")
        d["success"] = bool(d["done"] and (d["raw"] or 0) > 0)
        return d

    def _to_css(self, x, y):
        s = self.cfg.scale
        return self.rect[0] + float(x) / s, self.rect[1] + float(y) / s

    def _to_img(self, cx, cy):
        s = self.cfg.scale
        return round((cx - self.rect[0]) * s), round((cy - self.rect[1]) * s)

    def execute(self, action: dict, source: str = "agent", step: int = -1) -> dict:
        """
        Run one action. Returns what was actually executed (after any fault) so
        the trace records what really happened, as it would in a real system.
        """
        a = dict(action)
        fired = False
        if source == "agent":
            self.agent_actions += 1
            f = self.fault
            if f.kind != "none" and not f.fired and f.eligible(a):
                self.eligible_seen += 1
                if self.eligible_seen >= f.at:
                    fired = True
                    f.fired, f.fired_step = True, step
                    a = self._apply_fault(f, a)
        err = ""
        try:
            self._run(a)
        except Exception as e:  # a bad action is the agent's problem, not a crash
            err = f"{type(e).__name__}: {str(e)[:120]}"
        self.page.wait_for_timeout(self.cfg.settle_ms)
        return {"executed": a, "fault_fired": fired, "error": err}

    def _apply_fault(self, f: Fault, a: dict) -> dict:
        p = f.params
        if p:
            return self._apply_random(f, a, p)
        if f.kind == "loading":
            ms = 4000
            self.page.evaluate(LOADING_JS, ms)
            f.detail = f"loading cover for {ms}ms"
            return a
        if f.kind == "layout_shift":
            self.page.evaluate(SHIFT_JS, 26)
            f.detail = "content shifted down 26px just before the click"
            return a
        if f.kind == "grounding":
            cx, cy = self._to_css(a["x"], a["y"])
            off = self.cfg.grounding_offset * (1 if f.variant % 2 == 0 else -1)
            ny = cy + off
            # keep the click in the task area, below the instruction bar at the top (50 CSS px)
            top, bottom = self.rect[1] + 50, self.rect[1] + self.rect[3] - 2
            if not (top <= ny <= bottom):
                ny = cy - off
            a["x"], a["y"] = self._to_img(cx, ny)
            f.detail = f"click displaced by {ny - cy:+.0f}px"
        elif f.kind == "stagnation":
            self.page.evaluate(STAGNATE_JS, self.cfg.stagnation_ms)
            f.detail = f"input ignored for {self.cfg.stagnation_ms}ms"
        elif f.kind == "occlusion":
            cx, cy = self._to_css(a["x"], a["y"])
            self.page.evaluate(OVERLAY_JS, [cx, cy, f.variant])
            f.detail = ["cookie banner", "newsletter pop-up", "promo pop-up"][f.variant % 3]
        elif f.kind == "wrong_value":
            old = str(a.get("text", ""))
            a["text"] = corrupt(old, f.variant)
            f.detail = f"typed {a['text']!r} instead of {old!r}"
        return a

    def _apply_random(self, f: Fault, a: dict, p: dict) -> dict:
        if f.kind == "grounding":
            cx, cy = self._to_css(a["x"], a["y"])
            x0, y0, w, h = self.rect
            # keep the click in the task area, below the instruction bar at the top (50 CSS px)
            nx = min(max(cx + p["dx"], x0 + 2), x0 + w - 2)
            ny = min(max(cy + p["dy"], y0 + 50), y0 + h - 2)
            a["x"], a["y"] = self._to_img(nx, ny)
            f.detail = f"click displaced by ({nx - cx:+.0f}, {ny - cy:+.0f})px"
        elif f.kind == "stagnation":
            self.page.evaluate(STAGNATE_JS, p["ms"])
            f.detail = f"input ignored for {p['ms']}ms"
        elif f.kind == "occlusion":
            cx, cy = self._to_css(a["x"], a["y"])
            self.page.evaluate(OVERLAY_RANDOM_JS, [cx, cy, p])
            f.detail = f"pop-up style {p['style']}, height {p['h']}px, close at {p['corner']}{', full cover' if p['full'] else ''}"
        elif f.kind == "wrong_value":
            old = str(a.get("text", ""))
            a["text"] = corrupt_random(old, p["variant"])
            p["original"], p["corrupted"] = old, a["text"]
            f.detail = f"typed {a['text']!r} instead of {old!r}"
        elif f.kind == "loading":
            self.page.evaluate(LOADING_JS, p["ms"])
            f.detail = f"loading cover for {p['ms']}ms"
        elif f.kind == "layout_shift":
            self.page.evaluate(SHIFT_JS, p["px"])
            f.detail = f"content shifted down {p['px']}px just before the click"
        return a

    def _run(self, a: dict) -> None:
        t = a.get("type")
        kb = self.page.keyboard
        if t == "click":
            cx, cy = self._to_css(a["x"], a["y"])
            if a.get("double"):                 # UI-TARS left_double
                self.page.mouse.dblclick(cx, cy)
            else:
                self.page.mouse.click(cx, cy)
        elif t == "type":
            if "x" in a:
                cx, cy = self._to_css(a["x"], a["y"])
                self.page.mouse.click(cx, cy)
                self.page.wait_for_timeout(80)
                kb.press("Meta+A")
                kb.press("Backspace")
            kb.type(str(a.get("text", "")), delay=8)
            if a.get("submit"):                 # UI-TARS type(content='...\n')
                kb.press("Enter")
        elif t == "key":
            key = str(a.get("key", "Enter"))
            if sys.platform == "darwin":        # on macOS, shortcuts such as select-all use Cmd
                key = key.replace("Control+", "Meta+")
            kb.press(key)
        elif t == "scroll":
            cx, cy = self._to_css(a.get("x", 240), a.get("y", 400))
            self.page.mouse.move(cx, cy)
            self.page.mouse.wheel(0, float(a.get("dy", 100)))
        elif t == "wait":
            time.sleep(1.5)
        elif t in ("done", "noop", None):
            pass
        else:
            raise ValueError(f"unknown action type {t!r}")
