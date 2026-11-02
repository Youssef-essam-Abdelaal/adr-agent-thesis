"""
Verify: did the last action do what it should? Reports the symptom, never the cause.

Layer 1 is free: pixel change, repeated actions, revisited screens.
Layer 2 is a VLM judge that works evidence-first (describe the change, state the
expected effect, then decide), with a red circle on the click and a blue box on
the changed region. When the agent says "done", the done-check turns the task into
checkable criteria and rejects the claim unless all are visibly met.

judge_policy = every_step catches silent wrong-effect failures (wrong value,
wrong item) that pixels cannot see; suspicious_only is the cheap mode that only
calls the judge when Layer 1 fires.
"""
from __future__ import annotations

import re
from typing import List, Optional

from PIL import Image

from .agent import describe
from .config import Config
from .llm import LLM
from .vision import action_point, box, diff_info, mark_click, screen_hash

JUDGE_SYSTEM = """You check ONE step of a GUI agent working on a web page.
You get the task, the agent's recent actions, its reasoning and action for this step, a BEFORE screenshot (a red circle marks where the action landed, if it has a position) and an AFTER screenshot (a blue box marks the region that changed; no box means nothing changed).
Work evidence first: describe what visibly changed, state what this action should have produced if it worked, then decide.
Judge against the TASK, not only the click: a wrong item selected, a wrong or misspelled value, or a pop-up now covering the page is NOT ok even though something happened. A focus change or a partly filled form on the way to the goal IS ok.
Report only WHAT happened (the symptom), not why.
Reply with JSON only:
{"change": "<what visibly changed, or 'nothing'>",
 "expected": "<what this action should have produced>",
 "ok": true or false,
 "symptom": "none" | "no_effect" | "wrong_effect" | "blocked" | "unexpected_state",
 "confidence": <0..1, how sure you are about ok>,
 "evidence": "<one sentence pointing at what in the screenshots supports the decision>"}
no_effect = nothing relevant changed. wrong_effect = something changed but not what the task needs (wrong item, wrong value, wrong field, landed elsewhere). blocked = a pop-up, banner, dialog or overlay covers the page or the target. unexpected_state = error, wrong page, or a state that does not fit the task. none = the step did what it should."""

DONE_SYSTEM = """A GUI agent says its task is complete. Check that claim conservatively from the screenshot.
First turn the task into 1-3 concrete success criteria that can be checked on screen, then check each.
If any criterion is not visibly met, or something still has to be submitted or confirmed, the task is NOT complete.
Reply with JSON only:
{"criteria": [{"criterion": "...", "met": true or false}],
 "complete": true or false, "confidence": <0..1>, "evidence": "<one sentence>"}"""


PRECOMMIT_SYSTEM = """A GUI agent may be about to take an action that finalizes the task and cannot be undone. Check BEFORE it happens. A red circle marks where it is about to click, if it clicks.
0. First decide if THIS action will finish the task: the red marker is on a control that submits, sends, logs in, confirms or searches (or the action presses Enter to submit a form), OR it is the click the task itself asks for as its last step (the button, link, item or tab the task names). Selecting an option, ticking a box, opening a menu or tab on the way, or focusing a field is NOT final, even if the agent plans to submit afterwards. If it is not final, reply is_final=false, ready=true and stop.
1. If it is final, check the red marker is on exactly the element the task requires (the right link, button or item, not a similar one next to it).
2. Turn the task into the concrete conditions that must hold on screen before finishing: the right value in each field, exactly the right items selected and nothing extra, the right option chosen, earlier required steps done. Check each against what is visible NOW.
Block only for a clear, visible problem. If everything the task needs is in place, it is ready.
Reply with JSON only:
{"is_final": true or false, "on_right_element": true or false,
 "conditions": [{"condition": "...", "met": true or false}],
 "ready": true or false, "problem": "<what is wrong, or empty>", "fix": "<concrete steps to fix it before finalizing, or empty>",
 "confidence": <0..1>}"""


def is_commit(thought: str, action: dict, words: str) -> bool:
    """Does this action finalize the task? Clicks the agent describes as submitting, or Enter."""
    if action.get("type") == "key" and str(action.get("key", "")).lower() == "enter":
        return True
    if action.get("type") != "click":
        return False
    pat = r"\b(" + "|".join(re.escape(w.strip()) for w in words.split(",") if w.strip()) + r")\b"
    return re.search(pat, thought or "", re.I) is not None


def same_action(a: dict, b: dict, tol: float = 12) -> bool:
    if a.get("type") != b.get("type"):
        return False
    pa, pb = action_point(a), action_point(b)
    if pa and pb and (abs(pa[0] - pb[0]) > tol or abs(pa[1] - pb[1]) > tol):
        return False
    return a.get("text") == b.get("text") and a.get("key") == b.get("key")


class Monitor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.llm = LLM.for_role(cfg, "judge")
        self.reset()

    def reset(self) -> None:
        self.actions: List[dict] = []
        self.hashes: List[str] = []
        self.no_effect_run = 0

    def check(self, goal: str, before: Image.Image, after: Image.Image, thought: str,
              action: dict, history: List[dict]) -> dict:
        cfg = self.cfg
        sig = diff_info(before, after, cfg.scale)
        acting = action.get("type") in ("click", "type", "key")
        no_effect = acting and sig["changed_px"] < cfg.no_change_px
        self.no_effect_run = self.no_effect_run + 1 if no_effect else 0

        repeats = 1
        for prev in reversed(self.actions):
            if same_action(prev, action):
                repeats += 1
            else:
                break
        self.actions.append(action)
        h = screen_hash(after)
        seen = self.hashes.count(h)
        self.hashes.append(h)
        # repetition alone is not a loop (undoing a mis-click repeats an earlier click);
        # it must come with no progress: nothing changed, or a screen already seen
        loop = acting and ((repeats >= 2 and self.no_effect_run >= 2)
                           or (repeats >= cfg.loop_repeats and seen >= 1))
        signals = {"changed_px": sig["changed_px"], "bbox": sig["bbox"], "no_effect": no_effect,
                   "repeats": repeats, "screen_seen_before": seen, "loop": loop}

        judge = None
        if cfg.judge_policy == "every_step" or no_effect or loop:
            judge = self._judge(goal, before, after, thought, action, history, sig["bbox"])

        flagged, symptom, source, conf, evidence = False, "none", "", 0.0, ""
        if judge and judge.get("ok") is False and float(judge.get("confidence", 0) or 0) >= cfg.flag_confidence:
            flagged, source = True, "judge"
            symptom = judge.get("symptom") if judge.get("symptom") not in (None, "none") else "wrong_effect"
            conf, evidence = float(judge.get("confidence", 0)), judge.get("evidence", "")
        if loop:
            if not flagged:
                symptom, conf, evidence = "loop", 0.9, f"same action repeated {repeats}x without progress"
            flagged, source = True, (source + "+loop").strip("+")
        return {"flagged": flagged, "symptom": symptom, "source": source, "confidence": conf,
                "evidence": evidence, "judge": judge, "signals": signals}

    def check_done(self, goal: str, shot: Image.Image, thought: str) -> dict:
        out = self.llm.ask(DONE_SYSTEM, f"Task: {goal}\nAgent's final reasoning: {thought}", [shot], tag="judge")
        complete = out.get("complete")
        conf = float(out.get("confidence", 0) or 0)
        flagged = complete is False and conf >= self.cfg.flag_confidence
        unmet = [c.get("criterion", "") for c in out.get("criteria", []) if isinstance(c, dict) and c.get("met") is False]
        return {"flagged": flagged, "symptom": "false_done" if flagged else "none", "source": "done_check",
                "confidence": conf, "evidence": out.get("evidence", ""), "unmet": unmet, "judge": out,
                "signals": {}}

    def check_commit(self, goal: str, shot: Image.Image, thought: str, action: dict, history: List[dict]) -> dict:
        """Prevent, the click check: before an irreversible step, check that the screen is ready for it."""
        p = action_point(action)
        img = mark_click(shot, *p) if p else shot
        recent = "\n".join(f"- {describe(s['action'])}" for s in history[-5:]) or "- (none)"
        text = (f"Task: {goal}\nActions so far:\n{recent}\n"
                f"About to finalize. Reasoning: {thought}\nAction: {describe(action)}")
        out = self.llm.ask(PRECOMMIT_SYSTEM, text, [img], tag="judge")
        conf = float(out.get("confidence", 0) or 0)
        blocked = (out.get("is_final") is not False and out.get("ready") is False
                   and conf >= self.cfg.flag_confidence)
        return {"blocked": blocked, "is_final": out.get("is_final"), "problem": out.get("problem", ""),
                "fix": out.get("fix", ""),
                "confidence": conf, "judge": out}

    def _judge(self, goal, before, after, thought, action, history, bbox) -> Optional[dict]:
        p = action_point(action)
        b_img = mark_click(before, *p) if p else before
        a_img = box(after, bbox)
        recent = "\n".join(f"- {describe(s['action'])}" for s in history[-4:-1]) or "- (none)"
        text = (f"Task: {goal}\nRecent earlier actions:\n{recent}\n"
                f"This step. Reasoning: {thought}\nAction: {describe(action)}")
        return self.llm.ask(JUDGE_SYSTEM, text, [b_img, a_img], tag="judge")
