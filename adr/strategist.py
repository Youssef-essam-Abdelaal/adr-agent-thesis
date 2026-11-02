"""
Recover: turn a diagnosis into a concrete fix.

Deterministic mapping, type (+ axes) -> recovery primitive, read from the
taxonomy file. The only model is the element finder, which locates an element on
the screen (the missed target, the right element, the pop-up's close control). A
budget and an escalation rule (the same type a third time -> replan) keep
recovery from looping.

Also holds the generic reflect-and-retry baselines used in RQ2.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image

from .agent import describe
from .config import Config
from .llm import LLM
from .taxonomy import Taxonomy

GROUND_SYSTEM = """You locate one element on a screenshot of a web page ({w}x{h} pixels).
Reply with JSON only: {{"found": true or false, "x": <int>, "y": <int>, "why": "<short>"}}
x, y is the centre of the element in screenshot pixels. If it is not visible, found = false."""
# Qwen3-VL points on a 0-1000 scale whatever the image size (cfg.ground_coords = "norm1000")
GROUND_SYSTEM_NORM = """You locate one element on a screenshot of a web page.
Reply with JSON only: {"found": true or false, "point_2d": [x, y], "why": "<short>"}
point_2d is the centre of the element on a 0-1000 scale: [0, 0] is the top-left corner, [1000, 1000] the bottom-right.
If it is not visible, found = false."""
GROUND_SCHEMA_NORM = {"type": "object", "required": ["found", "point_2d"],
                      "properties": {"found": {"type": "boolean"}, "why": {"type": "string"},
                                     "point_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}}}

GENERIC_FEEDBACK = ("Your last action may not have worked as intended. Look carefully at the current "
                    "screen, think about what went wrong, and try again.")
GENERIC_DONE = "The task does not look complete yet. Look carefully at the screen and continue."


@dataclass
class Plan:
    kind: str
    actions: List[dict] = field(default_factory=list)   # each: {"action": {...}, "wait": seconds}
    feedback: Optional[str] = None
    stop: bool = False
    note: str = ""


class Strategist:
    def __init__(self, cfg: Config, taxonomy: Optional[Taxonomy] = None):
        self.cfg = cfg
        self.tax = taxonomy or Taxonomy(cfg.path(cfg.taxonomy))
        self.llm = LLM.for_role(cfg, "ground")
        self.seen = Counter()

    def reset(self) -> None:
        self.seen = Counter()

    def plan(self, d: dict, verdict: dict, goal: str, last_action: dict, after: Image.Image,
             hints_only: bool = False) -> Plan:
        if d.get("uncertain"):        # the vote did not agree: do not guess a typed fix
            p = generic_plan(verdict)
            p.kind = "generic_fallback"
            return p
        t, sub = d.get("type", "V"), d.get("subtype", "")
        kind = self.tax.recovery(t, sub)
        self.seen[t] += 1
        looping = d.get("behaviour") == "looping" or verdict.get("symptom") == "loop"
        if t != "N" and (self.seen[t] >= 3 or (looping and kind == "wait_retry" and self.seen[t] >= 2)):
            kind = "replan"                       # the targeted fix did not hold: escalate
        if verdict.get("symptom") == "false_done":
            # the done-check is the specialist here: a failed check always vetoes
            return self._veto_done(d, verdict)
        if d.get("recoverability") == "irreversible" and kind not in ("none", "escalate"):
            return Plan("report_irreversible", feedback=(
                f"Your last step cannot be undone: {d.get('evidence', '')} Do not repeat it; continue carefully."))
        if hints_only:      # adr_hints ablation: the same typed decision, told to the agent instead of done
            return self._hint(kind, d, verdict, goal, last_action, after)
        return getattr(self, f"_{kind}", self._replan)(d, verdict, goal, last_action, after)

    def _hint(self, kind, d, v, goal, last, after) -> Plan:
        """Typed advice with no wrapper actions and no element finding."""
        if kind == "reground":
            target = d.get("intended_target") or "the element you meant to use"
            undo = (" It may also have changed something else; undo that first."
                    if v.get("symptom") == "wrong_effect" else "")
            return Plan("hint_reground", feedback=(
                f"Your last action landed off its target.{undo} Do it again, aiming at the centre of {target}."))
        if kind == "wait_retry":
            return Plan("hint_wait_retry", feedback="The page did not react to your last action. "
                                                    "Wait a moment, then repeat it.")
        if kind == "dismiss_retry":
            target = d.get("dismiss_target") or "its close button"
            return Plan("hint_dismiss_retry", feedback=(
                f"Something is covering the page. Close it first ({target}), then repeat your last action."))
        if kind == "fix_target_value":
            val, target = d.get("correct_value"), d.get("correct_target")
            if last.get("type") == "type" and val not in (None, "", "null"):
                return Plan("hint_fix_value", feedback=f"The value you typed is wrong. It should be {str(val)!r}. "
                                                       f"Retype it.")
            if target not in (None, "", "null"):
                undo = "Undo it, then select" if d.get("undo_by_reclick") else "Select"
                return Plan("hint_fix_target", feedback=f"You selected the wrong element. {undo} {target}.")
            return Plan("hint_fix_target", feedback=(
                f"Your last action used the wrong element or value: {d.get('evidence', '')} "
                f"{d.get('screen_truth', '')} Undo it if needed, then choose the correct one for the task."))
        if kind in ("replan", "correct_belief", "none", "escalate"):
            return getattr(self, f"_{kind}")(d, v, goal, last, after)
        return self._replan(d, v, goal, last, after)

    # recovery primitives, named by the "recovery" field of the taxonomy

    def _none(self, d, v, goal, last, after) -> Plan:
        return Plan("none", note="diagnosed as no failure; alarm suppressed")

    def _reground(self, d, v, goal, last, after) -> Plan:
        target = d.get("intended_target")
        if not target or "x" not in last:
            return self._replan(d, v, goal, last, after)
        hit = self.locate(after, target)
        if not hit:
            return self._replan(d, v, goal, last, after)
        redo = dict(last, x=hit[0], y=hit[1])
        acts = []
        undone = (self.cfg.adr_version >= 2 and last.get("type") == "click"
                  and v.get("symptom") == "wrong_effect" and d.get("undo_by_reclick"))
        if undone:   # the stray click toggled something: click it again to undo
            acts.append({"action": {"type": "click", "x": last["x"], "y": last["y"]}, "wait": 0})
        acts.append({"action": redo, "wait": 0.2 if acts else 0})
        extra = (" The supervisor also undid what the stray click changed." if undone else
                 " The missed action may also have changed something else; undo that if needed."
                 if v.get("symptom") == "wrong_effect" else "")
        return Plan("reground", acts,
                    f"Your last action landed off its target ({target}); the supervisor has now done it on the "
                    f"target for you, so do not repeat it.{extra} Continue with the next part of the task.")

    def _wait_retry(self, d, v, goal, last, after) -> Plan:
        if last.get("type") not in ("click", "type", "key"):
            return self._replan(d, v, goal, last, after)
        return Plan("wait_retry", [{"action": dict(last), "wait": self.cfg.wait_s}],
                    "The page did not respond to your last action; the supervisor waited and repeated it for you, "
                    "so do not repeat it. Continue with the next part of the task.")

    def _dismiss_retry(self, d, v, goal, last, after) -> Plan:
        target = d.get("dismiss_target") or "the button that closes the pop-up or banner (x, Accept, Close, No thanks)"
        hit = self.locate(after, target)
        if not hit:
            return Plan("dismiss_failed", feedback=(
                f"Something is covering the page: {d.get('evidence', '')} Close it first, then continue."))
        acts = [{"action": {"type": "click", "x": hit[0], "y": hit[1]}, "wait": 0}]
        if last.get("type") in ("click", "type"):
            acts.append({"action": dict(last), "wait": 0.3})
        return Plan("dismiss_retry", acts,
                    "A pop-up was covering the page and blocked your last action; the supervisor closed it and "
                    "repeated your action, so do not repeat it. Continue with the next part of the task.")

    def _escalate(self, d, v, goal, last, after) -> Plan:
        return Plan("escalate", stop=True, note=f"needs a person: {d.get('evidence', '')}")

    def _fix_target_value(self, d, v, goal, last, after) -> Plan:
        val = d.get("correct_value")
        if last.get("type") == "type" and "x" in last and val not in (None, "", "null"):
            return Plan("fix_value", [{"action": dict(last, text=str(val)), "wait": 0}],
                        f"The value you entered was wrong; the supervisor corrected it to {str(val)!r}. Continue with the next part of the task.")
        target = d.get("correct_target")
        if self.cfg.adr_version == 1:      # v1: a hint only
            return Plan("fix_target", feedback=(
                f"Your last action used the wrong element or value: {d.get('evidence', '')} "
                f"{d.get('screen_truth', '')} Undo it if needed, then choose the correct one for the task."))
        if last.get("type") == "click" and target not in (None, "", "null"):
            hit = self.locate(after, target)
            if hit:
                acts = []
                if d.get("undo_by_reclick") and "x" in last:
                    acts.append({"action": {"type": "click", "x": last["x"], "y": last["y"]}, "wait": 0})
                acts.append({"action": {"type": "click", "x": hit[0], "y": hit[1]}, "wait": 0.2 if acts else 0})
                return Plan("fix_target", acts,
                            f"Your last action selected the wrong element. The supervisor "
                            f"{'undid it and ' if len(acts) > 1 else ''}selected {target} for you, so do not repeat "
                            f"it. Continue with the next part of the task.")
        return Plan("fix_target_hint", feedback=(
            f"Your last action used the wrong element or value: {d.get('evidence', '')} "
            f"{d.get('screen_truth', '')} Undo it if needed, then choose the correct one for the task."))

    def _replan(self, d, v, goal, last, after) -> Plan:
        return Plan("replan", feedback=(
            f"Your current approach is not working: {d.get('evidence', '')} Re-read the task: \"{goal}\". "
            f"Do not repeat {describe(last)}; take a different approach."))

    def _correct_belief(self, d, v, goal, last, after) -> Plan:
        return Plan("correct_belief", feedback=(
            f"Look at the screen again. {d.get('evidence', '')} What the screen actually shows: "
            f"{d.get('screen_truth', '')}"))

    def _veto_done(self, d, v) -> Plan:
        unmet = "; ".join(v.get("unmet") or []) or v.get("evidence", "")
        return Plan("veto_done", feedback=(
            f"The task is NOT complete yet. Not yet satisfied: {unmet}. {d.get('screen_truth', '')} Continue."))

    # when the fix itself did not work

    def escalate(self, plan: Plan, check: dict, last: dict) -> Plan:
        """One more attempt of a different kind after a fix that the check says failed."""
        if plan.kind == "wait_retry" and last.get("type") in ("click", "type", "key"):
            return Plan("wait_retry_longer", [{"action": dict(last), "wait": self.cfg.wait_s * 2}], plan.feedback)
        if plan.kind == "dismiss_retry":
            acts = [{"action": {"type": "key", "key": "Escape"}, "wait": 0}]
            if last.get("type") in ("click", "type"):
                acts.append({"action": dict(last), "wait": 0.3})
            return Plan("dismiss_escape", acts, plan.feedback)
        return Plan(f"{plan.kind}_failed", feedback=(
            f"The supervisor tried to fix your last step but it did not work: {check.get('evidence', '')} "
            f"Look at the screen and fix it yourself before continuing."))

    def locate(self, shot: Image.Image, target: str):
        w, h = shot.size
        norm = self.cfg.ground_coords == "norm1000"
        system = GROUND_SYSTEM_NORM if norm else GROUND_SYSTEM.format(w=w, h=h)
        kw = {"schema": GROUND_SCHEMA_NORM} if norm else {}
        out = self.llm.ask(system, f"Find: {target}", [shot], tag="ground", **kw)
        if norm and isinstance(out.get("point_2d"), list) and len(out["point_2d"]) == 2:
            out["x"], out["y"] = out["point_2d"]
        if out.get("found") and "x" in out and "y" in out:
            try:
                x, y = float(out["x"]), float(out["y"])
                if norm:
                    x, y = x * w / 1000, y * h / 1000
                return int(min(max(x, 0), w - 1)), int(min(max(y, 0), h - 1))
            except (TypeError, ValueError):
                return None
        return None


SYMPTOM_TEXT = {"no_effect": "nothing changed on the screen",
                "wrong_effect": "the screen changed, but not in the way the task needs",
                "blocked": "something is covering the page",
                "unexpected_state": "the page is in a state that does not fit the task",
                "loop": "you are repeating the same action without progress"}


def evidence_plan(verdict: dict) -> Plan:
    """The retry_evidence baseline: the generic message plus what the Monitor saw. No type, no fix."""
    if verdict.get("symptom") == "false_done":
        unmet = "; ".join(verdict.get("unmet") or []) or verdict.get("evidence", "")
        return Plan("evidence_retry", feedback=f"The task does not look complete yet. Not yet satisfied: {unmet}. "
                                               f"Look carefully at the screen and continue.")
    sym = SYMPTOM_TEXT.get(verdict.get("symptom"), "something went wrong")
    return Plan("evidence_retry", feedback=(
        f"Your last action may not have worked as intended: {sym}. What the supervisor saw: "
        f"{verdict.get('evidence', '')} Look carefully at the current screen, think about what went wrong, "
        f"and try again."))


def untyped_plan(advice: dict, verdict: dict) -> Plan:
    """The adr_untyped ablation: a free-form diagnosis and advice, no taxonomy, no wrapper actions."""
    if verdict.get("symptom") == "false_done":
        unmet = "; ".join(verdict.get("unmet") or []) or verdict.get("evidence", "")
        return Plan("untyped_veto_done", feedback=f"The task is NOT complete yet. Not yet satisfied: {unmet}. "
                                                  f"{advice.get('advice', '')} Continue.")
    if advice.get("no_failure") is True:
        return Plan("untyped_none", note="advisor saw no failure")
    return Plan("untyped_advice", feedback=(
        f"Your last step did not work: {advice.get('what_went_wrong', '')} {advice.get('advice', '')}"))


def generic_plan(verdict: dict) -> Plan:
    """The retry baseline: the same detection, but one generic message instead of a typed fix."""
    if verdict.get("symptom") == "false_done":
        return Plan("generic_retry", feedback=GENERIC_DONE)
    return Plan("generic_retry", feedback=GENERIC_FEEDBACK)
