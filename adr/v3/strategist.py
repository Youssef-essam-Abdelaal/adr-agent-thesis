"""
ADR v3 Strategist (v2: adr/strategist.py). Same type-to-fix table; the fix for type II changes.

v2's wait_retry repeated the last action after a wait and, if the check after the fix still failed, repeated it
again after a longer wait, telling the agent "the page did not respond". That repeated clicks on links and fields
that had worked, and the message made the agent undo correct steps.

v3: wait_check waits, looks at the screen again and repeats the action only if the screen is still unchanged
(the wrapper decides; adr/v3/wrapper.py). A click the Diagnostician marks as a toggle (undo_by_reclick) is never
repeated, since repeating it would undo it. If the one repeat does not help, the agent is asked to try another way
instead of a second, longer repeat. The agent is told that the page did not react only when the look confirmed it.

With ground_fields, the field is located before a value is retyped, since the type action can carry the point of an
earlier click somewhere else. With diag_rules, a re-ground for interaction mechanics repeats the click as a double
click (a single click in a file list only selects).
"""
from __future__ import annotations

from ..agent import describe
from ..strategist import Plan, Strategist

REPEATED = ("The page did not react to your last action; the supervisor waited, checked and repeated it once, so do "
            "not repeat it. Continue with the next part of the task.")
LATE = ("The screen did change after your last action, so the supervisor did not repeat it. Look at the current "
        "screen and continue with the next part of the task.")
RETYPED = ("Your text did not reach the field, so the supervisor waited and typed it again for you. Check that the field "
           "now shows the text before you continue.")
TOGGLE = ("Your last action may not have shown its effect yet. Look at the current screen before acting again, and "
          "do not click the same option or toggle again unless the screen shows it is still needed.")


# use_axes: what each axis of taxonomy v4 changes in the fix
ADVICE_ONLY = {"replan", "look_again", "correct_belief", "fix_target_hint", "fix_value_hint", "generic_fallback"}
REPEATS = {"wait_check", "wait_retry", "retype", "look_again"}
UNAWARE = "Your last step did not do what you intended, although your reasoning assumes it did."
THRASHING = "You have switched between several approaches without progress. Re-read the task and follow one approach step by step."
UNVERIFIED = "Before you continue, check on the screen that the value really changed."
IMPOSSIBLE = "If what the task asks for really does not exist, stop and report that the task cannot be done."


class StrategistV3(Strategist):

    def _wait_retry(self, d, v, goal, last, after) -> Plan:
        if last.get("type") not in ("click", "type", "key"):
            return self._replan(d, v, goal, last, after)
        if d.get("undo_by_reclick"):
            return Plan("look_again", feedback=TOGGLE)
        if self.cfg.retype and last.get("type") == "type":
            # typing is safe to repeat (the field is cleared first), and a screen check is fooled by changes the
            # typing did not make (a misplaced click that highlighted page text kept the text from being typed again)
            return Plan("retype", [{"action": dict(last), "wait": self.cfg.wait_s}], RETYPED)
        return Plan("wait_check", [{"action": dict(last), "wait": self.cfg.wait_s, "only_if_unchanged": True}],
                    REPEATED)

    def _fix_target_value(self, d, v, goal, last, after) -> Plan:
        plan = super()._fix_target_value(d, v, goal, last, after)
        if self.cfg.ground_fields and plan.kind == "fix_value" and plan.actions:
            text = str(last.get("text", ""))
            hit = self.locate(after, f"the text field that contains or should contain '{text}'")
            if not hit:
                val = plan.actions[0]["action"].get("text", "")
                return Plan("fix_value_hint", feedback=(
                    f"The value you entered is wrong; it should be {val!r}. Click into the field, replace its text "
                    "and check it before you continue."))
            plan.actions[0]["action"].update(x=hit[0], y=hit[1])
        return plan

    def _reground(self, d, v, goal, last, after) -> Plan:
        plan = super()._reground(d, v, goal, last, after)
        if (self.cfg.diag_rules and plan.kind == "reground" and plan.actions
                and "interaction" in str(d.get("subtype", "")).lower() and last.get("type") == "click"):
            plan.actions[-1]["action"]["double"] = True
            plan.feedback = plan.feedback.replace("has now done it on the target",
                                                  "has now double-clicked the target")
        return plan

    def escalate(self, plan: Plan, check: dict, last: dict) -> Plan:
        if plan.kind == "wait_check":
            return Plan("wait_check_failed", feedback=(
                "The supervisor waited and repeated your last action once, but the page still did not react: "
                f"{check.get('evidence', '')} Do not repeat it again; try a different way to do this step."))
        if plan.kind == "retype":
            return Plan("retype_failed", feedback=(
                "The supervisor typed your text again, but it still does not show in the field: "
                f"{check.get('evidence', '')} Click into the field yourself, type the text again, and check it before you "
                "continue."))
        return super().escalate(plan, check, last)

    axis_note = ("", "")        # text the wrapper adds before and after the fix's message
    progress = None             # (step, max_steps), set by the wrapper before each diagnosis

    gate_tried = False          # an access gate was already tried once in this episode

    def reset(self) -> None:
        super().reset()
        self.gate_tried = False

    def plan(self, d: dict, verdict: dict, goal: str, last_action: dict, after, hints_only: bool = False) -> Plan:
        plan = super().plan(d, verdict, goal, last_action, after, hints_only)
        if (self.cfg.gate_dismiss_first and plan.kind == "escalate" and d.get("type") == "III"
                and not self.gate_tried):
            self.gate_tried = True     # a sign-in window can often be closed; hand over only if it cannot
            target = (d.get("dismiss_target") or "the control that closes the window or continues without signing in "
                      "(x, Close, Not now, Continue as guest)")
            tried = self._dismiss_retry(dict(d, dismiss_target=target), verdict, goal, last_action, after)
            if tried.kind == "dismiss_retry":
                plan = tried
        self.axis_note = ("", "")
        if self.cfg.use_axes and not hints_only:
            plan = self._apply_axes(plan, d, verdict, goal, last_action, after)
        return plan

    def _apply_axes(self, plan: Plan, d: dict, v: dict, goal: str, last: dict, after) -> Plan:
        """Every axis of taxonomy v4 changes the fix; the type still decides the fix itself."""
        if d.get("type") in (None, "N") or plan.stop or plan.kind in ("none", "veto_done", "escalate", "report_irreversible"):
            return plan
        aware, beh = d.get("awareness"), d.get("behaviour")
        if aware == "aware" and beh == "progressing" and plan.kind in ADVICE_ONLY:
            return Plan("none", note="axes: the agent noticed the problem itself and is still progressing")
        if beh == "looping" and plan.kind in REPEATS:
            plan = self._replan(d, v, goal, last, after)          # repeating a looping step cannot help
        pre, post = [], []
        if beh == "looping":
            pre.append(f"You repeated {describe(last)} without progress. Do not do it again.")
        if aware == "unaware":
            pre.append(UNAWARE)
        if beh == "thrashing":
            post.append(THRASHING)
        feas = d.get("feasibility")
        if feas == "falsely_infeasible":
            post.append(f"The task can be done: {d.get('screen_truth') or d.get('evidence', '')} Do not give up.")
        elif feas == "infeasible":
            post.append(IMPOSSIBLE)
        if d.get("self_verification") == "not_verified" and (d.get("type") == "IV" or plan.kind in ("fix_value", "retype", "fix_value_hint")):
            post.append(UNVERIFIED)
        if self.progress and beh in ("looping", "thrashing"):
            step, max_steps = self.progress
            left = max_steps - step - 1
            if step >= 0.8 * max_steps and left > 0:
                post.append(f"Only {left} step{'s are' if left != 1 else ' is'} left. "
                            "Take the most direct way to finish the task.")
        self.axis_note = (" ".join(pre), " ".join(post))
        return plan
