"""
ADR v3 Monitor (v2: adr/monitor.py). Layer 1, the free checks, is the same (pixel change, repeated actions,
revisited screens); the judge, the done-check and the click check change:

1. Step judge: what the agent intended and what the task needs are judged apart, and the decision is against the
   task, so a typo the agent meant to type is still a failure. A step that did not do what the agent meant, but
   leaves the screen fine for the task (or closer to it), is not a failure: the agent sees the new screen and goes
   on. v2 flagged such steps, for example a click that had just set the right option but did not open the list the
   agent expected; the agent was then told the page had not responded and undid the setting.
2. The judge also gets both screenshots zoomed around the action and the size of the change near it, because small
   changes (a radio button, a check mark, text in a field, focus) were misread on the full screenshot.
3. Done-check: the start screen and the recent actions besides the screen now, and every criterion is met, not met
   or cannot tell; only a criterion the screen visibly contradicts vetoes "done". v2 vetoed finished runs whose result
   the final screen cannot show, such as which tab had been closed.
4. Click check: told plainly that the action has not happened yet, with a zoomed crop around the red circle. v2
   blocked a click on a close button "because the window has not closed yet".

Each behind a switch: date_context gives every prompt today's date and weekday; click_check=v4 is v2's click check
with conditions taken from the task only, never the result the action itself will produce, and v5 also quotes the
task's words next to what the screen shows; done_always lets the done-check read the judges' notes and report "cannot
be done" claims as infeasible, and done_pairs makes it quote the task's words too; no_effect_max_frac overrules a
"no effect" answer when much of the page changed.
"""
from __future__ import annotations

import datetime
import re
from typing import List, Optional

from PIL import Image

from ..agent import describe
from ..monitor import Monitor, same_action
from ..vision import action_point, box, crop_zoom, diff_info, mark_click, screen_hash

JUDGE_SYSTEM = """You check ONE step of a GUI agent (a web page, a desktop app or a phone app).
You get the task, the agent's recent actions, its reasoning and action for this step, a BEFORE screenshot (a red circle marks where the action landed, if it has a position), an AFTER screenshot (a blue box marks the region whose pixels changed; no box means no pixel changed) and, if the action has a position, both screenshots zoomed around it.
Work evidence first:
1. change: what visibly changed from BEFORE to AFTER. Look at the zoomed images: small changes count (a radio button or check box now selected, text now in a field, a field now focused, a menu, list or dialog opened or closed, a page or tab switched).
2. intended: what the agent meant this step to do, from its reasoning. intent_met: did exactly that happen? Answer from the change you described in 1, not from what would make sense: if the change is a different one (another element, menu, page or item than the one intended), intent_met = false, even if that change is useful.
3. task_progress: compared with BEFORE, is the screen now closer to finishing the TASK, about the same, or further from it (a wrong item or value, a wrong page, something needed undone, a pop-up now in the way)?
Then decide ok. Judge against the TASK, not only the agent's intention: a wrong item selected, a typed value that differs from what the task asks for (check it character by character, even if the agent meant to type it), or a pop-up now covering the page is NOT ok, even though something happened. A focus change or a partly filled form on the way to the goal IS ok.
- One exception: if the step did something other than what the agent intended, but the new screen is clearly closer to finishing the task (for example a setting the task needs is now selected), it is ok.
- ok = false only with one of these symptoms: no_effect (nothing relevant changed, although the action should have changed something), wrong_effect (the change works against the task), blocked (a pop-up, banner or dialog now covers the page or the target), unexpected_state (an error, a wrong page, or a state that does not fit the task).
Report only WHAT happened (the symptom), not why.
Reply with JSON only:
{"change": "<what visibly changed, or 'nothing'>",
 "intended": "<what the agent meant this step to do>",
 "intent_met": true or false,
 "task_progress": "closer" | "same" | "further",
 "ok": true or false,
 "symptom": "none" | "no_effect" | "wrong_effect" | "blocked" | "unexpected_state",
 "confidence": <0..1, how sure you are about ok>,
 "evidence": "<one sentence pointing at what in the screenshots supports the decision>"}"""

PRECOMMIT_READ_SYSTEM = """A GUI agent may be about to take an action that finalizes the task and cannot be undone. Check BEFORE it happens. A red circle marks where it is about to click, if it clicks.
0. First decide if THIS action will finish the task: the red marker is on a control that submits, sends, logs in, confirms or searches (or the action presses Enter to submit a form), OR it is the click the task itself asks for as its last step (the button, link, item or tab the task names). Selecting an option, ticking a box, opening a menu or tab on the way, or focusing a field is NOT final, even if the agent plans to submit afterwards. If it is not final, reply is_final=false, ready=true and stop.
1. If it is final, read the screen before you judge it: for every text field, write exactly the text that is inside the field, or "empty"; for every check box, radio button or list the task is about, write what is selected. Read only what is inside the fields themselves; the task text shown above the form is not part of any field.
2. Check that the red marker is on exactly the element the task requires (the right link, button or item, not a similar one next to it).
3. Turn the task into the concrete conditions that must hold on screen before finishing: the right value in each field, exactly the right items selected and nothing extra, the right option chosen, earlier required steps done. Check each condition against what you read in step 1, character by character for typed values.
Block only for a clear, visible problem. If everything the task needs is in place, it is ready.
Reply with JSON only:
{"is_final": true or false, "fields": [{"field": "...", "content": "..."}], "on_right_element": true or false,
 "conditions": [{"condition": "...", "met": true or false}],
 "ready": true or false, "problem": "<what is wrong, or empty>", "fix": "<concrete steps to fix it before finalizing, or empty>",
 "confidence": <0..1>}"""

BOX_TEXT = " (a blue box marks the region whose pixels changed; no box means no pixel changed)"
assert JUDGE_SYSTEM.count(BOX_TEXT) == 1
JUDGE_SYSTEM_NOBOX = JUDGE_SYSTEM.replace(BOX_TEXT, "")      # ablation judge_box=0: the judge is not told about a box

DONE_SYSTEM = """A GUI agent says its task is complete. Check that claim from the screen.
You get the task, the agent's recent actions and final reasoning, the screen NOW and, if given, the screen when the task STARTED.
First turn the task into 1-3 concrete success criteria. For each one decide:
- met: the screen now shows it;
- not_met: the screen visibly contradicts it (a wrong value or item, a form not submitted or not saved, a result that should be visible but is missing, an error);
- cannot_tell: it cannot be checked on this screen (for example a setting that is saved in the background, or something that happened earlier). Compare with the START screen to see what the agent changed, and use its recent actions.
The task is NOT complete only if some criterion is not_met. If every criterion is met or cannot_tell, it is complete.
Reply with JSON only:
{"criteria": [{"criterion": "...", "status": "met" | "not_met" | "cannot_tell"}],
 "complete": true or false, "confidence": <0..1>, "evidence": "<one sentence>"}"""

PRECOMMIT_SYSTEM = """A GUI agent is ABOUT to act. The action has NOT happened yet: you see the screen BEFORE it, with a red circle where it is about to click (also shown zoomed). Never judge whether the action worked, and never block because its effect is not visible yet.
0. First decide if THIS action will finish the task: the red marker is on a control that submits, sends, logs in, confirms, saves or searches (or the action presses Enter to submit a form), OR it is the click the task itself asks for as its last step. Selecting an option, ticking a box, opening a menu or tab on the way, focusing a field, or closing a window or dialog is NOT final. If it is not final, reply is_final=false, ready=true and stop.
1. If it is final, check that the red marker is on exactly the element the task requires (the right link, button or item, not a similar one next to it).
2. Turn the task into the concrete conditions that must already hold on screen: the right value in each field, exactly the right items selected and nothing extra, the right option or folder chosen, earlier required steps done. Check each against what is visible NOW.
Block only for a clear problem that is visible now, and name it. If everything the task needs is in place, it is ready.
Reply with JSON only:
{"is_final": true or false, "on_right_element": true or false,
 "conditions": [{"condition": "...", "met": true or false}],
 "ready": true or false, "problem": "<the visible problem, or empty>", "fix": "<concrete steps to fix it before finalizing, or empty>",
 "confidence": <0..1>}"""


PRECOMMIT_V4_SYSTEM = """A GUI agent may be about to take an action that finalizes the task and cannot be undone. Check BEFORE it happens. The action has NOT happened yet: you see the screen before it, with a red circle where it is about to click, if it clicks.
0. First decide if THIS action will finish the task: the red marker is on a control that submits, sends, logs in, confirms or saves the final result (or the action presses Enter to submit a whole form), OR it is the click the task itself asks for as its last step (the button, link, item or tab the task names). Opening a menu, list, dropdown or calendar, ticking a box, choosing an option, running a search, applying a filter or a sort, picking a folder or a name in a save dialog, focusing a field, and closing a window or dialog are steps on the way and NOT final, even if the agent plans to finish afterwards. A click on an item inside an open dropdown, list, menu or calendar only chooses that item: it is NOT final, even if it is the option the task needs. If it is not final, reply is_final=false, ready=true and stop.
1. If it is final, check that the red marker is on exactly the element the task requires (the right link, button or item, not a similar one next to it).
2. Turn the TASK text (never the agent's reasoning or claims) into the conditions that must ALREADY hold before this action: the right value in each field, exactly the right items selected and nothing extra, the right option, folder or date chosen, earlier required steps done. A value that must be picked from a suggestion list (an airport, a city, an address) is set only if a suggestion was clearly selected, not if it was only typed. For dates like "tomorrow" or "next Monday", work out the exact date from today's date. Never list this action itself or the result it will produce (the button being clicked, search results, a sorted list, a saved file, a booking): it cannot be visible yet.
3. Check each condition against what is visible NOW. If this click itself is what makes an unmet condition true (for example it picks the missing option), set fixes_problem = true: that is not a reason to block.
Block only for a clear problem that is visible now, and name it. If everything the task needs is in place, it is ready.
Reply with JSON only:
{"is_final": true or false, "on_right_element": true or false,
 "conditions": [{"condition": "...", "met": true or false}],
 "fixes_problem": true or false,
 "ready": true or false, "problem": "<the visible problem, or empty>", "fix": "<concrete steps to fix it before finalizing, or empty>",
 "confidence": <0..1>}"""

PRECOMMIT_V5_SYSTEM = """A GUI agent may be about to take an action that finalizes the task and cannot be undone. Check BEFORE it happens. The action has NOT happened yet: you see the screen before it, with a red circle where it is about to click, if it clicks.
0. First decide if THIS action will finish the task: the red marker is on a control that submits, sends, logs in, confirms or saves the final result (or the action presses Enter to submit a whole form), OR it is the click the task itself asks for as its last step (the button, link, item or tab the task names). Opening a menu, list, dropdown or calendar, ticking a box, choosing an option, running a search, applying a filter or a sort, picking a folder or a name in a save dialog, focusing a field, and closing a window or dialog are steps on the way and NOT final, even if the agent plans to finish afterwards. A click on an item inside an open dropdown, list, menu or calendar only chooses that item: it is NOT final, even if it is the option the task needs. If it is not final, reply is_final=false, ready=true and stop.
1. If it is final, check that the red marker is on exactly the element the task requires (the right link, button or item, not a similar one next to it).
2. Turn the TASK text (never the agent's reasoning or claims) into the conditions that must ALREADY hold before this action: the right value in each field, exactly the right items selected and nothing extra, the right option, folder or date chosen, earlier required steps done. For each condition, copy the words of the task it comes from (task_says), then write what the screen shows NOW for it (screen_shows). Never take a condition's value from the screen: if the task says "economy" and the form shows "Business", the condition is the economy class, and it is not met. A value that must be picked from a suggestion list (an airport, a city, an address) is set only if a suggestion was clearly selected, not if it was only typed. For dates like "tomorrow" or "next Monday", work out the exact date from today's date. Never list this action itself or the result it will produce (the button being clicked, search results, a sorted list, a saved file, a booking): it cannot be visible yet.
3. A condition is met only if screen_shows agrees with task_says. If this click itself is what makes an unmet condition true (for example it picks the missing option), set fixes_problem = true: that is not a reason to block.
Block only for a clear problem that is visible now, and name it. If everything the task needs is in place, it is ready.
Reply with JSON only:
{"is_final": true or false, "on_right_element": true or false,
 "conditions": [{"task_says": "...", "screen_shows": "...", "met": true or false}],
 "fixes_problem": true or false,
 "ready": true or false, "problem": "<the visible problem, or empty>", "fix": "<concrete steps to fix it before finalizing, or empty>",
 "confidence": <0..1>}"""

DONE_SYSTEM_V4 = """A GUI agent says its task is complete. Check that claim from the screen.
You get the task, today's date, the agent's recent actions with notes on what each step changed, its final reasoning, the screen NOW and, if given, the screen when the task STARTED.
First read the agent's final reasoning. If it says that the task cannot be done (for example the setting or feature does not exist, it is not possible, or the user has to be told), set agent_says_impossible = true. Then set impossible_plausible = false only if the screen shows a way to do it (for example the option it says is missing is visible); otherwise true.
Then turn the TASK text (never the agent's claims) into 1-3 concrete success criteria. For each one decide:
- met: the screen now shows it;
- not_met: the screen visibly contradicts it (a wrong value, item or date, a form not submitted or not saved, a result that should be visible but is missing, an error), or a note on an earlier step shows it went wrong and nothing later fixed it;
- cannot_tell: neither the screen nor the notes can show it (for example a setting saved in the background). Compare with the START screen to see what the agent changed.
For dates like "tomorrow", "next Monday" or "next weekend", work out the exact date from today's date and check the weekday.
The task is NOT complete if some criterion is not_met. If every criterion is met or cannot_tell, it is complete.
Reply with JSON only:
{"agent_says_impossible": true or false, "impossible_plausible": true or false,
 "criteria": [{"criterion": "...", "status": "met" | "not_met" | "cannot_tell"}],
 "complete": true or false, "confidence": <0..1>, "evidence": "<one sentence>"}"""

DONE_SYSTEM_V5 = """A GUI agent says its task is complete. Check that claim from the screen.
You get the task, today's date, the agent's recent actions with notes on what each step changed, its final reasoning, the screen NOW and, if given, the screen when the task STARTED.
First read the agent's final reasoning. If it says that the task cannot be done (for example the setting or feature does not exist, it is not possible, or the user has to be told), set agent_says_impossible = true. Then set impossible_plausible = false only if the screen shows a way to do it (for example the option it says is missing is visible); otherwise true.
Then turn the TASK text (never the agent's claims) into 1-3 concrete success criteria. For each one, copy the words of the task it comes from (task_says) and write what the screen or the notes show for it (shown). If a note on an earlier step names the value that was chosen (a folder, a date, an option, an amount, a setting) and no later note changed it, compare that value with the task's words: a different value is not_met, not cannot_tell. For each one decide:
- met: the screen now shows it;
- not_met: the screen visibly contradicts it (a wrong value, item or date, a form not submitted or not saved, a result that should be visible but is missing, an error), or a note on an earlier step shows it went wrong and nothing later fixed it;
- cannot_tell: neither the screen nor the notes can show it (for example a setting saved in the background). Compare with the START screen to see what the agent changed.
For dates like "tomorrow", "next Monday" or "next weekend", work out the exact date from today's date and check the weekday.
The task is NOT complete if some criterion is not_met. If every criterion is met or cannot_tell, it is complete.
Reply with JSON only:
{"agent_says_impossible": true or false, "impossible_plausible": true or false,
 "criteria": [{"criterion": "...", "task_says": "...", "shown": "...", "status": "met" | "not_met" | "cannot_tell"}],
 "complete": true or false, "confidence": <0..1>, "evidence": "<one sentence>"}"""


OPEN_LIST = re.compile(r"\b(drop-?down|list|menu|options|calendar|date picker|suggestions?)\b[^.]{0,45}?"
                       r"\b(open|opened|expanded|showing|shows)\b", re.I)


def choosing_in_open_list(history: Optional[List[dict]], action: dict) -> bool:
    """The agent clicks inside a dropdown, list, menu or calendar that its previous step just opened, so the click
    only chooses an item and is never final (the click check once blocked the very click that set the folder it asked
    for)."""
    p = action_point(action)
    if not history or not p or action.get("type") != "click":
        return False
    v = history[-1].get("verdict") or {}
    change = str((v.get("judge") or {}).get("change") or "")
    bbox = (v.get("signals") or {}).get("bbox")
    if not bbox or not OPEN_LIST.search(change):
        return False
    x0, y0, x1, y1 = bbox
    return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


def today_line() -> str:
    """Today's date and weekday, for tasks that say "tomorrow", "next Monday" or "next weekend"."""
    d = datetime.date.today()
    return f"Today is {d:%A}, {d.day} {d:%B %Y}."


def step_notes(history: Optional[List[dict]], n: int = 8) -> List[str]:
    """The agent's recent actions with what the step judge saw after each one, for the done-check."""
    out = []
    for s in (history or [])[-n:]:
        v = s.get("verdict") or {}
        j = v.get("judge") or {}
        seen = str(j.get("change") or "").strip()
        flag = f" [flagged: {v.get('symptom')}]" if v.get("flagged") else ""
        by = " (by the supervisor)" if s.get("by") == "supervisor" else ""
        out.append(f"- {describe(s['action'])}{by}" + (f"; the judge saw: {seen}" if seen else "") + flag)
    return out or ["- (none)"]


def zoom_half(img: Image.Image) -> int:
    """Half the side of the area zoomed around the action: larger on large screens."""
    return max(120, min(img.size) // 8)


def _crop_box(img: Image.Image, x: float, y: float, half: int):
    w, h = img.size
    return int(max(0, x - half)), int(max(0, y - half)), int(min(w, x + half)), int(min(h, y + half))


class MonitorV3(Monitor):

    def check(self, goal: str, before: Image.Image, after: Image.Image, thought: str,
              action: dict, history: List[dict]) -> dict:
        cfg = self.cfg
        sig = self._signals(before, after, action)
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
        loop = acting and ((repeats >= 2 and self.no_effect_run >= 2)
                           or (repeats >= cfg.loop_repeats and seen >= 1))
        signals = {**sig, "no_effect": no_effect, "repeats": repeats, "screen_seen_before": seen, "loop": loop}
        judge = None
        if cfg.judge_policy == "every_step" or no_effect or loop:
            judge = self._settle(goal, before, after, thought, action, history, sig)
        return self._verdict(judge, loop, repeats, signals)

    def recheck(self, goal: str, before: Image.Image, after: Image.Image, thought: str,
                action: dict, history: List[dict]) -> dict:
        """A second look at the same step on a fresh screenshot (the wrapper waits first). No state changes."""
        sig = self._signals(before, after, action)
        judge = self._settle(goal, before, after, thought, action, history, sig)
        return self._verdict(judge, False, 1, {**sig, "second_look": True})

    def check_done(self, goal: str, shot: Image.Image, thought: str, first: Optional[Image.Image] = None,
                   history: Optional[List[dict]] = None) -> dict:
        images = [shot] if first is None else [first, shot]
        labels = ("Image 1: the screen NOW" if first is None else
                  "Image 1: the screen when the task STARTED\nImage 2: the screen NOW")
        if self.cfg.done_always:            # earlier notes, the date, and "cannot be done" claims
            notes = "\n".join(step_notes(history))
            out = self.llm.ask(DONE_SYSTEM_V5 if self.cfg.done_pairs else DONE_SYSTEM_V4, f"Task: {goal}\n{today_line()}\n"
                                               f"The agent's recent actions (oldest first), with what the step judge saw:\n{notes}\n"
                                               f"Agent's final reasoning: {thought}\n{labels}", images, tag="judge")
        else:
            recent = "\n".join(f"- {describe(s['action'])}" for s in (history or [])[-6:]) or "- (none)"
            date = f"{today_line()}\n" if self.cfg.date_context else ""
            out = self.llm.ask(DONE_SYSTEM, f"Task: {goal}\n{date}The agent's recent actions:\n{recent}\n"
                                            f"Agent's final reasoning: {thought}\n{labels}", images, tag="judge")
        crit = [c for c in out.get("criteria", []) if isinstance(c, dict)]
        not_met = [c.get("criterion", "") for c in crit
                   if str(c.get("status", "")).strip().lower().replace(" ", "_") == "not_met"]
        conf = float(out.get("confidence", 0) or 0)
        impossible = bool(self.cfg.done_always and out.get("agent_says_impossible") is True
                          and out.get("impossible_plausible") is not False)
        flagged = (not impossible and bool(not_met) and out.get("complete") is not True
                   and conf >= self.cfg.flag_confidence)
        return {"flagged": flagged, "symptom": "false_done" if flagged else "none", "source": "done_check",
                "confidence": conf, "evidence": out.get("evidence", ""), "unmet": not_met, "judge": out,
                "impossible": impossible, "signals": {}}

    def check_commit(self, goal: str, shot: Image.Image, thought: str, action: dict, history: List[dict]) -> dict:
        if self.cfg.click_check == "v2":    # v2's click check: its prompt and one marked screenshot
            return super().check_commit(goal, shot, thought, action, history)
        if self.cfg.click_check in ("v4", "v5"):  # v2's check with conditions from the task only, never the action's
            p = action_point(action)               # result; v5: each condition quotes the task next to the screen
            recent = "\n".join(f"- {describe(s['action'])}" for s in history[-5:]) or "- (none)"
            out = self.llm.ask(PRECOMMIT_V5_SYSTEM if self.cfg.click_check == "v5" else PRECOMMIT_V4_SYSTEM, f"Task: {goal}\n{today_line()}\nActions so far:\n{recent}\n"
                               f"About to do (NOT done yet). Reasoning: {thought}\nAction: {describe(action)}",
                               [mark_click(shot, *p) if p else shot], tag="judge")
            conf = float(out.get("confidence", 0) or 0)
            unmet = (self.cfg.click_check == "v5"                     # an unmet condition counts even if
                     and any(isinstance(c, dict) and c.get("met") is False  # the reply says "ready"
                             for c in out.get("conditions") or []))
            blocked = (out.get("is_final") is not False and (out.get("ready") is False or unmet)
                       and out.get("fixes_problem") is not True and conf >= self.cfg.flag_confidence)
            return {"blocked": blocked, "is_final": out.get("is_final"), "problem": out.get("problem", ""),
                    "fix": out.get("fix", ""), "confidence": conf, "judge": out}
        if self.cfg.click_check == "read":  # v2's check, but it reads every field before it judges
            p = action_point(action)
            recent = "\n".join(f"- {describe(s['action'])}" for s in history[-5:]) or "- (none)"
            out = self.llm.ask(PRECOMMIT_READ_SYSTEM, f"Task: {goal}\nActions so far:\n{recent}\n"
                               f"About to finalize. Reasoning: {thought}\nAction: {describe(action)}",
                               [mark_click(shot, *p) if p else shot], tag="judge")
            conf = float(out.get("confidence", 0) or 0)
            blocked = (out.get("is_final") is not False and out.get("ready") is False
                       and conf >= self.cfg.flag_confidence)
            return {"blocked": blocked, "is_final": out.get("is_final"), "problem": out.get("problem", ""),
                    "fix": out.get("fix", ""), "confidence": conf, "judge": out}
        p = action_point(action)
        img = mark_click(shot, *p) if p else shot
        images = [img] + ([crop_zoom(img, *p, half=zoom_half(shot))] if p else [])
        recent = "\n".join(f"- {describe(s['action'])}" for s in history[-5:]) or "- (none)"
        text = (f"Task: {goal}\nActions so far:\n{recent}\n"
                f"About to do (NOT done yet). Reasoning: {thought}\nAction: {describe(action)}"
                + ("\nImage 1: the screen now, red circle = where it is about to click"
                   "\nImage 2: the same, zoomed around the red circle" if p else ""))
        out = self.llm.ask(PRECOMMIT_SYSTEM, text, images, tag="judge")
        conf = float(out.get("confidence", 0) or 0)
        blocked = (out.get("is_final") is not False and out.get("ready") is False
                   and conf >= self.cfg.flag_confidence)
        return {"blocked": blocked, "is_final": out.get("is_final"), "problem": out.get("problem", ""),
                "fix": out.get("fix", ""), "confidence": conf, "judge": out}

    def _signals(self, before: Image.Image, after: Image.Image, action: dict) -> dict:
        cfg = self.cfg
        sig = diff_info(before, after, cfg.scale)
        w, h = before.size
        sig["changed_frac"] = sig["changed_px"] / max(1, (w // cfg.scale) * (h // cfg.scale))
        p = action_point(action)
        sig["local_changed_px"] = None
        if p and before.size == after.size:
            c = _crop_box(before, *p, zoom_half(before))
            sig["local_changed_px"] = diff_info(before.crop(c), after.crop(c), cfg.scale)["changed_px"]
        return sig

    def _settle(self, goal, before, after, thought, action, history, sig) -> Optional[dict]:
        # a "no effect" answer is not asked again when the pixels contradict it: asking again turned correct flags
        # into hallucinated passes. A late page update is handled by the wrapper's second look instead.
        judge = self._closer_is_ok(self._judge3(goal, before, after, thought, action, history, sig))
        return self._pixels_overrule(judge, sig)

    def _pixels_overrule(self, judge: Optional[dict], sig: dict) -> Optional[dict]:
        """A "no effect" answer is wrong when much of the page changed, around the action too (the judge sometimes
        said "nothing" while most of the screen had changed)."""
        cfg = self.cfg
        local = sig.get("local_changed_px")
        if (cfg.no_effect_max_frac > 0 and judge and judge.get("ok") is False
                and str(judge.get("symptom", "")).lower() == "no_effect"
                and sig.get("changed_frac", 0) >= cfg.no_effect_max_frac
                and (local is None or local >= 10 * cfg.no_change_px)):
            return dict(judge, ok=True, symptom="none", overruled="pixels_changed")
        return judge

    @staticmethod
    def _closer_is_ok(judge: Optional[dict]) -> Optional[dict]:
        """A step that brought the task closer is never a failure, whatever the agent intended."""
        if judge and judge.get("ok") is False and str(judge.get("task_progress", "")).lower() == "closer":
            return dict(judge, ok=True, symptom="none", overruled="task_progress_closer")
        return judge

    def _judge3(self, goal, before, after, thought, action, history, sig, note: str = "") -> Optional[dict]:
        p = action_point(action)
        b_img = mark_click(before, *p) if p else before
        boxed = self.cfg.judge_box != 0
        images = [b_img, box(after, sig["bbox"]) if boxed else after]
        labels = ["Image 1: BEFORE (red circle = where the action landed)",
                  "Image 2: AFTER (blue box = the changed pixels)" if boxed else "Image 2: AFTER"]
        if p:
            half = zoom_half(before)
            images += [crop_zoom(b_img, *p, half=half), crop_zoom(after, *p, half=half)]
            labels += ["Image 3: BEFORE, zoomed around the action", "Image 4: AFTER, zoomed around the action"]
        recent = "\n".join(f"- {describe(s['action'])}" for s in history[-4:-1]) or "- (none)"
        text = "\n".join([f"Task: {goal}", *([today_line()] if self.cfg.date_context else []), "Recent earlier actions:", recent,
                          f"This step. Reasoning: {thought}", f"Action: {describe(action)}", *labels]
                         + ([note] if note else []))
        return self.llm.ask(JUDGE_SYSTEM if boxed else JUDGE_SYSTEM_NOBOX, text, images, tag="judge")

    def _verdict(self, judge, loop, repeats, signals) -> dict:
        cfg = self.cfg
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
