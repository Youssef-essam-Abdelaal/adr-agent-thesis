"""
ADR v3 Diagnostician (v2: adr/diagnostician.py). Same evidence pack and taxonomy, two added rules:

1. Type II (Unresponsive Environment) only if the screen did not change at all after the action. v2 often chose II
   for steps whose effect was small or late, and its fix (wait and repeat) then repeated clicks on toggles and links.
2. N (no failure) when the flagged step did not really go wrong: it moved the task forward, or left the screen fine
   for the task, and the Monitor misread it. v2 rarely chose N.

Votes are off in v3 (adr_version=3 sets votes=1); they did not change the success rate.

diag_rules adds more rules from the OSWorld failures (a step that did what the agent intended is not I or II; single
clicks that only select; banners and access gates; form errors after an on-target click; dates), and date_context
puts today's date in the message.
"""
from __future__ import annotations

from typing import Optional

from ..config import Config
from ..diagnostician import SYSTEM, Diagnostician
from ..taxonomy import Taxonomy

RULES = """Two rules that override the procedure above:
- Choose II only if the screen did not change at all after the action. Compare the zoomed crops: if anything relevant changed (a selection, a check mark, text, focus, a menu, a page), it is not II.
- If the flagged step did not really go wrong (it moved the task forward, or the new screen is fine for the task and the monitor misread it), answer N."""

SYSTEM_V3 = SYSTEM.replace("{taxonomy}\n\nReply with JSON only:", "{taxonomy}\n\n" + RULES + "\n\nReply with JSON only:")
assert SYSTEM_V3 != SYSTEM, "the v2 prompt changed: update the anchor for the v3 rules"

RULES_V4 = """More rules (v3.4):
- If the screen changed the way the agent intended (a menu, list or calendar opened, the page scrolled, a dialog opened), it is not I or II; if nothing went wrong, answer N.
- In a file dialog or a list of files, a single click only selects an item; opening it needs a double click. A single click meant to open something is I, sub-type interaction mechanics.
- A banner, cookie notice or overlay over the target is III, even if the click itself was aimed well. A page that says access is denied, blocked or not available in your region is III (access gate).
- If the action hit its intended control but the page now shows a form error or warning (for example "Please select destination"), it is IV; name the missing or unselected option as correct_target.
- For dates like "tomorrow", "next Monday" or "next weekend", use today's date from the message and check the weekday."""

SYSTEM_V34 = SYSTEM_V3.replace(RULES, RULES + "\n" + RULES_V4)
assert SYSTEM_V34 != SYSTEM_V3


# use_axes: the labeling-only axes of taxonomy v4 that a single step can show; the third one, termination,
# describes how a run ends and is handled by the step budget in the Strategist
EXTRA_AXES = {
    "feasibility": {"values": ["feasible", "falsely_infeasible", "infeasible", "unclear"],
                    "rule": "feasible unless the agent's reasoning says the task or the step cannot be done (an option or "
                            "feature does not exist, it is impossible). If it says so: falsely_infeasible when the screen or "
                            "common knowledge shows a way to do it, infeasible when the requested thing really does not "
                            "exist, unclear otherwise."},
    "self_verification": {"values": ["verified", "not_verified"],
                          "rule": "verified if the agent's reasoning describes the result of its previous action before "
                                  "acting on it; not_verified if it acted on a result it only assumed."},
}
AXES_FIELDS = (' "recoverability": "recoverable" | "irreversible",\n',
               ' "recoverability": "recoverable" | "irreversible",\n'
               ' "feasibility": "feasible" | "falsely_infeasible" | "infeasible" | "unclear",\n'
               ' "self_verification": "verified" | "not_verified",\n')


class DiagnosticianV3(Diagnostician):
    def __init__(self, cfg: Config, taxonomy: Optional[Taxonomy] = None):
        super().__init__(cfg, taxonomy)
        system = SYSTEM_V34 if cfg.diag_rules else SYSTEM_V3
        if cfg.use_axes:                     # report the two labeling-only axes too, so they can change the fix
            self.tax.data["axes"] = {**self.tax.data["axes"], **EXTRA_AXES}
            assert AXES_FIELDS[0] in system, "the Diagnostician's JSON changed: update AXES_FIELDS"
            system = system.replace(AXES_FIELDS[0], AXES_FIELDS[1], 1)
        self.system = system.format(taxonomy=self.tax.render())

    def evidence(self, goal, history, verdict, before, after, thought, action, frames=()):
        text, images = super().evidence(goal, history, verdict, before, after, thought, action, frames)
        if self.cfg.date_context:            # dates like "next Monday" need today's date
            from .monitor import today_line
            text = text.replace(f"Task: {goal}", f"Task: {goal}\n{today_line()}", 1)
        return text, images
