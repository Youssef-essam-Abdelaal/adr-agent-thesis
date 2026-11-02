"""
Diagnose: given a flagged step, find the cause with the taxonomy.

It is only called on a flag, so it can afford a stronger model and more
evidence: the marked before/after pair, zoomed crops around the action
(crop-and-zoom as in Learning-from-Failure), earlier frames and the step
history (trajectory context), and the Monitor's evidence. The taxonomy and its
ordered decision procedure come from the taxonomy file, so the label set can
change without touching code.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional

from PIL import Image

from .agent import describe
from .config import Config
from .llm import LLM
from .taxonomy import TYPE_IDS, Taxonomy
from .vision import action_point, box, crop_zoom, mark_click, thumb

SYSTEM = """You are the Diagnostician in a recovery system for GUI agents. A monitor flagged a step as failed.
Find the CAUSE using the taxonomy, then give what the recovery step needs.
Read the evidence before deciding: the red circle in the BEFORE image marks where the action landed, the blue box in the AFTER image marks what changed, the zoomed crops show the area around the action, and earlier frames show how the run got here. Compare where the action landed with the element the agent's reasoning names.

{taxonomy}

Reply with JSON only:
{{"evidence": "<2-3 sentences: what the agent intended, where the action actually landed, what the screen shows now>",
 "type": "I" | "II" | "III" | "IV" | "V" | "VI" | "N",
 "subtype": "<one subtype of that type>",
 "awareness": "aware" | "unaware",
 "behaviour": "progressing" | "looping" | "thrashing",
 "recoverability": "recoverable" | "irreversible",
 "intended_target": "<the element the agent meant to act on, described so it can be found on screen (its visible text and kind), or null>",
 "correct_value": "<if a wrong value was entered: the exact value the task requires, else null>",
 "dismiss_target": "<if something obstructs the page: the control that closes it, else null>",
 "correct_target": "<if the action used the wrong element: the element the TASK actually needs (visible text and kind), else null>",
 "undo_by_reclick": <true if clicking the same spot again would undo what the action did (a checkbox or toggle), else false>,
 "screen_truth": "<one sentence: what the screen actually shows that matters for the next step>",
 "confidence": <0..1>}}"""


class Diagnostician:
    def __init__(self, cfg: Config, taxonomy: Optional[Taxonomy] = None):
        self.cfg = cfg
        self.tax = taxonomy or Taxonomy(cfg.path(cfg.taxonomy))
        self.llm = LLM.for_role(cfg, "diag")
        self.system = SYSTEM.format(taxonomy=self.tax.render())
        if cfg.adr_version == 1:   # v1 did not ask for these two fields
            self.system = "\n".join(l for l in self.system.split("\n")
                                     if not l.startswith((' "correct_target"', ' "undo_by_reclick"')))

    def diagnose(self, goal: str, history: List[dict], verdict: dict, before: Image.Image,
                 after: Image.Image, thought: str, action: dict, frames: List[Image.Image] = ()) -> dict:
        cfg = self.cfg
        text, images = self.evidence(goal, history, verdict, before, after, thought, action, frames)
        out = self._one(text, images)
        # an unsure diagnosis is asked again and settled by a vote;
        # with no majority the Strategist falls back to the generic message
        if float(out.get("confidence", 0) or 0) < cfg.vote_below and cfg.votes > 1:
            others = [self._one(text, images) for _ in range(cfg.votes - 1)]
            votes = [out["type"]] + [o["type"] for o in others]
            top, n = Counter(votes).most_common(1)[0]
            out["votes"] = votes
            if n * 2 > len(votes):
                if out["type"] != top:
                    out = dict(next(o for o in others if o["type"] == top), votes=votes)
            else:
                out["uncertain"] = True
        return out

    def evidence(self, goal: str, history: List[dict], verdict: dict, before: Image.Image,
                 after: Image.Image, thought: str, action: dict, frames: List[Image.Image] = ()):
        """The evidence pack: the same for the typed and the untyped Diagnostician."""
        cfg = self.cfg
        p = action_point(action)
        bbox = (verdict.get("signals") or {}).get("bbox")
        images = [mark_click(before, *p) if p else before, box(after, bbox)]
        labels = ["Image 1: BEFORE (red circle = where the action landed)",
                  "Image 2: AFTER (blue box = what changed)"]
        if p:
            images += [crop_zoom(before, *p), crop_zoom(after, *p)]
            labels += ["Image 3: BEFORE zoomed around the action", "Image 4: AFTER zoomed around the action"]
        ctx = list(frames)[-cfg.context_frames:] if cfg.context_frames > 0 else []
        for i, f in enumerate(ctx):
            images.append(thumb(f))
            labels.append(f"Image {len(images)}: earlier frame {i + 1} of {len(ctx)} (oldest first)")

        steps = []
        for i, s in enumerate(history[-cfg.history_steps:]):
            flag = s.get("verdict") or {}
            note = f" [monitor: {flag.get('symptom')}]" if flag.get("flagged") else ""
            steps.append(f"- {describe(s['action'])}: {s.get('thought', '')[:160]}{note}")
        j = verdict.get("judge") or {}
        text = "\n".join([
            f"Task: {goal}",
            "Steps so far (oldest first):", *(steps or ["- (none)"]),
            "",
            "FLAGGED STEP",
            f"Agent reasoning: {thought}",
            f"Action as executed: {describe(action)}",
            f"Monitor symptom: {verdict.get('symptom')} (confidence {verdict.get('confidence', 0):.2f})",
            f"Monitor evidence: {verdict.get('evidence', '')}",
            f"Monitor saw change: {j.get('change', 'n/a')}; expected: {j.get('expected', 'n/a')}",
            f"Unmet completion criteria: {verdict.get('unmet')}" if verdict.get("unmet") else "",
            "",
            *labels,
        ])
        return text, images

    def _one(self, text, images) -> dict:
        out = self.llm.ask(self.system, text, images, tag="diag")
        t = str(out.get("type", "")).strip().upper()
        out["type"] = t if t in TYPE_IDS else "V"
        return out


UNTYPED_SYSTEM = """You help a GUI agent recover from a failed step. A monitor flagged the step below.
Read the evidence before deciding: the red circle in the BEFORE image marks where the action landed, the blue box in the AFTER image marks what changed, the zoomed crops show the area around the action, and earlier frames show how the run got here. Compare where the action landed with the element the agent's reasoning names.
Explain in your own words what went wrong, then give the agent short, concrete advice for its next step. If nothing went wrong, say so.
Reply with JSON only:
{"what_went_wrong": "<2-3 sentences, or 'nothing'>",
 "advice": "<one or two sentences telling the agent exactly what to do next>",
 "no_failure": true or false,
 "confidence": <0..1>}"""


class UntypedDiagnostician(Diagnostician):
    """
    The adr_untyped ablation: the same model and the same evidence as the
    Diagnostician, but no taxonomy, no types and no fix table. It explains the
    failure in its own words and advises the agent. Comparing it with typed
    hints (adr_hints) isolates what the taxonomy itself adds.
    """

    def __init__(self, cfg: Config, taxonomy: Optional[Taxonomy] = None):
        super().__init__(cfg, taxonomy)
        self.system = UNTYPED_SYSTEM

    def advise(self, goal: str, history: List[dict], verdict: dict, before: Image.Image,
               after: Image.Image, thought: str, action: dict, frames: List[Image.Image] = ()) -> dict:
        text, images = self.evidence(goal, history, verdict, before, after, thought, action, frames)
        return self.llm.ask(self.system, text, images, tag="diag")
