"""
Every knob in one place. Load a JSON file, then override single keys from the
command line with --set key=value. Nothing else in the package hard-codes a
model name, threshold or budget.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # which system version to run
    adr_version: int = 2           # 1 = ADR v1, 2 = ADR v2, 3 = ADR v3 (adr/v3/)
    # models (any OpenAI vision model id)
    agent_model: str = "gpt-5.6-luna"     # the base agent being wrapped
    agent_effort: str = "low"
    judge_model: str = "gpt-5.6-luna"     # the judge: Layer 2 of Verify, the click check and the done-check
    judge_effort: str = "low"
    diag_model: str = "gpt-5.6-terra"     # Diagnostician
    diag_effort: str = "medium"
    ground_model: str = "gpt-5.6-luna"    # the element finder, used by the Strategist
    ground_effort: str = "low"
    # where each model runs (see adr/llm.py): "" = OpenAI, "http://host:port/v1" = an OpenAI-compatible
    # server such as vLLM, "ollama:http://host:port" = Ollama. Example: configs/open_models.json
    agent_base_url: str = ""
    judge_base_url: str = ""
    diag_base_url: str = ""
    ground_base_url: str = ""
    agent_kind: str = "json"       # json = the JSON agent in adr/agent.py; uitars = UI-TARS (adr/agent_uitars.py)
    local_ctx: int = 8192          # context length asked of Ollama
    local_temperature: float = 0.0 # sampling temperature for open models
    ground_coords: str = "pixels"  # pixels, or norm1000 for models that point on a 0-1000 scale (Qwen3-VL)
    local_max_side: int = 0        # if > 0, screenshots sent to Ollama are shrunk to this longest side

    # environment kind: miniwob (default) or osworld (adr/env/osworld.py)
    env_kind: str = "miniwob"
    osworld_dir: str = "third_party/OSWorld"
    osworld_vm: str = ""           # path to the .vmx; "" = let OSWorld find or install it
    osworld_domain: str = "chrome"
    osworld_pause: float = 2.0     # seconds to wait after an action before the screenshot

    # environment
    miniwob_dir: str = "third_party/miniwob-plusplus/miniwob/html/miniwob"
    scale: int = 3                 # screenshot pixels per CSS pixel
    settle_ms: int = 450           # wait after an action before the next screenshot
    max_steps: int = 10            # the step budget: agent decisions per episode
    max_actions: int = 16          # agent + recovery actions per episode

    # Monitor (Verify)
    judge_policy: str = "every_step"   # every_step | suspicious_only
    no_change_px: int = 10         # changed pixels (CSS scale) below this = no visible effect
    loop_repeats: int = 3          # same action this many times in a row, ending on a screen seen before = loop
    flag_confidence: float = 0.55  # judge must be at least this sure to flag

    # Diagnostician
    taxonomy: str = "taxonomy/taxonomy_v4.json"
    history_steps: int = 6         # earlier steps given as text
    vote_below: float = 0.7        # if the diagnosis is less sure than this, ask again and take a vote
    votes: int = 3                 # total diagnoses in a vote
    context_frames: int = 2        # earlier screenshots given as images (0 = before/after only)

    # Strategist (Recover)
    max_recoveries: int = 3            # the fix budget: flagged steps handled per episode
    wait_s: float = 3.0                # wait before an action is repeated (type II)

    # Prevent, the done-check and Check the fix
    precommit: int = 1                 # the click check: check the screen before an action that finalizes the task
    precommit_mode: str = "clicks"     # clicks = check before every click and Enter; keywords = only when the reasoning names a final action
    precommit_scope: str = "adr"       # adr = ADR conditions only; all = also the retry baselines
    precommit_recovery: int = 0        # 1 = also check the wrapper's own final clicks (ADR v2.1)
    precommit_max_blocks: int = 2      # block at most this many clicks per episode
    precommit_words: str = ("submit,log in,login,sign in,send,confirm,save,search,ok,done,finish,"
                            "forward,reply,book,buy,apply,next,continue")
    verify_recovery: int = 1           # check the fix: re-check the screen after a fix, escalate once if it failed
    done_check: int = 1                # the done-check when the agent says "done" (0 = off, an ablation)
    second_look_s: float = 2.0         # v3: wait before the second look at a step judged "no effect" (0 = off)
    judge_box: int = 1                 # v3: draw Layer 1's box on the AFTER screenshot and explain it to the judge
                                       # (0 = ablation: no box and no sentence about it)
    retype: int = 0                    # v3: after a typing step judged "no effect", type it again (no screen check)
    click_check: str = "v3"            # v3: which click check (Prevent): v3 (adr/v3/monitor.py), v2 (adr/monitor.py),
                                       # read: v2's check that first reads every field,
                                       # v4: v2's check with conditions from the task only, never the action's result,
                                       # v5: v4, and each condition quotes the task next to what the screen shows
    # more ADR v3 switches, all off by default; the final settings, ADR v3.7, turn them on (V37 in adr/v3/replay.py)
    date_context: int = 0              # today's date and weekday in the judge, click check, done-check and Diagnostician
    done_always: int = 0               # the done-check also runs once the fix budget is used (at most max_done_vetoes
                                       # vetoes in all), reads the judges' notes on earlier steps, and a "done" whose
                                       # reasoning says the task cannot be done is reported as infeasible (OSWorld's FAIL)
    max_done_vetoes: int = 2
    no_effect_max_frac: float = 0.0    # > 0: overrule a judge "no effect" when this share of the page changed, including
                                       # the area around the action (0 = off)
    n_free: int = 0                    # a diagnosis N (no failure) does not use a fix
    drop_blocked: int = 0              # a click the click check blocked is removed from the agent's own history
    diag_rules: int = 0                # extra Diagnostician rules: intended changes, selection-only clicks, banners and
                                       # access gates, form errors, dates
    fix_report: int = 0                # the message after a checked fix says what the screen shows afterwards
    ground_fields: int = 0             # a value correction first finds the field with the element finder, instead of
                                       # reusing the point of the type action
    replan_gap: int = 0                # > 0: at most one "try another way" (replan) per this many agent steps; a replan
                                       # sooner than that is dropped and uses no fix (0 = off)
    done_pairs: int = 0                # done-check: each criterion quotes the task's words and what the screen or the
                                       # earlier notes show; a value in the notes that differs from the task is not met
    use_axes: int = 0                  # the Diagnostician also reports feasibility and self-verification, and every axis
                                       # changes the fix (adr/v3/strategist.py, _apply_axes); 0 = off
    replan_strikes: int = 0            # 2: "try another way" only if the step before was flagged too or the loop rule fired;
                                       # otherwise the replan is held back and uses no fix (0 = off)
    gate_dismiss_first: int = 0        # an access gate (III) is first closed once, if a close control is found, before
                                       # the task is handed to a person

    # fault injection
    fault_randomize: int = 0       # 1 = draw every fault setting per episode (confirmatory run)
    stagnation_ms: int = 3000      # the ignored-input fault: how long the page ignores input
    grounding_offset: int = 22     # CSS pixels a mis-click is displaced by

    # experiment
    workers: int = 10              # episodes run in parallel, one browser each
    image_detail: str = "high"     # detail level of the images sent to the OpenAI API

    @classmethod
    def load(cls, path: Optional[str] = None, overrides: Optional[List[str]] = None) -> "Config":
        cfg = cls()
        if path:
            data = json.loads(Path(path).read_text())
            for k, v in data.items():
                if k.startswith("_"):
                    continue
                cfg.set(k, v)
        for item in overrides or []:
            k, _, v = item.partition("=")
            cfg.set(k.strip(), v.strip())
        return cfg.apply_version()

    def apply_version(self) -> "Config":
        """ADR v1 = v2 without the click check, the check after a fix and the vote, and with taxonomy v4.0.
        v3 = adr/v3/, votes off."""
        if self.adr_version == 1:
            self.precommit, self.verify_recovery, self.votes = 0, 0, 1
            if self.taxonomy == "taxonomy/taxonomy_v4.json":
                self.taxonomy = "taxonomy/taxonomy_v4.0.json"
        if self.adr_version >= 3:      # v3: votes off, they did not change the success rate
            self.votes = 1
        if self.click_check not in ("v2", "v3", "read", "v4", "v5"):
            raise ValueError(f"click_check must be v2, v3, read, v4 or v5, not {self.click_check!r}")
        return self

    def set(self, key: str, value) -> None:
        types = {f.name: f.type for f in fields(self)}
        if key not in types:
            raise KeyError(f"unknown config key: {key}")
        t = types[key]
        if isinstance(value, str) and t in ("int", "float"):
            value = int(value) if t == "int" else float(value)
        setattr(self, key, value)

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p

    def to_dict(self) -> dict:
        return asdict(self)
