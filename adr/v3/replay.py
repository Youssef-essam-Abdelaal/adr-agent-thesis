"""
ADR v3.7 on a FINISHED run: the offline replay mode, to compare ADR with CUADebug on CUAErrorBench, where the
debugger is not given the root step.

ADR is never given a step: live, its Monitor checks each step as it happens and flags the one that goes wrong. Here
the recorded actions of a finished run go, in their order, through the same parts that run live, with the v3.7
settings in V37 (adr/v3, unchanged):
  - each action is judged by MonitorV3.check from the screens before and after it and the agent's reasoning;
  - on a flag, DiagnosticianV3 confirms it with a type or cancels it with N, as live;
  - the first confirmed failure is ADR's answer: its step, type and CUADebug category;
  - if no step was confirmed and the run ends with the agent's DONE, MonitorV3.check_done judges that claim, and a
    flagged claim goes to the Diagnostician at that last step, as live.
Off, because a recording cannot change: the click check before an action (Prevent), the fixes (Recover) and the
second look (a fresh screenshot after a wait). Not judged: the first action, since the recordings have no screen from
before it. ADR sees each run only up to the current action, as live; CUADebug's debugger reads the whole run afterwards.

Inputs from the dataset (Claude Sonnet 4.5's 144 runs): each JSONL row is one action with the screenshot after it.
The action is translated to ADR's own action format (adr/env/osworld.py) from the recorded pyautogui command, which
holds the real screen position; the agent's reasoning is the row's `response`, or its step's when a row of a burst has
none. The Diagnostician also names CUADebug's top-level category, as in adr/rq1_offline.py plus "others" (O), and
CUADebug's subtype code. The date ADR gives the models is the run's recording date, as it would have been live. Every
model call is kept with its prompt, reply, tokens, cost and the images sent (history/, images as JPEG copies, full
screens at half size).

    python -m adr.v3.replay --stop-after 4      (then without --stop-after for the rest)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from PIL import Image

from .. import llm as llm_mod
from ..config import ROOT, Config
from ..llm import USAGE, usage_snapshot
from ..rq1_offline import DATA, EXTRA, FAMILY, ZIPS, human_type
from ..runstore import CaseStore, atomic_write, fingerprint, run_folder
from ..taxonomy import Taxonomy
from . import monitor as monitor_mod
from .diagnostician import DiagnosticianV3
from .monitor import MonitorV3

# the ADR v3.7 settings
V37 = ["adr_version=3", "click_check=v5", "retype=1", "date_context=1", "done_always=1", "no_effect_max_frac=0.1",
       "n_free=1", "drop_blocked=1", "diag_rules=1", "fix_report=1", "ground_fields=1", "max_recoveries=10",
       "replan_gap=3", "done_pairs=1", "use_axes=1", "replan_strikes=2", "gate_dismiss_first=1", "precommit_scope=all"]
OTHERS = ("O = Others (outside the four families), for example a task that is impossible by design, which the "
          "agent tried anyway instead of reporting that it cannot be done.\n")
# CUADebug's subtype codes (names from the CUAErrorBench dataset card), so that their subtype accuracy (L2) and
# subtype-and-step score can be computed too; O1 is their "infeasible task" (IF codes in the data)
TAGS = ("Also name CUADebug's subtype code, one that belongs to the family you named: "
        "P1 Visual Hallucination, P2 Misrecognition / OCR Error, P3 Cross-Modal Misbinding, P4 Observation Omission, "
        "P5 Semantic Misunderstanding; G1 Coordinate / Element Grounding Error, G2 Visibility / Accessibility Error, "
        "G3 Interaction Mechanics Error, G4 Distraction / Adversarial Misdirection; R1 Constraint Violation, "
        "R2 Infeasible Plan / Impossible Action, R3 Decomposition Failure, R4 Inefficient / Redundant Strategy, "
        "R5 Action-Intent Misalignment, R6 Invalid / Malformed Action, R7 Parameter / Argument Error, "
        "R8 Context Loss / Over-Simplification, R9 Memory Hallucination, R10 Progress Misjudgment, "
        "R11 Outcome Misinterpretation, R12 Failed Self-Correction, R13 Causal Misattribution; "
        "S1 Rendering / Layout Failure, S2 Timing / Race Condition, S3 Unexpected System Behavior, "
        "S4 Step / Resource Limit, S5 Tool / API Failure, S6 Environment Instability, "
        "S7 Benchmark / Evaluation Artifact; O1 Infeasible Task (impossible by design, attempted anyway).\n"
        'Add the key "cuadebug_tag": the code, for example "R7".\n')
CODES = {f"P{i}" for i in range(1, 6)} | {f"G{i}" for i in range(1, 5)} | {f"R{i}" for i in range(1, 14)} \
    | {f"S{i}" for i in range(1, 8)} | {"O1"}
FILES = ["adr/config.py", "adr/llm.py", "adr/vision.py", "adr/diagnostician.py", "adr/monitor.py", "adr/agent.py",
         "adr/taxonomy.py", "adr/rq1_offline.py", "adr/v3/monitor.py", "adr/v3/diagnostician.py", "adr/v3/replay.py"]
_NUM = r"(-?\d+(?:\.\d+)?)"
_LOCAL = threading.local()


# the run's own date for "today" (live, it was the day of the run)

_live_today_line = monitor_mod.today_line


def _today_line() -> str:
    d = getattr(_LOCAL, "date", None)
    return _live_today_line() if d is None else f"Today is {d:%A}, {d.day} {d:%B %Y}."


monitor_mod.today_line = _today_line            # read through the module by the Monitor and the Diagnostician


def run_date(rows) -> Optional[datetime.date]:
    m = re.match(r"(\d{4})(\d{2})(\d{2})@", str(rows[0].get("action_timestamp", "")))
    return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


# Claude's recorded actions in ADR's action format

def _point(cmd: str, fn: str):
    m = re.search(r"\b" + fn + r"\(\s*" + _NUM + r"\s*,\s*" + _NUM, cmd)
    return (int(float(m.group(1))), int(float(m.group(2)))) if m else None


def to_adr(a: dict) -> dict:
    """One recorded computer-use action as ADR's action (adr/env/osworld.py: click, type, key, scroll; others by
    name). Positions come from the executed pyautogui command (real screen pixels), not from Claude's scaled input."""
    inp = a.get("input") or {}
    kind = inp.get("action") or a.get("action_type") or "?"
    cmd = str(a.get("command") or "")
    clicks = {"left_click": "click", "double_click": "doubleClick", "right_click": "rightClick",
              "triple_click": "tripleClick", "middle_click": "middleClick"}
    if kind in clicks:
        p = _point(cmd, clicks[kind])
        d = {"type": "click", "x": p[0], "y": p[1]} if p else {"type": "click"}
        if kind == "double_click":
            d["double"] = True
        elif kind == "right_click":
            d["button"] = "right"
        elif kind == "middle_click":
            d["button"] = "middle"
        elif kind == "triple_click":
            d["triple"] = True
        return d
    if kind == "type":                                       # as the live adapter (adr/agent_uitars.py): a final
        text = str(inp.get("text", ""))                      # newline means "and press Enter"
        return {"type": "type", "text": text.rstrip("\n"), **({"submit": True} if text.endswith("\n") else {})}
    if kind == "key":
        return {"type": "key", "key": inp.get("text", "")}
    if kind == "scroll":
        m = re.search(r"\b(h?scroll)\(\s*" + _NUM + r"\s*,\s*(?:x=)?\s*" + _NUM + r"\s*,\s*(?:y=)?\s*" + _NUM, cmd)
        if m:
            n, x, y = float(m.group(2)), int(float(m.group(3))), int(float(m.group(4)))
            if m.group(1) == "hscroll":
                return {"type": "scroll", "dx": -n * 100, "dy": 0, "x": x, "y": y}
            return {"type": "scroll", "dy": -n * 100, "x": x, "y": y}        # pyautogui: negative = down; ADR: dy > 0
        return {"type": "scroll", "dy": 300 if inp.get("scroll_direction") == "down" else -300}
    if kind == "left_click_drag":
        s, e = _point(cmd, "moveTo"), _point(cmd, "dragTo")
        d = {"type": "drag"}
        if s:
            d.update(x=s[0], y=s[1])
        if e:
            d.update(x2=e[0], y2=e[1])
        return d
    if kind == "mouse_move":
        p = _point(cmd, "moveTo")
        return {"type": "move", "x": p[0], "y": p[1]} if p else {"type": "move"}
    if kind in ("wait", "screenshot", "cursor_position"):
        return {"type": "wait", "what": kind}
    return {"type": str(kind)}


def load_run(z: zipfile.ZipFile, prefix: str, task_dir: str):
    base = prefix + task_dir + "/"
    rows = [json.loads(l) for l in z.read(base + "traj.jsonl").decode().splitlines() if l.strip()]
    said = {}                                 # a step's reasoning, for rows of a burst that carry none
    for r in rows:
        t = str(r.get("response") or "").strip()
        if t and int(r["step_num"]) not in said:
            said[int(r["step_num"])] = t
    acts, done = [], None
    for r in rows:
        a = r["action"] if isinstance(r["action"], dict) else {}
        k = int(r["step_num"])
        thought = str(r.get("response") or "").strip() or said.get(k, "")
        if a.get("action_type") == "DONE" or (r.get("done") and not a.get("input")):
            done = {"step": k, "thought": thought or str(a.get("raw_response") or "").replace("[TEXT]", "").strip()}
            continue
        acts.append({"step": k, "action": to_adr(a), "recorded": {"input": a.get("input"), "command": a.get("command")},
                     "thought": thought, "shot": base + r["screenshot_file"] if r.get("screenshot_file") else None})
    return rows, acts, done


# the record of one case: every model call with its images

class Recorder:
    def __init__(self, folder):
        self.folder = folder
        self.calls = []
        self.n = 0

    def add(self, call: dict):
        self.n += 1
        self.folder.mkdir(parents=True, exist_ok=True)
        files = []
        for i, png in enumerate(call.pop("pngs", []) or [], 1):
            im = Image.open(io.BytesIO(png)).convert("RGB")
            size = im.size
            if im.size[0] > 1000:                 # full screens kept at half size; crops as they were
                im = im.resize((im.size[0] // 2, im.size[1] // 2))
            name = f"call{self.n:03d}_img{i}.jpg"
            im.save(self.folder / name, quality=80)
            files.append({"file": name, "sent_size": size})
        sysp = call.pop("system", "")
        h = hashlib.sha256(sysp.encode()).hexdigest()[:12]
        pdir = self.folder.parent.parent / "prompts"
        pdir.mkdir(exist_ok=True)
        if not (pdir / f"{h}.txt").exists():
            atomic_write(pdir / f"{h}.txt", sysp)
        self.calls.append({"n": self.n, "system_prompt": f"history/prompts/{h}.txt", "images": files, **call})


def _hook(call: dict):
    rec = getattr(_LOCAL, "rec", None)
    if rec is not None:
        rec.add(call)


llm_mod.CALL_HOOK = _hook


def replay(cfg: Config, tax: Taxonomy, z: zipfile.ZipFile, prefix: str, task_dir: str, goal: str) -> dict:
    """ADR v3.7 over one finished run, as live, until the first confirmed failure."""
    rows, acts, done = load_run(z, prefix, task_dir)
    _LOCAL.date = run_date(rows)
    monitor = MonitorV3(cfg)
    monitor.reset()
    diag = DiagnosticianV3(cfg, tax)
    fam = EXTRA.split("Here the failure")[0] + OTHERS + TAGS   # the category question, plus O and the codes; N
                                                                # stays allowed (it cancels a false alarm, as live)
    diag.system = (diag.system + fam).replace('"P" | "G" | "R" | "S"', '"P" | "G" | "R" | "S" | "O"')
    history, frames, trace = [], [], []
    shot = lambda p: Image.open(io.BytesIO(z.read(p))).convert("RGB")
    before = None
    for a in acts:
        after = shot(a["shot"]) if a["shot"] else before
        entry = {"action": a["action"], "thought": a["thought"]}
        history.append(entry)
        t = {"step": a["step"], "action": a["action"], "recorded": a["recorded"]}
        if before is None:                                      # the first action: no screen from before it
            t["judged"] = False
            trace.append(t)
            before = after
            continue
        frames.append(before)
        v = monitor.check(goal, before, after, a["thought"], a["action"], history)
        entry["verdict"] = v
        t.update(judged=True, flagged=v["flagged"], symptom=v.get("symptom"), source=v.get("source"),
                 confidence=v.get("confidence"), evidence=v.get("evidence"))
        if v["flagged"]:
            d = diag.diagnose(goal, history[:-1], v, before, after, a["thought"], a["action"], frames[:-1])
            t["diagnosis"] = d
            trace.append(t)
            ty = str(d.get("type", "")).strip().upper()
            if ty and ty != "N":
                return {"pred_step": a["step"], "pred_type": ty, **tag_family(d),
                        "found_by": "monitor", "diagnosis": d, "trace": trace, "date": str(_LOCAL.date)}
        else:
            trace.append(t)
        before = after
    if done is not None and before is not None:                 # the agent's DONE: the done-check, as live
        v = monitor.check_done(goal, before, done["thought"], first=None, history=history)
        t = {"step": done["step"], "action": {"type": "done"}, "done_check": True, "flagged": v["flagged"],
             "impossible": v.get("impossible"), "evidence": v.get("evidence"), "unmet": v.get("unmet")}
        if v["flagged"]:
            d = diag.diagnose(goal, history, v, before, before, done["thought"], {"type": "done"}, frames)
            t["diagnosis"] = d
            trace.append(t)
            ty = str(d.get("type", "")).strip().upper()
            if ty and ty != "N":
                return {"pred_step": done["step"], "pred_type": ty, **tag_family(d), "found_by": "done_check",
                        "diagnosis": d, "trace": trace, "date": str(_LOCAL.date)}
        else:
            trace.append(t)
    return {"pred_step": None, "pred_type": "", "pred_family": "", "pred_family_said": "", "pred_tag": "",
            "found_by": "none", "diagnosis": None,
            "trace": trace, "date": str(_LOCAL.date)}


def tag_family(d: dict) -> dict:
    """CUADebug's category comes from its tag (L1 is the tag's top level); the category the model named is kept too."""
    tag = re.sub(r"[^A-Z0-9]", "", str(d.get("cuadebug_tag", "")).upper())
    said = str(d.get("cuadebug_family", "")).strip().upper()[:1]
    return {"pred_tag": tag if tag in CODES else "", "pred_family": tag[0] if tag in CODES else said,
            "pred_family_said": said}


def gold_tag(tag: str) -> str:
    return "O1" if not tag[:1] in "PGRS" else tag             # IF1, IF2 and the free-text one: their "infeasible"


# scoring as CUADebug counts: every run; its fifth category "others" = the IF codes in the data

def summarize(results):
    n = max(1, len(results))
    fam_gold = lambda r: r["human_family"] if r["human_family"] in FAMILY else "O"
    within = lambda r, w: r["pred_step"] is not None and abs(r["pred_step"] - r["gold_step"]) <= w
    four = [r for r in results if r["human_family"] in FAMILY]
    typed = [r for r in results if r["human_type"]]
    return {
        "n": len(results),
        "answered": sum(r["pred_step"] is not None for r in results),
        "found_by": {k: sum(r["found_by"] == k for r in results) for k in ("monitor", "done_check", "none")},
        "step_exact": sum(r["pred_step"] == r["gold_step"] for r in results) / n,
        "step_within_2": sum(within(r, 2) for r in results) / n,
        "step_within_5": sum(within(r, 5) for r in results) / n,
        "family_accuracy_all": sum(r["pred_family"] == fam_gold(r) for r in results) / n,
        "family_said_accuracy_all": sum(r.get("pred_family_said") == fam_gold(r) for r in results) / n,
        "subtype_accuracy_all": sum(r.get("pred_tag") == gold_tag(r["tag"]) for r in results) / n,
        "subtype_and_step_exact": sum(r.get("pred_tag") == gold_tag(r["tag"]) and r["pred_step"] == r["gold_step"]
                                      for r in results) / n,
        "family_accuracy_four": sum(r["pred_family"] == r["human_family"] for r in four) / max(1, len(four)),
        "family_n_four": len(four),
        "type_accuracy": sum(r["pred_type"] == r["human_type"] for r in typed) / max(1, len(typed)),
        "type_n": len(typed),
        "pred_step_before_gold": sum(r["pred_step"] is not None and r["pred_step"] < r["gold_step"] for r in results),
        "pred_step_after_gold": sum(r["pred_step"] is not None and r["pred_step"] > r["gold_step"] for r in results),
    }


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="e33_adr37")
    ap.add_argument("--models", default="claude_4.5_sonnet")
    ap.add_argument("--config", default="configs/osworld_e25.json")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--stop-after", type=int, default=0, help="only this many of the open runs (the smoke test, same folder)")
    a = ap.parse_args(argv)
    cfg = Config.load(str(ROOT / a.config), V37 + a.set)
    tax = Taxonomy(cfg.path(cfg.taxonomy))
    fp = fingerprint(FILES, cfg.to_dict(), {"replay": "v3.7", "models": a.models, "others": True})
    out = run_folder(ROOT / "runs" / "rq1", a.name, fp)
    out.mkdir(parents=True, exist_ok=True)
    from ..provenance import code_version
    import sys
    meta_p = out / "config.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {**cfg.to_dict(), "_fingerprint": fp, "_sessions": []}
    meta["_sessions"].append({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "command": " ".join(sys.argv),
                              "provenance": code_version()})
    atomic_write(meta_p, json.dumps(meta, indent=1))
    print(f"Run folder: {out}", flush=True)
    store = CaseStore(out)
    tasks = json.loads((DATA / "tasks.json").read_text())
    df = pd.read_parquet(DATA / "train.parquet")
    rows = []
    for m in a.models.split(","):
        rows += [(m, x) for x in df[df.model == m].to_dict("records")]
    total = len(rows)
    rows = [(m, x) for m, x in rows if not store.has(f"{m}/{x['task_dir']}")]
    if a.stop_after:
        rows = rows[:a.stop_after]
    print(f"{total} runs, {total - len(rows)} done or left for later, {len(rows)} to replay", flush=True)
    zips = {m: zipfile.ZipFile(DATA / ZIPS[m][0]) for m in set(m for m, _ in rows)}
    zlock = threading.Lock()

    class LockedZip:                                         # one zip handle shared by the threads
        def __init__(self, z):
            self.z = z

        def read(self, name):
            with zlock:
                return self.z.read(name)

    def work(item):
        m, row = item
        cid = f"{m}/{row['task_dir']}"
        folder = out / "history" / cid.replace("/", "__")
        attempt = 1 + (sum(1 for _ in folder.glob("attempt_*")) if folder.exists() else 0)
        rec = Recorder(folder / f"attempt_{attempt}")
        _LOCAL.rec = rec
        t0 = time.time()
        try:
            res = replay(cfg, tax, LockedZip(zips[m]), ZIPS[m][1], row["task_dir"], tasks.get(row["task_dir"]) or "")
            err = None
        except Exception as e:                               # kept in the history; the run is retried next time
            res, err = None, f"{type(e).__name__}: {e}"
        finally:
            _LOCAL.rec = None
            _LOCAL.date = None
        rec.folder.mkdir(parents=True, exist_ok=True)
        atomic_write(rec.folder / "calls.json", json.dumps(rec.calls, indent=1, default=str))
        atomic_write(rec.folder / "result.json", json.dumps({"case": cid, "error": err, "seconds": round(time.time() - t0, 1),
                                                             "human_label_not_sent": {"root_step": int(row["root_error_step"]),
                                                                                      "tag": row["taxonomy_tag"]},
                                                             **(res or {})}, indent=1, default=str))
        if err:
            raise RuntimeError(err)
        return m, row, res, rec

    done_n = 0
    with ThreadPoolExecutor(a.threads) as ex:
        futs = [ex.submit(work, it) for it in rows]
        for f in as_completed(futs):
            try:
                m, row, res, rec = f.result()
            except Exception as e:
                print("error", e, flush=True)
                continue
            tag = row["taxonomy_tag"]
            r = {"id": f"{m}/{row['task_dir']}", "model": m, "tag": tag, "human_type": human_type(tax, tag),
                 "human_family": tag[:1], "gold_step": int(row["root_error_step"]), "num_steps": int(row["num_steps"]),
                 "pred_step": res["pred_step"], "pred_type": res["pred_type"], "pred_family": res["pred_family"],
                 "pred_family_said": res.get("pred_family_said", ""), "pred_tag": res.get("pred_tag", ""),
                 "found_by": res["found_by"], "calls": len(rec.calls),
                 "cost_usd": round(sum(float(((c.get("usage") or {}).get("cost")) or 0) for c in rec.calls), 6)}
            store.save(r["id"], r)
            done_n += 1
            u = USAGE
            cost = sum(v.get("cost_usd", 0) for v in u.values())
            print(f"[{done_n}/{len(rows)}] {r['id'].split('/')[-1][:8]} gold {r['gold_step']} {tag} | ADR {r['pred_step']} "
                  f"{r['pred_type']} {r['pred_tag'] or r['pred_family']} ({r['found_by']}) | {r['calls']} calls | cost so far ${cost:.2f}",
                  flush=True)
    results = store.all()
    atomic_write(out / "results.json", json.dumps(results, indent=1, default=str))
    usage = usage_snapshot()
    if (out / "usage.json").exists():                        # sessions add up (smoke test, then the rest)
        for tag, vals in json.loads((out / "usage.json").read_text()).items():
            for k, v in vals.items():
                usage.setdefault(tag, {})[k] = usage.get(tag, {}).get(k, 0) + v
    atomic_write(out / "usage.json", json.dumps(usage, indent=1))
    s = summarize(results)
    atomic_write(out / "summary.json", json.dumps(s, indent=1))
    meta = json.loads(meta_p.read_text())
    meta["_sessions"][-1]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta.update(_done=len(results), _planned=total)
    atomic_write(meta_p, json.dumps(meta, indent=1))
    print("DONE", json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
