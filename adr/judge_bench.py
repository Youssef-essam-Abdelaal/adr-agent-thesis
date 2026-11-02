"""
ADR's own checks on public judge benchmarks, scored with each benchmark's own metric, so the numbers sit next to
the published ones. Offline: no agent and no browser, only the judge model.

  gui_reflection  GUI-Reflection action verification (arXiv 2506.08012): 603 before/after pairs from GUI-Odyssey
                  (phone). Each pair is asked twice, with the step's true purpose (should pass) and a wrong one (should
                  fail); accuracy over both, as in the paper's script.
                  adr: the step judge (Monitor.check); the purpose is the agent's stated intention for the step, the
                  episode's instruction is the task. plain: the paper's prompt, the model alone.
  gui_critic      GUI-Critic-Test (arXiv 2506.04614): 1,198 steps (GUI-I and GUI-S phone, GUI-W web), half right and
                  half wrong, judged before the action runs; accuracy per subset.
                  adr: as ADR runs live: clicks and Enter go to the click check (check_commit), "terminate" to the
                  done-check, other actions are not checked (counted as "correct"). plain: the benchmark's prompt.
  cuastepbench    CUAStepBench (arXiv 2607.23263): 278 runs (desktop, phone, web) with a human good/bad label on every
                  step and a success label. adr: the step judge on every step in order, as live (loop rule included);
                  a step where the agent says it is done goes to the done-check; the outcome is the done-check on the
                  run's last screenshot. Step-error F1 as SeekJudge's analyse_judge (bad = positive, pooled over all
                  steps), and outcome accuracy.

    python -m adr.judge_bench --bench gui_reflection --name qwen32 --config configs/e23_open.json
    python -m adr.judge_bench --bench gui_critic --mode plain --name qwen32 --config configs/e23_open.json
    python -m adr.judge_bench --bench cuastepbench --name qwen32 --config configs/e23_open.json --limit 50

Data, from Hugging Face, under data/e23/: gui_reflection/ holds action_verification.jsonl of
craigwu/GUI_Reflection_Task_Suite_Benchmark and the GUI-Odyssey screenshots it names (OpenGVLab/GUI-Odyssey) in
images/; gui_critic/ holds the three jsonl files of BonnieOne/GUI-Critic-Test in raw/ and its image zips unpacked in
images/; cuastepbench/ is a copy of ZJUSCL/CUAStepBench.
Output: runs/e23/<bench>_<mode>_<name>__<fingerprint>/ (cases/, results.json, summary.json). --limit is not part of
the fingerprint, so a small first run and the full run share a folder. A rerun resumes; a case that failed (for
example the server was unreachable) is not saved, so it is retried.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from PIL import Image

from .config import ROOT, Config
from .llm import LLM, usage_snapshot
from .monitor import Monitor

DATA = ROOT / "data" / "e23"
JUDGE_FILES = ["adr/config.py", "adr/llm.py", "adr/vision.py", "adr/monitor.py", "adr/agent.py", "adr/judge_bench.py",
               "adr/v3/monitor.py"]
MAX_THOUGHT = 2000          # characters of an agent's reasoning passed on (long desktop traces)


def make_monitor(cfg: Config) -> Monitor:
    """ADR's Monitor of the configured version (v3: adr/v3/monitor.py)."""
    if cfg.adr_version >= 3:
        from .v3.monitor import MonitorV3
        return MonitorV3(cfg)
    return Monitor(cfg)


def load(path) -> Optional[Image.Image]:
    p = Path(path)
    return Image.open(p).convert("RGB") if p.exists() else None


# GUI-Reflection

# the paper's prompt (internvl_chat/eval/eval_tasks/reflection_action_verification_infer.py), without the image tags
REFLECTION_PROMPT = """Given an action purpose and two screenshots (the first screenshot corresponds to the step before a certain action while the second one is the outcome screenshot after the execution of the action). You need to judge whether the action purpose has been satisfied by the action executed between these screenshots based on the screenshots content. Directly answer Yes or No.

## Input
action purpose: {action_purpose}

## Output:
"""
YESNO_JSON = '\nReply with JSON only: {"answer": "Yes" or "No"}'
YESNO_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["Yes", "No"]}},
                "required": ["answer"]}


def reflection_items(limit: int = 0) -> list:
    d = DATA / "gui_reflection"
    rows = [json.loads(l) for l in open(d / "action_verification.jsonl") if l.strip()]
    out = [{"id": f"{i:04d}_{r['episode_id']}_{r['step_index']}", "task": r["instruction"],
            "before": str(d / "images" / r["images"][0]), "after": str(d / "images" / r["images"][1]),
            "action": r["action"], "pos": r["positive_action_purpose"], "neg": r["negative_action_purpose"]}
           for i, r in enumerate(rows)]
    return out[:limit] if limit else out


def reflection_action(raw: str, size) -> dict:
    """CLICK[[x, y]] and SCROLL[[x1, y1, x2, y2]] on a 0-1000 grid, as an action in image pixels."""
    w, h = size
    n = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", raw)]
    if raw.startswith("CLICK") and len(n) >= 2:
        return {"type": "click", "x": round(n[0] / 1000 * w), "y": round(n[1] / 1000 * h)}
    if raw.startswith("SCROLL") and len(n) >= 4:
        x1, y1, x2, y2 = (round(v / 1000 * s) for v, s in zip(n[:4], (w, h, w, h)))
        return {"type": f"swipe from ({x1}, {y1}) to ({x2}, {y2})", "x": x1, "y": y1}
    return {"type": raw}


def run_reflection(item: dict, cfg: Config, mode: str, llm: LLM) -> dict:
    before, after = load(item["before"]), load(item["after"])
    if before is None or after is None:
        raise FileNotFoundError("screenshot missing")
    out = {"id": item["id"], "action": item["action"]}
    if mode == "plain":
        for k in ("pos", "neg"):
            r = llm.ask("", REFLECTION_PROMPT.format(action_purpose=item[k]) + YESNO_JSON, [before, after],
                        tag="judge", schema=YESNO_SCHEMA)
            out[k] = {"purpose": item[k], "satisfied": str(r.get("answer", "")).strip().lower().startswith("yes"),
                      "reply": r}
    else:
        mon = make_monitor(cfg)
        a = reflection_action(item["action"], before.size)
        for k in ("pos", "neg"):
            mon.reset()
            thought = f"My intention for this step: {item[k]}."
            v = mon.check(item["task"], before, after, thought, a, [{"action": a, "thought": thought}])
            j = v.get("judge") or {}
            # v3 judges the agent's intention apart from the task: the benchmark asks about the intention
            satisfied = bool(j["intent_met"]) if cfg.adr_version >= 3 and "intent_met" in j else not v["flagged"]
            out[k] = {"purpose": item[k], "satisfied": satisfied, "flagged": v["flagged"], "verdict": v}
    out["correct_pos"] = out["pos"]["satisfied"]
    out["correct_neg"] = not out["neg"]["satisfied"]
    return out


# GUI-Critic-Test

CRITIC_FILES = {"GUI-I": "gui_i.jsonl", "GUI-S": "gui_s.jsonl", "GUI-W": "gui_web.jsonl"}
CRITIC_MARK = "Below is the information for the current step:"
CRITIC_JSON = ('\n\nReply with JSON only: {"thinking": "<observation, possible result and critique>", '
               '"score": "Correct" or "Incorrect", "suggestion": "<brief>"}')
CRITIC_SCHEMA = {"type": "object", "properties": {"thinking": {"type": "string"},
                                                  "score": {"type": "string", "enum": ["Correct", "Incorrect"]},
                                                  "suggestion": {"type": "string"}},
                 "required": ["thinking", "score", "suggestion"]}


def _section(text: str, start: str, end: Optional[str]) -> str:
    s = text.split(start, 1)[1] if start in text else ""
    return (s.split(end, 1)[0] if end and end in s else s).strip()


def critic_items(limit: int = 0) -> list:
    """limit applies per subset, so a small test covers all three."""
    out = []
    for subset, f in CRITIC_FILES.items():
        rows = [json.loads(l) for l in open(DATA / "gui_critic" / "raw" / f) if l.strip()]
        for i, r in enumerate(rows[:limit] if limit else rows):
            tail = r["problem"].split(CRITIC_MARK, 1)[1]
            hist = _section(tail, "2. History:", "3. Decision:")
            out.append({"id": f"{subset}_{i:04d}", "subset": subset,
                        "image": str(DATA / "gui_critic" / "images" / r["images"][0]),
                        "task": _section(tail, "1. User instruction:", "2. History:"),
                        "history": [s.strip(" ;\n") for s in
                                    re.findall(r"Step \d+:.*?(?=\n\s*;?\s*Step \d+:|\Z)", hist, re.S)],
                        "decision": _section(tail, "3. Decision:", "4. Image"),
                        "problem": r["problem"].replace("<image>\n", ""),
                        "label_correct": r["solution"].strip().lower().startswith("correct")})
    return out


def critic_action(decision: str) -> dict:
    """The benchmark's decision as one of our actions (coordinates in image pixels)."""
    s = decision[decision.find("{"): decision.rfind("}") + 1]
    try:
        d = json.loads(s)
    except Exception:
        try:
            d = ast.literal_eval(s)                    # Python-style dicts in the phone subsets
        except Exception:
            return {"type": decision[:160]}
    if isinstance(d.get("arguments"), dict):
        d = d["arguments"]
    kind, c = str(d.get("action", "")), d.get("coordinate")
    if kind in ("click", "left_click") and c:
        return {"type": "click", "x": c[0], "y": c[1]}
    if kind == "swipe" and c and d.get("coordinate2"):
        c2 = d["coordinate2"]
        return {"type": f"swipe from ({c[0]}, {c[1]}) to ({c2[0]}, {c2[1]})", "x": c[0], "y": c[1]}
    if kind == "type":
        return {"type": f"type {d.get('text', '')!r}"}
    if kind == "terminate":
        return {"type": "done", "status": d.get("status")}
    if kind == "system_button":
        return {"type": "key", "key": str(d.get("button"))}
    if kind == "key":
        return {"type": "key", "key": "+".join(str(k) for k in d.get("keys", []))}
    if kind == "scroll":
        return {"type": f"scroll by {d.get('pixels')} pixels"}
    return {"type": json.dumps(d)[:160]}


def run_critic(item: dict, cfg: Config, mode: str, llm: LLM) -> dict:
    shot = load(item["image"])
    if shot is None:
        raise FileNotFoundError("screenshot missing")
    a = critic_action(item["decision"])
    out = {"id": item["id"], "subset": item["subset"], "label_correct": item["label_correct"], "action": a}
    if mode == "plain":
        r = llm.ask("", item["problem"] + CRITIC_JSON, [shot], tag="judge", schema=CRITIC_SCHEMA)
        return {**out, "route": "plain", "pred_correct": str(r.get("score", "")).strip().lower().startswith("correct"),
                "reply": r}
    mon = make_monitor(cfg)
    hist = [{"action": {"type": s}} for s in item["history"]]
    if a["type"] == "done":
        v = mon.check_done(item["task"], shot, "")
        return {**out, "route": "done_check", "pred_correct": not v["flagged"], "verdict": v}
    if a["type"] == "click" or (a["type"] == "key" and a.get("key", "").lower() == "enter"):
        v = mon.check_commit(item["task"], shot, "", a, hist)
        return {**out, "route": "click_check", "pred_correct": not v["blocked"], "verdict": v}
    return {**out, "route": "not_checked", "pred_correct": True}


# CUAStepBench

NUM = r"-?\d+(?:\.\d+)?"
DONE_CALLS = ("terminate(", "send_msg_to_user(", "answer(", "finished(")


def cua_runs(limit: int = 0) -> list:
    """All runs with a human label, in a fixed random order (so --limit is a fair sample). Label i belongs to step i
    (SeekJudge: "keyed by the 0-based action index"); most runs end with one extra step without an action."""
    base, runs = DATA / "cuastepbench", []
    for meta in base.rglob("metafile.json"):
        d = meta.parent
        res = d / "judge" / "human-annotated-zzh" / "result.json"
        if not res.exists():
            continue
        m, r = json.loads(meta.read_text()), json.loads(res.read_text())
        ann = r.get("step_annotations") or {}
        rel = d.relative_to(base)
        runs.append({"id": str(rel), "benchmark": rel.parts[0], "source": m.get("benchmark"), "goal": m.get("goal", ""),
                     "dir": str(d), "steps": m.get("steps", []),
                     "labels": [ann[k] for k in sorted(ann, key=int)],
                     "success": str(r.get("judge_result")).lower() == "true"})
    runs.sort(key=lambda x: hashlib.sha256(x["id"].encode()).hexdigest())
    return runs[:limit] if limit else runs


def _xy(s: str, size) -> Optional[tuple]:
    """The first position in an action: x=..,y=.. (pyautogui, fractions of the screen), [x, y] (mobile_use, pixels)
    or (x, y) (mouse_click, scroll_at, pixels)."""
    m = (re.search(rf"x\s*=\s*({NUM})\s*,\s*y\s*=\s*({NUM})", s) or re.search(rf"\[\s*({NUM})\s*,\s*({NUM})", s)
         or re.search(rf"\(\s*({NUM})\s*,\s*({NUM})", s))
    if not m:
        return None
    x, y = float(m.group(1)), float(m.group(2))
    w, h = size
    if x <= 1.0 and y <= 1.0:                    # a fraction of the screen
        x, y = x * w, y * h
    return round(x), round(y)


def _typed(text: str, last_xy: Optional[tuple]) -> dict:
    """Typing goes where the agent clicked last, as ADR records it live for UI-TARS."""
    text = text[:200]
    return {"type": "type", "x": last_xy[0], "y": last_xy[1], "text": text} if last_xy else {"type": f"type {text!r}"}


def cua_action(raw, size, last_xy: Optional[tuple]) -> dict:
    """One step's action as one of our actions, so the judge sees the same kind of input as live: a red circle on a
    click, typing at the last clicked position, keys, a "done" claim (to the done-check), or the action as written
    (element ids, scrolls, drags, waits). Formats: pyautogui (desktop), BrowserGym (WebArena family), Online-Mind2Web,
    the phone formats of GUI-Odyssey and AndroidControl (0-1000 grid) and Mobile-R1 (pixels)."""
    s = str(raw or "").strip()
    if not s or s == "None":
        return {"type": "none"}
    w, h = size
    word = s.split("(")[0].split()[0]
    low = s.lower()
    if word in ("COMPLETE", "INCOMPLETE") or any(k in low for k in DONE_CALLS):
        return {"type": "done", "raw": s[:300]}
    if word in ("CLICK", "LONG_PRESS", "SCROLL") and "[[" in s:             # phone, 0-1000 grid
        n = [float(v) for v in re.findall(NUM, s)]
        x, y = round(n[0] / 1000 * w), round(n[1] / 1000 * h)
        if word == "CLICK":
            return {"type": "click", "x": x, "y": y}
        if word == "SCROLL" and len(n) >= 4:
            return {"type": f"swipe from ({x}, {y}) to ({round(n[2] / 1000 * w)}, {round(n[3] / 1000 * h)})",
                    "x": x, "y": y}
        return {"type": f"long press at ({x}, {y})", "x": x, "y": y}
    if word in ("TEXT", "TYPE"):
        return _typed(s.split(None, 1)[1].strip("'\"") if " " in s else "", last_xy)
    if word == "PRESS_BACK":
        return {"type": "key", "key": "Back"}
    if word == "fill" or word == "select_option":                            # BrowserGym: element id and value
        return {"type": s[:200]}
    m = re.search(r"(?:write|typewrite|keyboard_type|\.type|^type)\(\s*(?:message=|text=)?['\"](.*?)['\"]", s, re.S)
    if m and "click" not in word.lower():
        return _typed(m.group(1), last_xy)
    xy = None if word in ("scroll", "pyautogui.scroll") else _xy(s, size)  # BrowserGym scroll(dx, dy) is no position
    wl = word.lower()
    if xy and ("click" in wl or "tap" in wl):
        if "right" in wl or "triple" in wl:                                  # shown as written, with the red circle
            return {"type": s[:200], "x": xy[0], "y": xy[1]}
        a = {"type": "click", "x": xy[0], "y": xy[1]}
        if "double" in wl:
            a["double"] = True
        return a
    if any(k in wl for k in ("press", "hotkey", "keyboard_down", "keyboard_up")):
        keys = re.findall(r"['\"]([\w ]+)['\"]", s)
        return {"type": "key", "key": "+".join(keys) or s[:60]}
    a = {"type": s[:200]}                                                    # as written
    if xy:
        a["x"], a["y"] = xy
    return a


def run_cua(run: dict, cfg: Config, mode: str, llm: LLM) -> dict:
    d, steps, n = Path(run["dir"]), run["steps"], len(run["labels"])

    def path(i) -> Optional[Path]:
        """Screenshot of step i (before its action). Web runs store paths from the original machine, so the file
        is looked up by name in screenshots/ too."""
        if i >= len(steps) or not steps[i].get("screenshot_path"):
            return None
        p = Path(steps[i]["screenshot_path"])
        for c in (d / p, d / "screenshots" / p.name):
            if c.exists():
                return c
        return None

    mon, hist, preds, last_xy, last_thought = make_monitor(cfg), [], [], None, ""
    v3 = cfg.adr_version >= 3
    first = load(path(0)) if v3 and path(0) else None       # v3's done-check also sees the start screen
    for i in range(n):
        pb, pa = path(i), path(i + 1)
        if pa is not None and pa == pb:              # the final step repeats the last screenshot: no after-image
            pa = None
        step = steps[i] if i < len(steps) else {}
        thought = str(step.get("reasoning") or "")[:MAX_THOUGHT]
        if pb is None:
            preds.append({"bad": False, "route": "no_screenshot"})
            continue
        before, after = load(pb), (load(pa) if pa else None)
        last_thought = thought or last_thought
        a = cua_action(step.get("action"), before.size, last_xy)
        if a["type"] == "done":
            v = (mon.check_done(run["goal"], before, thought, first=first, history=hist) if v3 else
                 mon.check_done(run["goal"], before, thought))
            preds.append({"bad": v["flagged"], "route": "done_check", "action": a, "verdict": v})
        else:
            hist.append({"action": a, "thought": thought})
            v = mon.check(run["goal"], before, after if after is not None else before, thought, a, hist)
            preds.append({"bad": v["flagged"], "route": "judge" if after is not None else "judge_no_after",
                          "action": a, "verdict": v})
        if "x" in a:
            last_xy = (a["x"], a["y"])
    last = next((p for p in (path(i) for i in range(len(steps) - 1, -1, -1)) if p is not None), None)
    outcome = None
    if last:
        outcome = (mon.check_done(run["goal"], load(last), last_thought, first=first, history=hist) if v3 else
                   mon.check_done(run["goal"], load(last), last_thought))
    return {"id": run["id"], "benchmark": run["benchmark"], "labels": run["labels"], "success": run["success"],
            "pred_bad": [p["bad"] for p in preds], "routes": [p["route"] for p in preds], "steps": preds,
            "pred_success": (not outcome["flagged"]) if outcome else None, "outcome": outcome}


# scoring

def prf(y_true: list, y_pred: list) -> dict:
    """Accuracy, precision, recall and F1 with True as the positive class (as compute_metrics in SeekJudge)."""
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    n = len(y_true)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": n, "accuracy": (tp + tn) / n if n else 0.0, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def score(bench: str, results: list) -> dict:
    if bench == "gui_reflection":
        n = len(results)
        pos = sum(r["correct_pos"] for r in results)
        neg = sum(r["correct_neg"] for r in results)
        return {"pairs": n, "accuracy": (pos + neg) / (2 * n) if n else 0.0,
                "accuracy_true_purpose": pos / n if n else 0.0, "accuracy_wrong_purpose": neg / n if n else 0.0}
    if bench == "gui_critic":
        out = {}
        for subset in list(CRITIC_FILES) + ["all"]:
            rs = [r for r in results if subset in ("all", r["subset"])]
            if not rs:
                continue
            ok = [r["pred_correct"] == r["label_correct"] for r in rs]
            checked = [r for r in rs if r["route"] != "not_checked"]
            out[subset] = {"n": len(rs), "accuracy": sum(ok) / len(rs),
                           "routes": dict(Counter(r["route"] for r in rs)),
                           "accuracy_checked_only": (sum(r["pred_correct"] == r["label_correct"] for r in checked)
                                                     / len(checked)) if checked else None,
                           "wrong_actions_caught": prf([not r["label_correct"] for r in rs],
                                                       [not r["pred_correct"] for r in rs])}
        return out
    # cuastepbench: steps pooled over runs, bad = positive (SeekJudge analyse_judge, "TF flipped")
    yt, yp, by = [], [], defaultdict(lambda: ([], []))
    for r in results:
        t = [lab != "good" for lab in r["labels"]]
        p = list(r["pred_bad"])
        yt += t
        yp += p
        by[r["benchmark"]][0].extend(t)
        by[r["benchmark"]][1].extend(p)
    oc = [r for r in results if r.get("pred_success") is not None]
    return {"runs": len(results), "steps": len(yt), "step_error": prf(yt, yp),
            "step_error_by_benchmark": {b: prf(t, p) for b, (t, p) in sorted(by.items())},
            "routes": dict(Counter(x for r in results for x in r["routes"])),
            "outcome_success_positive": prf([r["success"] for r in oc], [r["pred_success"] for r in oc]),
            "outcome_failure_positive": prf([not r["success"] for r in oc], [not r["pred_success"] for r in oc])}


# runner

def out_folder(a, cfg: Config) -> Path:
    from .provenance import code_version
    from .runstore import atomic_write, fingerprint, other_versions, read_result, run_folder
    extra = {"bench": a.bench, "mode": a.mode}
    fp = fingerprint(JUDGE_FILES, cfg.to_dict(), extra)
    base, name = ROOT / "runs" / "e23", f"{a.bench}_{a.mode}_{a.name}"
    out = run_folder(base, name, fp)
    if not out.exists() and other_versions(base, name, fp):
        print(f"Code or settings differ from earlier versions of '{name}'; starting a new folder {out.name}. "
              f"Earlier results are kept.", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "config.json"
    meta = read_result(p) or {**cfg.to_dict(), "_provenance": code_version(), "_fingerprint": fp, "_sessions": [],
                              "_bench": a.bench, "_mode": a.mode}
    meta.setdefault("_sessions", []).append({"started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                             "command": " ".join(sys.argv), "provenance": code_version()})
    atomic_write(p, json.dumps(meta, indent=1))
    print(f"Run folder: {out}", flush=True)
    return out


ITEMS = {"gui_reflection": reflection_items, "gui_critic": critic_items, "cuastepbench": cua_runs}
RUN = {"gui_reflection": run_reflection, "gui_critic": run_critic, "cuastepbench": run_cua}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bench", required=True, choices=list(ITEMS))
    ap.add_argument("--mode", default="adr", choices=["adr", "plain"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N items (per subset for gui_critic; a fixed random sample of runs for cuastepbench)")
    a = ap.parse_args(argv)
    if a.mode == "plain" and a.bench == "cuastepbench":
        sys.exit("plain mode exists for gui_reflection and gui_critic only")
    from .runstore import CaseStore, atomic_write
    cfg = Config.load(a.config, a.set)
    llm = LLM.for_role(cfg, "judge")
    out = out_folder(a, cfg)
    items = ITEMS[a.bench](a.limit)
    store = CaseStore(out)
    todo = [it for it in items if not store.has(it["id"])]
    print(f"{len(items)} items, {len(items) - len(todo)} already done, {len(todo)} to run", flush=True)
    errors, t0 = 0, time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        futs = {ex.submit(RUN[a.bench], it, cfg, a.mode, llm): it for it in todo}
        for i, f in enumerate(as_completed(futs), 1):
            it = futs[f]
            try:
                rec = f.result()
            except Exception as e:                     # not saved, so retried by the next run
                errors += 1
                print(f"error {it['id']}: {type(e).__name__}: {str(e)[:150]}", flush=True)
                continue
            store.save(it["id"], rec)
            if i % 25 == 0 or i == len(todo):
                s = score(a.bench, store.all())
                head = (s.get("accuracy") if a.bench == "gui_reflection" else
                        s.get("all", {}).get("accuracy") if a.bench == "gui_critic" else s["step_error"]["f1"])
                print(f"[{i}/{len(todo)}] {time.time() - t0:.0f} s, so far {head:.3f}", flush=True)
    results = store.all()
    summ = {"bench": a.bench, "mode": a.mode, "model": cfg.judge_model, "errors_this_session": errors,
            **score(a.bench, results)}
    atomic_write(out / "results.json", json.dumps(results, indent=1, default=str))
    atomic_write(out / "summary.json", json.dumps(summ, indent=1))
    atomic_write(out / "usage.json", json.dumps(usage_snapshot(), indent=1))
    print("DONE", json.dumps({k: v for k, v in summ.items() if k not in ("step_error_by_benchmark",)})[:1500],
          flush=True)
    if errors:
        sys.exit(1)                                    # a calling script can retry until every case is saved


if __name__ == "__main__":
    main()
