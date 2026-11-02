"""
RQ1 on real failures: CUAErrorBench (CUADebug, arXiv 2608.02643).

Failed OSWorld runs by Claude Sonnet 4.5 and Gemini 2.5 Pro, each with a human
root-cause step and a tag from CUADebug's taxonomy. The Diagnostician gets the
evidence at the human root step (the same evidence pack it gets live) and
classifies it with taxonomy v4; human tags are mapped to v4 through the
`cuadebug` field of the taxonomy file. It also names CUADebug's top-level
category (P/G/R/S), so the result can be set against the numbers in that paper.

    python -m adr.rq1_offline --name full                     # trajectory context (default)
    python -m adr.rq1_offline --name twoframe --set context_frames=0 --set history_steps=0
    python -m adr.rq1_offline --name noreason --mask-reasoning
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import re
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from .config import ROOT, Config
from .llm import LLM, usage_snapshot
from .taxonomy import Taxonomy
from .diagnostician import SYSTEM
from .vision import crop_zoom, mark_click, thumb

DATA = ROOT / "data" / "cuaerrorbench"
ZIPS = {"claude_4.5_sonnet": ("claude.zip", "claude_4.5_traj_144/"),
        "gemini_2.5_pro": ("gemini.zip", "gemini_2.5_pro_traj_30/")}
FAMILY = {"P": "Perception", "G": "Grounding & Interaction", "R": "Task Reasoning & Control", "S": "External / System"}
EXTRA = """
Also name the top-level family in the CUADebug taxonomy, for comparison with prior work:
P = Perception (misperceives the screen), G = Grounding & Interaction (where or how to act: coordinates,
visibility, interaction mechanics, distraction), R = Task Reasoning & Control (planning, constraints,
action-intent misalignment, parameters, context, progress or outcome judgement, self-correction),
S = External / System (rendering, timing, unexpected system behaviour, tool failure, instability).
Add the key "cuadebug_family": "P" | "G" | "R" | "S" to your JSON.
Here the failure is confirmed by a human analysis, so do not answer N: choose the type among I-VI that best
explains why this step is the root error (skip Q0)."""

POINT = re.compile(r"(?:click|doubleClick|rightClick|tripleClick|moveTo|dragTo)\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)")
SCALE = 2 / 3   # 1920x1080 -> 1280x720


def rq1_folder(a, cfg, extra: dict) -> Path:
    """Versioned output folder for an offline run: runs/rq1/<name>__<fingerprint> (see adr/runstore.py)."""
    import sys as _sys
    import time as _t
    from .provenance import code_version
    from .runstore import RQ1_FILES, atomic_write, fingerprint, other_versions, read_result, run_folder
    fp = fingerprint(RQ1_FILES, cfg.to_dict(), extra)
    base = ROOT / "runs" / "rq1"
    out = base / a.name if getattr(a, "legacy_folder", False) else run_folder(base, a.name, fp)
    if not out.exists() and not getattr(a, "legacy_folder", False) and other_versions(base, a.name, fp):
        print(f"Code or settings differ from earlier versions of '{a.name}'; starting a new folder {out.name}. "
              f"Earlier results are kept.", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "config.json"
    meta = read_result(p) or {**cfg.to_dict(), "_provenance": code_version(), "_fingerprint": fp,
                              "_sessions": [], **{f"_{k}": v for k, v in extra.items()}}
    meta.setdefault("_sessions", []).append({"started": _t.strftime("%Y-%m-%d %H:%M:%S"),
                                             "command": " ".join(_sys.argv), "provenance": code_version()})
    atomic_write(p, json.dumps(meta, indent=1))
    print(f"Run folder: {out}", flush=True)
    return out


def _mark_finished(out_dir, done: int = None, planned: int = None):
    """Record the finish time and completeness in the run's config.json."""
    import time as _t
    from .runstore import atomic_write, read_result
    p = Path(out_dir) / "config.json"
    meta = read_result(p) or {}
    meta["_finished"] = _t.strftime("%Y-%m-%d %H:%M:%S")
    if done is not None:
        meta.update(_episodes=done, _planned=planned, _partial=bool(planned and done < planned))
        if meta.get("_sessions"):
            meta["_sessions"][-1]["finished"] = meta["_finished"]
    atomic_write(p, json.dumps(meta, indent=1))


def human_type(tax: Taxonomy, tag: str):
    for t in tax.data["types"]:
        if tag in t.get("cuadebug", []):
            return t["id"]
    return None   # S4, IF1, IF2, free text: not an agent failure cause


def _steps(rows, model):
    steps = defaultdict(lambda: {"action": [], "reason": [], "shot": None})
    for r in rows:
        k = int(r["step_num"])
        s = steps[k]
        s["shot"] = r.get("screenshot_file") or s["shot"]
        if model.startswith("claude"):
            try:
                a = ast.literal_eval(r["action"])
            except Exception:
                a = {"command": str(r["action"])}
            inp = a.get("input", {}) if isinstance(a, dict) else {}
            cmd = (a.get("command") or "").strip() if isinstance(a, dict) else ""
            s["action"].append(f"{inp.get('action', '')} {inp.get('coordinate', '') or ''} "
                               f"{inp.get('text', '') or ''} -> {cmd[:160]}".strip())
            think = (a.get("raw_response") or "") if isinstance(a, dict) else ""
            s["reason"].append(((r.get("response") or "") + " " + think)[:900])
        elif model == "uitars":
            code = str(r.get("action", ""))
            s["action"].append(code.split("\'\'\'")[-1].replace("import pyautogui", "").replace("import time", "").strip()[:200])
            s["reason"].append((r.get("_thought_en") or r.get("response") or "")[:900])
        else:
            s["action"].append(f"{r.get('plan_code', '')} -> {str(r.get('action', ''))[:160]}")
            plan = r.get("executor_plan") or ""
            i = plan.find("(Next Action)")
            s["reason"].append((plan[i:] if i >= 0 else plan[-900:])[:900])
    return dict(steps)


def load_case(z, prefix: str, row: dict, model: str, tasks: dict, traj_name: str = "traj.jsonl"):
    base = prefix + row["task_dir"] + "/"
    rows = [json.loads(l) for l in z.read(base + traj_name).decode().splitlines() if l.strip()]
    steps = _steps(rows, model)
    k = int(row["root_error_step"])

    def img(step):
        s = steps.get(step)
        if not s or not s["shot"]:
            return None
        try:
            im = Image.open(io.BytesIO(z.read(base + s["shot"]))).convert("RGB")
        except KeyError:
            return None
        return im.resize((int(im.size[0] * SCALE), int(im.size[1] * SCALE)))

    root = steps.get(k, {"action": [], "reason": []})
    m = POINT.search(" ".join(root["action"]))
    point = (float(m.group(1)) * SCALE, float(m.group(2)) * SCALE) if m else None
    return {
        "id": f"{model}/{row['task_dir']}", "model": model, "app": row["app"], "tag": row["taxonomy_tag"],
        "confidence": row["confidence"], "step": k, "task": tasks.get(row["task_dir"]) or "",
        "history": [(i, steps[i]) for i in sorted(steps) if k - 6 <= i < k],
        "all_steps": [(i, steps[i]) for i in sorted(steps)],
        "root": root, "point": point,
        "before": img(k - 1), "after": img(k), "frames": [f for f in (img(k - 3), img(k - 2)) if f is not None],
    }


V3_TO_V4 = {"I": "I", "II": "II", "III": "III", "V": "V", "VI": "VI"}


def v3_to_v4(category: str, subcategory: str):
    m = re.match(r"\s*(VI|IV|V|III|II|I)\b", category or "")
    if not m:
        return None
    t = m.group(1)
    if t == "IV":
        return "V" if "Goal" in (subcategory or "") else "IV"
    return V3_TO_V4[t]


class DirZip:
    """Reads a plain folder the way load_case reads a zip file."""
    def __init__(self, root):
        self.root = Path(root)

    def read(self, name):
        p = self.root / name
        if not p.exists():
            raise KeyError(name)
        return p.read_bytes()


ROUND2 = ROOT / "labels" / "osworld34" / "round2"


def round2_reference():
    """Second-round labels: per case the step (the fixed step, or the step at least two annotators chose; None if
    there is none), the majority type (2 of 3, else None) and each annotator's type."""
    from collections import Counter
    L = {f.stem.replace("labels_v4_", ""): json.loads(f.read_text())["labels"] for f in sorted(ROUND2.glob("labels_v4_*.json"))}
    ref = {}
    for fid in sorted(next(iter(L.values()))):
        s = Counter(L[n][fid]["failure_step"] for n in L).most_common(1)[0]
        t = Counter(L[n][fid]["type"] for n in L).most_common(1)[0]
        ref[fid] = {"step": int(s[0]) if s[1] >= 2 else None, "majority": t[0] if t[1] >= 2 else None,
                    "types": {n: L[n][fid]["type"] for n in L}, "fixed": L[next(iter(L))][fid]["step_fixed"] == "yes"}
    return ref


def osworld34_cases(tasks_unused=None, steps="a"):
    """steps="a": annotator A's first-round step. steps="round2": the second-round step; cases without one
    (no fixed step and no step chosen by two annotators) are left out."""
    import csv
    root = ROOT / "data"
    trans = json.loads((root / "translations.json").read_text())
    tasks = json.loads((root / "tasks.json").read_text()) if (root / "tasks.json").exists() else {}
    lab = ROOT / "labels" / "osworld34"
    a = {r["fid"]: r for r in csv.DictReader(open(lab / "annotator_A.csv"))}
    b = json.loads((lab / "annotator_B.json").read_text())
    z = DirZip(root / "extracted")
    r2 = round2_reference() if steps == "round2" else {}
    cases = []
    for fid in sorted(r2 if r2 else a):
        if r2 and r2[fid]["step"] is None:
            continue
        info = dict(l.split(": ", 1) for l in (root / "failures" / fid / "info.txt").read_text().splitlines() if ": " in l)
        td = info["trajectory"].strip()
        step = r2[fid]["step"] if r2 else (a[fid].get("failure_step") or info.get("failure_step_num"))
        rows = [json.loads(l) for l in (root / "extracted" / td / "traj.jsonl").read_text().splitlines() if l.strip()]
        for i, r in enumerate(rows):
            r["_thought_en"] = trans.get(f"{td}|row{i}|thought")
        (root / "extracted" / td / "_rows.cache").write_text("\n".join(json.dumps(r) for r in rows))
        row = {"task_dir": td, "app": "chrome", "taxonomy_tag": fid, "confidence": "high",
               "root_error_step": int(step)}
        uuid = td.split("/")[-1]
        c = load_case(z, "", row, "uitars", {td: info.get("task") or tasks.get(uuid, {}).get("instruction", "")},
                      traj_name="_rows.cache")
        c["gold_a"] = v3_to_v4(a[fid]["category"], a[fid]["subcategory"])
        c["gold_b"] = v3_to_v4(b.get(fid, {}).get("category", ""), b.get(fid, {}).get("subcategory", ""))
        if r2:
            c.update(gold_r2=r2[fid]["majority"], r2_types=r2[fid]["types"], r2_fixed=r2[fid]["fixed"])
        cases.append(c)
    return cases


def diagnose_case(case, llm, system, cfg, mask_reasoning=False, full_run=False):
    imgs, labels = [], []
    b, a, p = case["before"], case["after"], case["point"]
    if b is not None:
        imgs.append(mark_click(b, *p) if p else b)
        labels.append(f"Image {len(imgs)}: BEFORE the root step (red circle = where the action landed)")
    if a is not None:
        imgs.append(a)
        labels.append(f"Image {len(imgs)}: AFTER the root step")
    if p and b is not None and a is not None:
        imgs += [crop_zoom(b, *p, half=180), crop_zoom(a, *p, half=180)]
        labels += [f"Image {len(imgs) - 1}: BEFORE zoomed around the action",
                   f"Image {len(imgs)}: AFTER zoomed around the action"]
    if cfg.context_frames > 0:
        for f in case["frames"][-cfg.context_frames:]:
            imgs.append(thumb(f, 640))
            labels.append(f"Image {len(imgs)}: earlier frame (oldest first)")
    hist = []
    if cfg.history_steps > 0:
        for i, s in case["history"][-cfg.history_steps:]:
            r = "" if mask_reasoning else f" | reasoning: {' '.join(s['reason'])[:220]}"
            hist.append(f"- step {i}: {' ; '.join(s['action'])[:200]}{r}")
    reason = "(hidden)" if mask_reasoning else " ".join(case["root"]["reason"])[:1500]
    if full_run:               # annotator mode: every step, before and after the failure step, as the annotators saw it
        run = [f"- step {i}{'  <== FAILURE STEP' if i == case['step'] else ''}: {' ; '.join(s['action'])[:300]}"
               + ("" if mask_reasoning else f" | reasoning: {' '.join(s['reason'])[:900]}") for i, s in case["all_steps"]]
        text = "\n".join([
            f"Task: {case['task']}",
            "This run FAILED. The failure step is marked below.",
            "The whole run (every step, oldest first):", *run, "",
            f"FAILURE STEP {case['step']}",
            f"Agent reasoning: {reason}",
            f"Action: {' ; '.join(case['root']['action'])[:400]}", "",
            *labels,
        ])
        return llm.ask(system, text, imgs, tag="diag")
    text = "\n".join([
        f"Task: {case['task']}",
        "This run FAILED. A failure analysis located the root error at the step below; diagnose its cause.",
        "Earlier steps (oldest first):", *(hist or ["- (not shown)"]), "",
        f"ROOT STEP {case['step']}",
        f"Agent reasoning: {reason}",
        f"Action: {' ; '.join(case['root']['action'])[:400]}", "",
        *labels,
    ])
    return llm.ask(system, text, imgs, tag="diag")


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", default="claude_4.5_sonnet,gemini_2.5_pro")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mask-reasoning", action="store_true")
    ap.add_argument("--dataset", default="cuaerrorbench", choices=["cuaerrorbench", "osworld34"])
    ap.add_argument("--legacy-folder", action="store_true", help="use runs/rq1/<name> without a fingerprint")
    ap.add_argument("--prompt", default="diagnostician", choices=["diagnostician", "plain", "annotator"],
                    help="osworld34: the Diagnostician's prompt, or the plain labelling prompt (no N; RQ1 only)")
    ap.add_argument("--steps", default="a", choices=["a", "round2"],
                    help="osworld34: diagnose annotator A's first-round step, or the second-round step")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = Config.load(None, a.set)
    tax = Taxonomy(cfg.path(cfg.taxonomy))
    system = SYSTEM.format(taxonomy=tax.render()) + EXTRA
    if cfg.adr_version >= 3:   # ADR v3's Diagnostician prompt: v3 rules, extra rules (diag_rules), axes (use_axes)
        from .v3.diagnostician import DiagnosticianV3
        system = DiagnosticianV3(cfg, tax).system + EXTRA
    llm = LLM.for_role(cfg, "diag", cache_dir=str(ROOT / "runs" / "cache" / "rq1"))
    tasks = json.loads((DATA / "tasks.json").read_text())
    df = pd.read_parquet(DATA / "train.parquet")
    df = df[df.model.isin(a.models.split(","))]
    extra = {"dataset": a.dataset, "models": a.models, "limit": a.limit, "mask_reasoning": a.mask_reasoning}
    if a.steps != "a":         # only then, so the fingerprints of the earlier runs stay the same
        extra["steps"] = a.steps
    if a.prompt != "diagnostician":
        assert a.dataset == "osworld34" or a.prompt == "plain", "the annotator prompt is only for the 34 OSWorld cases"
        extra["prompt"] = a.prompt
    out_dir = rq1_folder(a, cfg, extra)

    if a.dataset == "osworld34":
        if a.prompt == "plain":
            return run_osworld34(a, cfg, llm, plain_system(tax), out_dir)
        if a.prompt == "annotator":
            return run_osworld34(a, cfg, llm, annotator_system(tax), out_dir)
        return run_osworld34(a, cfg, llm, system.replace(EXTRA, "") + OSW_NOTE, out_dir)
    if a.prompt == "plain":         # CUAErrorBench as the plain labeling task: our type only, no CUADebug category
        system = plain_system(tax)
    cases = []
    for model in a.models.split(","):
        zf, prefix = ZIPS[model]
        z = zipfile.ZipFile(DATA / zf)
        rows = df[df.model == model].to_dict("records")
        if a.limit:
            rows = rows[:a.limit]
        for row in rows:
            cases.append(load_case(z, prefix, row, model, tasks))
    from .runstore import CaseStore
    store = CaseStore(out_dir)
    total = len(cases)
    cases = [c for c in cases if not store.has(c["id"])]
    print(f"{total} cases, {total - len(cases)} already done, {len(cases)} to run", flush=True)

    results = []
    with ThreadPoolExecutor(a.threads) as ex:
        futs = {ex.submit(diagnose_case, c, llm, system, cfg, a.mask_reasoning): c for c in cases}
        for i, f in enumerate(as_completed(futs), 1):
            c = futs[f]
            try:
                d = f.result()
            except Exception as e:
                d = {"error": str(e)}
            ht = human_type(tax, c["tag"])
            pt = str(d.get("type", "")).upper()
            fam = str(d.get("cuadebug_family", "")).upper()[:1]
            rec = {"id": c["id"], "model": c["model"], "app": c["app"], "tag": c["tag"],
                   "human_type": ht, "human_family": c["tag"][:1], "confidence": c["confidence"],
                   "pred_type": pt, "pred_family": fam, "step": c["step"], "diag": d}
            results.append(rec)
            if "error" not in d:
                store.save(c["id"], rec)          # saved now; an error is retried on the next run
            if i % 10 == 0:
                sc = [r for r in results if r["human_type"]]
                acc = sum(r["pred_type"] == r["human_type"] for r in sc) / max(1, len(sc))
                print(f"[{i}/{len(cases)}] type acc so far {acc:.3f}", flush=True)
    results = store.all()                         # everything saved so far, including earlier sessions
    from .runstore import atomic_write
    atomic_write(out_dir / "results.json", json.dumps(results, indent=1, default=str))
    atomic_write(out_dir / "usage.json", json.dumps(usage_snapshot(), indent=1))
    sc = [r for r in results if r["human_type"]]
    acc = sum(r["pred_type"] == r["human_type"] for r in sc) / max(1, len(sc))
    fam = [r for r in results if r["human_family"] in FAMILY]
    facc = sum(r["pred_family"] == r["human_family"] for r in fam) / max(1, len(fam))
    _mark_finished(out_dir, len(results), total)
    fam_txt = "not asked (plain prompt)" if a.prompt == "plain" else f"{facc:.3f} (n={len(fam)})"
    print(f"DONE {a.name}: v4 type accuracy {acc:.3f} (n={len(sc)}), CUADebug family accuracy {fam_txt}")


# RQ1 as a plain labeling task: the VLM labels the failure like the human annotators did, without the
# Diagnostician's role in the live system (the Monitor, the fix fields, N). Only this offline evaluation uses it.
PLAIN = """You label failures of a GUI agent with a failure taxonomy, like a human annotator.
Every case is a run that FAILED its task (all cases come from OSWorld's failed runs), and the step shown is the step where it failed.
Read the evidence: the task, the earlier steps, the agent's reasoning and action at this step, and the screenshots (the red circle in the BEFORE image marks where the action landed, the zoomed crops show the area around it, and earlier frames show how the run got here).

{taxonomy}

There is no "no failure" answer: choose the type among I to VI that best explains why this step is the failure.

Reply with JSON only:
{{"evidence": "<2-3 sentences: what the agent intended, what happened, what the screen shows now>",
 "type": "I" | "II" | "III" | "IV" | "V" | "VI",
 "subtype": "<one subtype of that type>",
 "confidence": <0..1>}}"""


ANNOTATOR = """You label failures of a GUI agent with a failure taxonomy, like a human annotator.
Every case is a run that FAILED its task (all cases come from OSWorld's failed runs), and the step marked as the failure step is where it failed.
Read the whole trajectory first: every step's action and reasoning, also the steps after the failure step. Then look at the screenshots (the red circle in the BEFORE image marks where the action landed, the zoomed crops show the area around it, and earlier frames show how the run got here).

{taxonomy}

There is no "no failure" answer: choose the type among I to VI that best explains why the failure step is the failure.

WORKED EXAMPLES (filled in by the person who prepared the labelling, to show how it works; they are not among the cases you label):
{examples}

Reply with JSON only:
{{"note": "<one or two sentences: what went wrong at this step and what the agent should have done>",
 "type": "I" | "II" | "III" | "IV" | "V" | "VI",
 "subtype": "<one subtype of that type>",
 "confidence": <0..1>}}"""


def worked_examples() -> str:
    """The three calibration examples the annotators saw filled in (labels/round2_config.json); not scored."""
    cfg = json.loads((ROOT / "labels" / "round2_config.json").read_text())
    out = []
    for fid, ex in cfg["examples"].items():
        info = dict(l.split(": ", 1) for l in (ROOT / "data" / "failures" / fid / "info.txt").read_text().splitlines() if ": " in l)
        out.append(f"- Task: {info.get('task', '').strip()}\n  Failure step: {ex['failure_step']}. Type: {ex['type']} "
                   f"({ex['subtype']}). Why: {ex['notes']}")
    return "\n".join(out)


def annotator_system(tax) -> str:
    """The plain prompt plus what the annotators had: the guide's words and the three worked examples."""
    plain = plain_system(tax)
    taxonomy = plain[plain.index("FAILURE TYPES"):plain.index('\n\nThere is no "no failure" answer')]
    return ANNOTATOR.format(taxonomy=taxonomy, examples=worked_examples())


def plain_system(tax) -> str:
    """The taxonomy as the annotators used it for these cases: the types I to VI (no N), the decision procedure without
    Q0 (the question that leads to N) and the tie-breakers."""
    d = tax.data
    out = ["FAILURE TYPES (choose exactly one id):"]
    out += [f"- {t['id']} {t['name']}: {t['definition']} Subtypes: {', '.join(t['subtypes'])}." for t in d["types"] if t["id"] != "N"]
    out += ["", "DECISION PROCEDURE (answer in order; the first 'yes' decides):"]
    out += [f"  {q}" for q in d["procedure"] if not q.startswith("Q0")]
    out += ["TIE-BREAKERS:"] + [f"  - {r}" for r in d["tie_breakers"]]
    return PLAIN.format(taxonomy="\n".join(out))


OSW_NOTE = """
Here the failure is confirmed by a human analysis, so do not answer N: choose the type among I-VI that best
explains why this step is the root error (skip Q0)."""


def run_osworld34(a, cfg, llm, system, out_dir):
    from .analyze import kappa
    from .runstore import CaseStore, atomic_write
    store = CaseStore(out_dir)
    cases = osworld34_cases(steps=getattr(a, "steps", "a"))
    full_run = getattr(a, "prompt", "") == "annotator"
    total = len(cases)
    cases = [c for c in cases if not store.has(c["tag"])]
    print(f"{total} cases, {total - len(cases)} already done, {len(cases)} to run", flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        futs = {ex.submit(diagnose_case, c, llm, system, cfg, a.mask_reasoning, full_run): c for c in cases}
        for f in as_completed(futs):
            c = futs[f]
            try:
                d = f.result()
            except Exception as e:
                d = {"error": str(e)}
            if "error" not in d:
                store.save(c["tag"], {"id": c["tag"], "step": c["step"], "gold_a": c["gold_a"],
                                      "gold_b": c["gold_b"], "pred_type": str(d.get("type", "")).upper(), "diag": d,
                                      **{k: c[k] for k in ("gold_r2", "r2_types", "r2_fixed") if k in c}})
    results = store.all()
    atomic_write(out_dir / "results.json", json.dumps(results, indent=1, default=str))
    sc = [r for r in results if r["gold_a"] and "error" not in r["diag"]]
    both = [r for r in sc if r["gold_b"]]
    acc_a = sum(r["pred_type"] == r["gold_a"] for r in sc) / max(1, len(sc))
    acc_b = sum(r["pred_type"] == r["gold_b"] for r in both) / max(1, len(both))
    hum = sum(r["gold_a"] == r["gold_b"] for r in both) / max(1, len(both))
    summ = {"n": len(sc), "model_vs_A": acc_a, "model_vs_B": acc_b, "A_vs_B": hum,
            "kappa_model_A": kappa([r["gold_a"] for r in sc], [r["pred_type"] for r in sc]),
            "kappa_model_B": kappa([r["gold_b"] for r in both], [r["pred_type"] for r in both]),
            "kappa_A_B": kappa([r["gold_a"] for r in both], [r["gold_b"] for r in both])}
    atomic_write(out_dir / "summary.json", json.dumps(summ, indent=1))
    _mark_finished(out_dir, len(results), total)
    print("DONE", json.dumps(summ))


if __name__ == "__main__":
    main()
