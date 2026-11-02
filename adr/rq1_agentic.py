"""
RQ1, harder setting: the Diagnostician must FIND the root step, not only classify it.

Like CUADebug's debugger, it starts from the whole failed run as text (each
step's action and the start of the agent's reasoning) plus the final screen,
and may inspect a few steps (their screenshots and full reasoning) before
answering. Scored on root-step localization and on type and category accuracy
with no step given.

    python -m adr.rq1_agentic --name agentic --threads 6
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from .config import Config
from .llm import LLM, parse_json, to_png_bytes, usage_snapshot, USAGE
from .rq1_offline import DATA, EXTRA, FAMILY, POINT, ZIPS, _steps, human_type
from .taxonomy import Taxonomy
from .vision import mark_click

SYSTEM = """You are the Diagnostician, debugging a FAILED run of a computer-use agent.
Find the ROOT ERROR: the earliest step that made the run fail. Then classify its cause with the taxonomy.
You get the whole run as text (each step's action and the start of the agent's reasoning) and the final screen.
Before answering you may inspect steps to see their before/after screenshots and full reasoning.

{taxonomy}
{extra}

Reply with JSON only, either
{{"action": "inspect", "steps": [<one or two step numbers>], "why": "<short>"}}
or, when you are ready (you must answer after at most {k} inspections),
{{"action": "answer", "root_step": <int>, "type": "I" | "II" | "III" | "IV" | "V" | "VI",
  "cuadebug_family": "P" | "G" | "R" | "S", "evidence": "<2-3 sentences>", "confidence": <0..1>}}
The run failed, so N is not allowed."""

SCALE = 0.5   # 1920x1080 -> 960x540 for inspection images


def _url(img: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(to_png_bytes(img)).decode()


class Case:
    def __init__(self, model: str, row: dict, tasks: dict):
        zf, prefix = ZIPS[model]
        self.z = zipfile.ZipFile(DATA / zf)
        self.base = prefix + row["task_dir"] + "/"
        rows = [json.loads(l) for l in self.z.read(self.base + "traj.jsonl").decode().splitlines() if l.strip()]
        self.steps = _steps(rows, model)
        self.model, self.row = model, row
        self.task = tasks.get(row["task_dir"]) or ""

    def shot(self, k: int):
        s = self.steps.get(k)
        if not s or not s["shot"]:
            return None
        try:
            im = Image.open(io.BytesIO(self.z.read(self.base + s["shot"]))).convert("RGB")
        except KeyError:
            return None
        return im.resize((int(im.size[0] * SCALE), int(im.size[1] * SCALE)))

    def outline(self) -> str:
        lines = []
        for k in sorted(self.steps):
            s = self.steps[k]
            act = " ; ".join(s["action"])[:150]
            why = " ".join(s["reason"]).replace("\n", " ")[:160]
            lines.append(f"{k}. {act} | {why}")
        return "\n".join(lines)

    def detail(self, k: int):
        s = self.steps.get(k)
        if not s:
            return [{"type": "text", "text": f"Step {k} does not exist."}]
        m = POINT.search(" ".join(s["action"]))
        before, after = self.shot(k - 1), self.shot(k)
        parts = [{"type": "text", "text": (f"STEP {k}\nAction: {' ; '.join(s['action'])[:400]}\n"
                                           f"Reasoning: {' '.join(s['reason'])[:1500]}")}]
        if before is not None:
            if m:
                before = mark_click(before, float(m.group(1)) * SCALE, float(m.group(2)) * SCALE)
            parts += [{"type": "text", "text": f"Step {k}, BEFORE (red circle = where the action landed):"},
                      {"type": "image_url", "image_url": {"url": _url(before), "detail": "high"}}]
        if after is not None:
            parts += [{"type": "text", "text": f"Step {k}, AFTER:"},
                      {"type": "image_url", "image_url": {"url": _url(after), "detail": "high"}}]
        return parts


def run_case(case: Case, llm: LLM, system: str, k: int) -> dict:
    final = case.shot(max(case.steps))
    first = [{"type": "text", "text": f"Task: {case.task}\n\nThe run (step. action | reasoning):\n{case.outline()}\n\n"
                                      f"Final screen:"}]
    if final is not None:
        first.append({"type": "image_url", "image_url": {"url": _url(final), "detail": "high"}})
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": first}]
    inspected = []
    for rnd in range(k + 2):
        out = chat(llm, msgs)
        if out.get("action") == "answer":
            out["inspected"] = inspected
            return out
        if rnd >= k:
            msgs += [{"role": "assistant", "content": json.dumps(out)},
                     {"role": "user", "content": "No more inspections. Answer now with the answer JSON."}]
            continue
        want = [int(x) for x in (out.get("steps") or []) if str(x).lstrip("-").isdigit()][:2]
        inspected += want
        parts = []
        for s in want:
            parts += case.detail(s)
        msgs += [{"role": "assistant", "content": json.dumps(out)},
                 {"role": "user", "content": parts or [{"type": "text", "text": "Name step numbers to inspect."}]}]
    return {"action": "answer", "error": "no answer", "inspected": inspected}


def chat(llm: LLM, msgs) -> dict:
    kwargs = dict(model=llm.model, messages=msgs, response_format={"type": "json_object"},
                  max_completion_tokens=llm.max_tokens)
    if llm.effort:
        kwargs["reasoning_effort"] = llm.effort
    r = llm._create(kwargs)
    u = USAGE["diag"]
    u["calls"] += 1
    if r.usage:
        u["prompt_tokens"] += r.usage.prompt_tokens or 0
        u["completion_tokens"] += r.usage.completion_tokens or 0
    return parse_json(r.choices[0].message.content or "")


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="agentic")
    ap.add_argument("--models", default="claude_4.5_sonnet,gemini_2.5_pro")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--inspections", type=int, default=3)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--legacy-folder", action="store_true", help="use runs/rq1/<name> without a fingerprint")
    a = ap.parse_args(argv)
    cfg = Config.load(None, a.set)
    tax = Taxonomy(cfg.path(cfg.taxonomy))
    system = SYSTEM.format(taxonomy=tax.render(), extra=EXTRA.split("Here the failure")[0], k=a.inspections)
    llm = LLM(cfg.diag_model, cfg.diag_effort, max_tokens=8000)
    tasks = json.loads((DATA / "tasks.json").read_text())
    df = pd.read_parquet(DATA / "train.parquet")
    from .rq1_offline import rq1_folder
    from .runstore import CaseStore, atomic_write
    out_dir = rq1_folder(a, cfg, {"agentic": True, "models": a.models, "limit": a.limit,
                                  "inspections": a.inspections})
    store = CaseStore(out_dir)
    rows = []
    for m in a.models.split(","):
        r = df[df.model == m].to_dict("records")
        rows += [(m, x) for x in (r[:a.limit] if a.limit else r)]
    total = len(rows)
    rows = [(m, x) for m, x in rows if not store.has(f"{m}/{x['task_dir']}")]
    print(f"{total} cases, {total - len(rows)} already done, {len(rows)} to run", flush=True)

    def work(item):
        m, row = item
        c = Case(m, row, tasks)
        return m, row, run_case(c, llm, system, a.inspections)

    results = []
    with ThreadPoolExecutor(a.threads) as ex:
        futs = [ex.submit(work, it) for it in rows]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                m, row, d = f.result()
            except Exception as e:
                print("error", e, flush=True)
                continue
            gold = int(row["root_error_step"])
            pred = d.get("root_step")
            try:
                pred = int(pred)
            except (TypeError, ValueError):
                pred = None
            rec = ({"id": f"{m}/{row['task_dir']}", "model": m, "tag": row["taxonomy_tag"],
                            "human_type": human_type(tax, row["taxonomy_tag"]), "human_family": row["taxonomy_tag"][:1],
                            "gold_step": gold, "pred_step": pred, "num_steps": int(row["num_steps"]),
                            "pred_type": str(d.get("type", "")).upper(),
                            "pred_family": str(d.get("cuadebug_family", "")).upper()[:1], "diag": d})
            results.append(rec)
            if "error" not in d and d.get("action") == "answer":
                store.save(rec["id"], rec)        # saved now; a failed case is retried on the next run
            if i % 10 == 0:
                ex_ = sum(r["pred_step"] == r["gold_step"] for r in results) / len(results)
                print(f"[{i}/{len(rows)}] step exact so far {ex_:.3f}", flush=True)
    results = store.all()
    atomic_write(out_dir / "results.json", json.dumps(results, indent=1, default=str))
    atomic_write(out_dir / "usage.json", json.dumps(usage_snapshot(), indent=1))
    summary = summarize(results)
    atomic_write(out_dir / "summary.json", json.dumps(summary, indent=1))
    from .rq1_offline import _mark_finished
    _mark_finished(out_dir, len(results), total)
    print("DONE", json.dumps(summary))


def summarize(results):
    n = max(1, len(results))
    sc = [r for r in results if r["human_type"]]
    fam = [r for r in results if r["human_family"] in FAMILY]
    within = lambda r, w: r["pred_step"] is not None and abs(r["pred_step"] - r["gold_step"]) <= w
    return {
        "n": len(results),
        "step_exact": sum(r["pred_step"] == r["gold_step"] for r in results) / n,
        "step_within_2": sum(within(r, 2) for r in results) / n,
        "step_within_5": sum(within(r, 5) for r in results) / n,
        "type_accuracy": sum(r["pred_type"] == r["human_type"] for r in sc) / max(1, len(sc)),
        "family_accuracy": sum(r["pred_family"] == r["human_family"] for r in fam) / max(1, len(fam)),
        "family_and_step_exact": sum(r["pred_family"] == r["human_family"] and r["pred_step"] == r["gold_step"]
                                     for r in fam) / max(1, len(fam)),
        "mean_inspections": sum(len(r["diag"].get("inspected", [])) for r in results) / n,
    }


if __name__ == "__main__":
    main()
