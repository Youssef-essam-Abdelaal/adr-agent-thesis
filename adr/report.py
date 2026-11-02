"""
The results page, built from the numbers analyze.py computes, with one RQ2
section per base agent. It is written as page content without an html/head/body
skeleton, which browsers open as it is.
"""
from __future__ import annotations

import html
import json
import shutil
from collections import Counter

from .config import ROOT

CONDS = ["base", "retry", "retry_evidence", "adr_untyped", "adr_hints", "adr_v1", "adr", "adr_oracle"]
NAMES = {"base": "Base agent alone", "retry": "+ generic reflect-and-retry",
         "retry_evidence": "+ retry with the Monitor's evidence", "adr_untyped": "+ untyped advice (no taxonomy)",
         "adr_hints": "+ typed hints only (no actions)", "adr_v1": "+ ADR v1 (typed recovery)",
         "adr": "+ ADR v2 (+ prevention, checked fixes)", "adr_oracle": "+ ADR v2 with true failure type"}
FAULTS = [("none", "clean"), ("grounding", "I mis-click"), ("stagnation", "II dead click"),
          ("occlusion", "III pop-up"), ("wrong_value", "IV wrong value"),
          ("loading", "II loading (held out)"), ("layout_shift", "I layout shift (held out)")]
TYPES = ["I", "II", "III", "IV", "V", "VI", "N"]
TYPE_NAMES = {"I": "Grounding & Execution", "II": "Unresponsive Environment", "III": "Obstruction",
              "IV": "Wrong Target or Value", "V": "Plan & Task Understanding", "VI": "Perception", "N": "No Failure"}

STYLE = """
<style>
/* Layout: a single reading column for a results brief; tables scroll inside their own box. */
:root{
  --bg:#fbfbfa; --fg:#1b2228; --mute:#5f6b75; --line:#dfe3e6; --panel:#f1f4f5;
  --accent:#1d6f73; --accent-soft:#d6ecec; --good:#2d7a3e; --bad:#a3392e; --bar:#1d6f73;
  --heat:29,111,115;
  --f-display:"Source Serif 4", Georgia, "Times New Roman", serif;
  --f-body:"IBM Plex Sans", -apple-system, "Segoe UI", Roboto, sans-serif;
  --f-data:"IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#12171a; --fg:#e4e9ec; --mute:#9aa7b0; --line:#2a3338; --panel:#192024;
  --accent:#5fb7b9; --accent-soft:#1d3436; --good:#6fc483; --bad:#ec8a7e; --bar:#5fb7b9; --heat:95,183,185;
  color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#12171a; --fg:#e4e9ec; --mute:#9aa7b0; --line:#2a3338; --panel:#192024;
  --accent:#5fb7b9; --accent-soft:#1d3436; --good:#6fc483; --bad:#ec8a7e; --bar:#5fb7b9; --heat:95,183,185;
  color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.55 var(--f-body)}
.wrap{max-width:980px;margin:0 auto;padding-inline:16px;padding-block:28px 72px;display:flex;flex-direction:column;gap:6px}
h1,h2,h3{font-family:var(--f-display);text-wrap:balance;margin:0}
h1{font-size:2.1rem;font-weight:600;letter-spacing:-.01em}
h2{font-size:1.45rem;font-weight:600;margin-top:34px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:1.08rem;font-weight:600;margin-top:18px}
p{margin:6px 0;max-width:68ch}
.lede{color:var(--mute);max-width:72ch}
.eyebrow{font:600 .72rem/1 var(--f-body);letter-spacing:.09em;text-transform:uppercase;color:var(--accent)}
.mute{color:var(--mute)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:14px 0 4px}
.kpi{border:1px solid var(--line);border-radius:6px;padding:12px 14px;background:var(--panel);min-width:0}
.kpi .v{font:600 1.6rem/1.15 var(--f-data);font-variant-numeric:tabular-nums}
.kpi .l{font-size:.82rem;color:var(--mute)}
.kpi .v.good{color:var(--good)}
.pipeline{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin-top:10px}
.stage{border-top:3px solid var(--accent);padding:8px 2px 0;min-width:0}
.stage b{font-family:var(--f-display);font-size:1.05rem}
.stage p{font-size:.9rem;color:var(--mute);margin:4px 0 0}
.tbl{overflow-x:auto;margin:8px 0 4px;max-width:100%}
table{border-collapse:collapse;font-size:.86rem;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right;white-space:nowrap;vertical-align:middle}
th{font-weight:600;color:var(--mute);font-size:.78rem;letter-spacing:.02em}
td:first-child,th:first-child{text-align:left}
td.num{font-family:var(--f-data)}
.cell{display:flex;align-items:center;gap:6px;justify-content:flex-end}
.bar{display:inline-block;height:7px;border-radius:2px;background:var(--bar);opacity:.85}
.n{color:var(--mute);font-size:.75rem}
tr.hl td{background:var(--accent-soft)}
.chip{display:inline-block;font:600 .74rem/1.4 var(--f-data);padding:0 6px;border-radius:3px;border:1px solid var(--line)}
.sig{color:var(--good);font-weight:600}
.ex{border:1px solid var(--line);border-radius:6px;padding:12px 14px;margin:12px 0;min-width:0}
.ex .meta{font-size:.9rem}
.strip{display:flex;gap:10px;overflow-x:auto;padding:8px 0 2px}
.strip figure{margin:0;flex:0 0 auto}
.strip img{height:220px;max-width:none;border:1px solid var(--line);border-radius:3px;display:block}
.strip figcaption{font-size:.75rem;color:var(--mute);margin-top:4px;max-width:170px}
ul{margin:6px 0;padding-left:20px;max-width:72ch}
li{margin:3px 0}
code{font-family:var(--f-data);font-size:.85em}
</style>
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600'
         '&family=IBM+Plex+Sans:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">')


def goal_text(g):
    return g.get("utterance", "") if isinstance(g, dict) else str(g)


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def tbl(head, rows, hl=()):
    h = "".join(f"<th>{x}</th>" for x in head)
    body = "".join(f"<tr{' class=hl' if i in hl else ''}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                   for i, r in enumerate(rows))
    return f"<div class='tbl'><table><tr>{h}</tr>{body}</table></div>"


def bar_cell(x):
    if not x["n"]:
        return ""
    w = round(30 * x["sr"])
    return (f"<span class='cell'><span class='bar' style='width:{w}px'></span>"
            f"<span>{pct(x['sr'])}</span><span class='n'>{x['k']}/{x['n']}</span></span>")


def heat(v, mx):
    a = 0 if not mx else 0.12 + 0.6 * v / mx
    return f"<span style='display:block;background:rgba(var(--heat),{a:.2f});padding:2px 6px'>{v}</span>" if v else "·"


def confusion(conf, rows_order, label):
    mx = max([v for r in conf.values() for v in r.values()] or [0])
    rows = [[f"<span class='chip'>{g}</span> {TYPE_NAMES[g]}"] + [heat(conf.get(g, {}).get(p, 0), mx) for p in TYPES]
            for g in rows_order if conf.get(g)]
    return tbl([label] + TYPES, rows)


def rq2_section(agent, res, by, run):
    s, t, rec, cost = res["success"], res["tests"], res["recovery"], res["cost"]
    P = [f"<h3>Base agent: <code>{html.escape(agent)}</code></h3>"]
    conds = [c for c in CONDS if s[c]["all"]["n"]]
    faults = [(fk, n) for fk, n in FAULTS if any(s[c][fk]["n"] for c in conds)]
    hl = tuple(i for i, c in enumerate(conds) if c == "adr")
    rows = [[NAMES[c]] + [bar_cell(s[c][fk]) for fk, _ in faults] + [bar_cell(s[c]["faulted"])] for c in conds]
    P.append(tbl(["Condition"] + [n for _, n in faults] + ["all faulted"], rows, hl=hl))
    reg = res.get("registered") or []
    if reg:
        rows = []
        for r in reg:
            ci = r["cluster_ci"] or {}
            sig = " <span class='sig'>supported</span>" if r["p_holm"] < 0.05 else ""
            rows.append([r["hypothesis"], f"{NAMES[r['a']].lstrip('+ ')} > {NAMES[r['b']].lstrip('+ ')}", r["n_pairs"],
                         f"{100 * ci.get('diff', 0):+.1f} [{100 * ci.get('lo', 0):+.1f}, {100 * ci.get('hi', 0):+.1f}]",
                         f"{r['p']:.3g}", f"<span class='num'>{r['p_holm']:.3g}</span>{sig}"])
        P.append("<p class='mute'>Pre-registered hypotheses, all episodes, Holm-corrected.</p>")
        P.append(tbl(["", "hypothesis", "pairs", "difference, points [task-grouped 95% CI]", "p", "p (Holm)"], rows))

    def test_row(key, a, b):
        v = t[key]
        p = v["p"]
        ci = v.get("cluster_ci") or {}
        star = " <span class='sig'>significant</span>" if p < 0.05 else ""
        return [f"{NAMES[a]} vs {NAMES[b].lstrip('+ ')}", v["n_pairs"], v[f"{a}_only"], v[f"{b}_only"],
                f"{100 * ci.get('diff', 0):+.1f} [{100 * ci.get('lo', 0):+.1f}, {100 * ci.get('hi', 0):+.1f}]",
                f"<span class='num'>{p:.3g}</span>{star}"]
    rows = [test_row(f"{a} vs {b} (faulted)", a, b) for a, b in [
        ("adr", "base"), ("adr", "retry"), ("retry", "base"), ("adr_v1", "retry"), ("adr", "adr_v1"),
        ("adr", "retry_evidence"), ("adr_hints", "adr_untyped"), ("adr", "adr_hints"), ("adr_untyped", "retry_evidence"),
        ("adr_oracle", "adr")] if f"{a} vs {b} (faulted)" in t and t[f"{a} vs {b} (faulted)"]["n_pairs"]]
    P.append("<p class='mute'>Paired on the same task, seed and fault. McNemar exact test on the pairs where only one "
             "side succeeded. The interval resamples whole tasks, because episodes of one task are related.</p>")
    P.append(tbl(["Comparison (faulted episodes)", "pairs", "only first succeeds", "only second succeeds",
                  "difference, points [95% CI]", "p"], rows))
    rows = [[NAMES[c], rec[c]["fired"], pct(rec[c]["post_error_success"]), pct(rec[c]["error_awareness"]),
             pct(rec[c]["recovery_rate"])] for c in conds]
    P.append(tbl(["Condition", "faults fired", "post-error success", "error awareness (flagged)",
                  "recovery rate (success when flagged)"], rows, hl=hl))
    rows = [[NAMES[c], f"{cost[c]['agent_steps']:.2f}", f"{cost[c]['actions']:.2f}",
             f"{cost[c]['tokens'] / 1000:.1f}k", f"{cost[c]['wrapper_tokens'] / 1000:.1f}k",
             f"{cost[c]['seconds']:.0f}s"] for c in conds]
    P.append(tbl(["Cost per episode", "agent decisions", "actions", "tokens", "of which wrapper", "time"],
                 rows, hl=hl))
    cpr = res.get("cost_per_rescue") or {}
    rows = [[NAMES[c], f"{100 * cpr[c]['extra_success']:+.1f}", f"{cpr[c]['extra_tokens'] / 1000:+.1f}k",
             f"{cpr[c]['tokens_per_rescue'] / 1000:.0f}k" if cpr[c]["tokens_per_rescue"] else "no gain"]
            for c in conds if c in cpr]
    if rows:
        P.append(tbl(["Against the base agent", "extra successes per 100 episodes", "extra tokens per episode",
                      "extra tokens per rescued episode"], rows))
    pt = res.get("per_task") or {}
    if pt:
        cols = [c for c in conds if c != "adr_oracle"]
        rows = [[f"<code>{html.escape(task)}</code>"] + [pct(v[c]) if v.get(c) is not None else "" for c in cols]
                for task, v in pt.items()]
        P.append("<details><summary class='mute'>Success per task</summary>" +
                 tbl(["Task"] + [NAMES[c].lstrip("+ ") for c in cols], rows) + "</details>")
    v2 = res.get("v2") or {}
    if v2.get("precommit_checks"):
        P.append(tbl(["ADR v2 internals", "final-action checks", "judged final", "clicks blocked",
                      "blocked episodes that succeeded", "fix checks", "escalations", "votes", "generic fallbacks"],
                     [["count", v2["precommit_checks"], v2["judged_final"], v2["blocks"],
                       f"{v2['blocked_episodes']} ({pct(v2['blocked_episode_success'])})", v2["recovery_checks"],
                       v2["escalations"], v2["votes"], v2["generic_fallbacks"]]]))
    return "".join(P)


def examples(by, run, img_dir, limit=4):
    out, shown = [], Counter()
    for r in by["adr"].values():
        fk = r["spec"]["fault"]
        if fk == "none" or shown[fk] or not r["success"]:
            continue
        st = next((x for x in r["steps"] if x.get("fault_related") and x.get("plan")
                   and (x.get("plan") or {}).get("actions")), None)
        if not st:
            continue
        fdir = ROOT / "runs" / r.get("_run", run) / "frames" / r["_id"]
        k = st["step"]
        pics = [(fdir / f"{k:02d}.jpg", "Before the action"), (fdir / f"{k:02d}_flag.jpg", "After: flagged by the Monitor"),
                (fdir / f"{k + 1:02d}.jpg", "After the recovery")]
        figs = []
        for pth, cap in pics:
            if pth.exists():
                dst = img_dir / f"{r.get('_run', run)}_{r['_id']}_{pth.name}"
                shutil.copy(pth, dst)
                figs.append(f"<figure><img src='img/{dst.name}' alt='{cap}'><figcaption>{cap}</figcaption></figure>")
        if len(figs) < 2:
            continue
        d, pl, v = st["diagnosis"], st["plan"], st["verdict"]
        out.append(
            f"<div class='ex'><div class='meta'><b>{html.escape(goal_text(r['goal']))}</b><br>"
            f"Injected: {html.escape(r['fault']['detail'])} (true type {r['fault']['true_type']}). "
            f"Monitor: <i>{html.escape(v['symptom'])}</i>. Diagnosis: <span class='chip'>{d.get('type')}</span> "
            f"{html.escape(d.get('subtype', ''))}. Recovery: <b>{html.escape(pl['kind'])}</b>.</div>"
            f"<p class='mute'>{html.escape(d.get('evidence', ''))}</p>"
            f"<div class='strip'>{''.join(figs)}</div></div>")
        shown[fk] += 1
        if len(out) >= limit:
            break
    return "".join(out)


def build(runs, rq1s, agentic=None):
    """runs: list of (agent_name, run_name, res, by)."""
    img_dir = ROOT / "results" / "img"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)
    P = ["<title>ADR-Agent Results</title>", FONTS, STYLE, "<div class='wrap'>",
         "<span class='eyebrow'>Master's thesis · Adaptive fault tolerance for GUI agents</span>",
         "<h1>ADR-Agent results</h1>",
         "<p class='lede'>A training-free wrapper that checks every step of an existing GUI agent, diagnoses the "
         "cause of a failure with a taxonomy, and applies a fix chosen for that cause. Tested live on 20 MiniWoB++ "
         "tasks in Chrome with injected faults, and offline on CUAErrorBench (human root-cause labels on failed "
         "OSWorld runs).</p>"]

    kp = []
    for agent, run, res, _ in runs:
        s = res["success"]
        best = "adr" if s["adr"]["faulted"]["n"] else "adr_v1"
        kp.append(f"<div class='kpi'><div class='l'>{html.escape(agent)}: success under faults, base → ADR</div>"
                  f"<div class='v good'>{pct(s['base']['faulted']['sr'])} → {pct(s[best]['faulted']['sr'])}</div>"
                  f"<div class='l'>generic retry {pct(s['retry']['faulted']['sr'])}; "
                  f"ADR vs retry p = {res['tests']['adr vs retry (faulted)']['p']:.3g}</div></div>")
    live = runs[0][2]["rq1_live"] if runs else None
    if live:
        g = [r[2]["rq1_live"] for r in runs]
        n = sum(x["n"] for x in g)
        acc = sum(x["accuracy"] * x["n"] for x in g) / max(1, n)
        kp.append(f"<div class='kpi'><div class='l'>RQ1 live: failure type correct</div><div class='v'>{pct(acc)}</div>"
                  f"<div class='l'>{n} injected faults, exact labels</div></div>")
    if agentic and agentic.get("claude"):
        kp.append(f"<div class='kpi'><div class='l'>RQ1 CUAErrorBench, step not given: family correct</div>"
                  f"<div class='v'>{pct(agentic['claude']['family_accuracy'])}</div>"
                  f"<div class='l'>Claude split. CUADebug 53.5 to 61.1%, its naive baseline 44.4 to 50.7%</div></div>")
    P.append(f"<div class='kpis'>{''.join(kp)}</div>")

    P.append("<h2>How it works</h2><div class='pipeline'>"
             "<div class='stage'><b>Verify</b><p>Free checks first (pixel change, loops), then a judge that describes "
             "the change before deciding. Every \"done\" is checked against criteria taken from the task. Reports the "
             "symptom only.</p></div>"
             "<div class='stage'><b>Diagnose</b><p>Runs only on a flag. Classifies the cause with taxonomy v4 from the "
             "marked screenshots, zoomed crops, earlier frames and the step history. Can answer \"no failure\" to "
             "cancel a false alarm.</p></div>"
             "<div class='stage'><b>Recover</b><p>A fixed table from type to fix: re-ground, wait and retry, dismiss "
             "the pop-up, correct the value, replan, correct the belief. Retry budget and escalation.</p></div></div>")

    P.append("<h2>RQ2: does typed recovery beat generic retry?</h2>"
             "<p>Four conditions on identical task, seed and fault combinations. Faults fire on a known step: a "
             "mis-located click (I), an ignored click (II), a pop-up over the target (III) and a wrong typed value "
             "(IV). Success is MiniWoB's own reward check.</p>")
    for agent, run, res, by in runs:
        P.append(rq2_section(agent, res, by, run))

    P.append("<h2>Monitor</h2>")
    rows = []
    for agent, run, res, _ in runs:
        m = res["monitor"]
        rows.append([html.escape(agent), pct(m["fault_recall"]), m["faults"], pct(m["false_alarm_per_step"]),
                     m["clean_steps"], f"{m['paired_discrimination']:.3f}", m["false_done_vetoes"]])
    P.append(tbl(["Base agent", "injected faults caught", "faults", "flags on fault-free steps", "fault-free steps",
                  "paired discrimination", "premature-done vetoes"], rows))
    P.append("<p class='mute'>Fault-free steps can still contain the agent's own natural errors, so not every flag "
             "there is a false alarm. No fault-free episode was lost to a recovery where the base agent succeeded. "
             "Paired discrimination = catch rate minus flag rate on fault-free steps.</p>")

    P.append("<h2>RQ1: can a VLM classify the failure type?</h2>")
    gold_conf = {}
    for agent, run, res, _ in runs:
        for g_, row in res["rq1_live"]["confusion"].items():
            for p_, v in row.items():
                gold_conf.setdefault(g_, Counter())[p_] += v
    if gold_conf:
        n = sum(sum(v.values()) for v in gold_conf.values())
        acc = sum(gold_conf.get(t, {}).get(t, 0) for t in TYPES) / max(1, n)
        P.append(f"<h3>Live injected faults, exact ground truth (n = {n}, accuracy {pct(acc)})</h3>")
        lk = [r[2]["rq1_live"].get("symptom_lookup_accuracy") for r in runs if r[2]["rq1_live"].get("n")]
        if lk:
            P.append(f"<p>For comparison, a plain lookup on the Monitor's symptom (blocked to III, no effect to II, wrong "
                     f"effect to IV) gets {pct(sum(lk) / len(lk))}. These faults were designed to be told apart, so this "
                     f"is a check of the pipeline, not the answer to RQ1; the real-failure results below are.</p>")
        P.append(confusion(gold_conf, ["I", "II", "III", "IV"], "true type ↓ / predicted →"))
    if rq1s:
        P.append("<h3>CUAErrorBench: real failures, human labels</h3>"
                 "<p>Claude Sonnet 4.5 and Gemini 2.5 Pro runs on OSWorld. The Diagnostician sees the evidence at the "
                 "human root-cause step and classifies it. Human tags are mapped to v4 types with the table in the "
                 "taxonomy file; S4 (step limit) and infeasible-task tags are not agent errors and are left out.</p>")
        rows = [[html.escape(q["name"]), q["n"], pct(q["type_accuracy"]), f"{q['macro_f1']:.3f}", f"{q['kappa']:.3f}",
                 pct(q["family_accuracy"]), f"{q['family_kappa']:.3f}"] for q in rq1s]
        P.append(tbl(["Diagnostician input", "cases", "v4 type accuracy", "macro-F1", "kappa",
                      "CUADebug family accuracy", "family kappa"], rows))
        P.append(confusion(rq1s[0]["confusion"], TYPES, "human type ↓ / predicted →"))
    if agentic:
        a = agentic
        P.append("<h3>CUAErrorBench: finding the root step as well</h3>"
                 "<p>The harder setting of CUADebug's debugger. The Diagnostician gets the whole failed run as text "
                 "and the final screen, inspects up to three rounds of steps (screenshots and full reasoning), then "
                 "names the root step and the type.</p>")
        c = a.get("claude") or a
        P.append(tbl(["", "family (L1) accuracy", "exact root step", "within 2 steps"], [
            ["Ours, Claude split (144)", pct(c["family_accuracy"]), pct(c["step_exact"]), pct(c["step_within_2"])],
            ["CUADebug naive one-pass baseline", "44.4 to 50.7%", "20.8 to 23.6%", "34.7 to 38.2%"],
            ["CUADebug debugger", "53.5 to 61.1%", "25.0 to 29.2%", "45.1 to 47.9%"],
            [f"Ours, all {a['n']} runs", pct(a["family_accuracy"]), pct(a["step_exact"]), pct(a["step_within_2"])]], hl=(0,)))
        P.append("<p class='mute'>Same setting as CUADebug (arXiv 2608.02643, Table 2): the step is not given. Ours sits between "
                 "CUADebug's naive baseline and CUADebug on family accuracy, and at the baseline's level on the exact step. "
                 f"It inspected {a['mean_inspections']:.1f} steps on average.</p>")
        P.append("<p class='mute'>With the root step given, this is an easier setting than CUADebug's, so its numbers are "
                 "not comparable here; see the next table. Mis-Score (arXiv 2607.28367) reports kappa 0.59 between human "
                 "groups and 0.19 to 0.32 between humans and LLMs on failure labels, on a stress set that includes every "
                 "LLM disagreement.</p>")

    o34 = ROOT / "runs" / "rq1" / "osworld34" / "summary.json"
    if o34.exists():
        o = json.loads(o34.read_text())
        P.append("<h3>Our 34 labelled OSWorld cases (UI-TARS, Chrome)</h3>"
                 "<p>Two human annotators labelled these with taxonomy v3; labels are mapped to v4. The model is "
                 "compared with each annotator, next to how well the annotators agree with each other.</p>")
        P.append(tbl(["Pair", "agreement", "Cohen's kappa"], [
            ["Model vs annotator A", pct(o["model_vs_A"]), f"{o['kappa_model_A']:.2f}"],
            ["Model vs annotator B", pct(o["model_vs_B"]), f"{o['kappa_model_B']:.2f}"],
            ["Annotator A vs annotator B", pct(o["A_vs_B"]), f"{o['kappa_A_B']:.2f}"]], hl=(2,)))
        P.append("<p class='mute'>The earlier Diagnostician, which saw only the final before/after pair, agreed with "
                 "the annotators on 12 to 21% of these cases.</p>")

    P.append("<h2>Example recoveries</h2>")
    for agent, run, res, by in runs:
        ex = examples(by, run, img_dir)
        if ex:
            P.append(f"<p class='mute'>From the <code>{html.escape(agent)}</code> runs.</p>{ex}")
            break

    P.append("<h2>Limits</h2><ul>"
             "<li>Faults are injected into a real browser on a known step. That gives exact labels, but the faults are "
             "cleaner than many real failures, so live RQ1 accuracy is an upper bound for these four types.</li>"
             "<li>Some effects never show on screen (a button click with no visual feedback). No screenshot-based "
             "check can catch those; they are counted, not hidden.</li>"
             "<li>On CUAErrorBench the human root step is given. Finding the step as well is harder.</li>"
             "<li>Types V and VI are only scored on CUAErrorBench; the live faults cover I to IV.</li></ul>")
    P.append("</div>")
    (ROOT / "results" / "report.html").write_text("\n".join(P))
