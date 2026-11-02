"""
Turn saved episodes into the thesis numbers and a results page.

    python -m adr.analyze --runs <run folder> --rq1 full,twoframe

RQ2: success rate per condition and fault, paired McNemar tests, recovery
rate, post-error success and error awareness, Monitor detection vs
false alarms (paired discrimination), cost and step overhead.
RQ1: Diagnostician type accuracy on live injected faults (exact ground truth)
and on CUAErrorBench human labels, with confusion matrices and Cohen's kappa.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter

from .config import ROOT

CONDS = ["base", "retry", "retry_evidence", "adr_untyped", "adr_hints", "adr_v1", "adr", "adr_oracle"]

# hypotheses fixed before the confirmatory run, Holm-corrected together
REGISTERED = [("H1", "adr", "retry_evidence"), ("H2", "adr_hints", "adr_untyped"), ("H3", "adr", "adr_hints")]

# final views: base/retry/ADR v1 come from the first runs, ADR v2 and its oracle from the v2 runs
COMBINED = {
    "final_mini": ("gpt-5.4-mini", [("main_mini", {"base": "base", "retry": "retry", "adr": "adr_v1"}),
                                    ("v2_mini", {"adr": "adr", "adr_oracle": "adr_oracle"})]),
    "final_luna": ("gpt-5.6-luna", [("main", {"base": "base", "retry": "retry", "adr": "adr_v1"}),
                                    ("v2_luna", {"adr": "adr", "adr_oracle": "adr_oracle"})]),
}
FAULTS = ["none", "grounding", "stagnation", "occlusion", "wrong_value", "loading", "layout_shift"]
HELD_OUT = ["loading", "layout_shift"]
TRUE = {"grounding": "I", "stagnation": "II", "occlusion": "III", "wrong_value": "IV",
        "loading": "II", "layout_shift": "I"}
TYPES = ["I", "II", "III", "IV", "V", "VI", "N"]


# statistics

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(pairs):
    """Exact two-sided McNemar on paired booleans (a, b). Returns (b_only, a_only, p)."""
    a_only = sum(1 for a, b in pairs if a and not b)
    b_only = sum(1 for a, b in pairs if b and not a)
    n = a_only + b_only
    if n == 0:
        return b_only, a_only, 1.0
    k = min(a_only, b_only)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return b_only, a_only, min(1.0, 2 * p)


def kappa(y1, y2):
    n = len(y1)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(y1, y2)) / n
    c1, c2 = Counter(y1), Counter(y2)
    pe = sum(c1[k] * c2[k] for k in set(c1) | set(c2)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def macro_f1(gold, pred, labels):
    f1s = []
    for l in labels:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        if tp + fn == 0:
            continue
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn)
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, same order as given."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def cluster_ci(da: dict, db: dict, keys, n_boot: int = 2000, seed: int = 0):
    """Difference in success rate (a - b) with a 95% interval from resampling whole tasks."""
    import random
    if not keys:
        return None
    by_task = {}
    for k in keys:
        by_task.setdefault(k[0], []).append(k)
    tasks = sorted(by_task)

    def diff(ts):
        ks = [k for t in ts for k in by_task[t]]
        return (sum(da[k]["success"] for k in ks) - sum(db[k]["success"] for k in ks)) / len(ks)
    rng = random.Random(seed)
    boots = sorted(diff([rng.choice(tasks) for _ in tasks]) for _ in range(n_boot))
    return {"diff": diff(tasks), "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot) - 1]}


def pct(x):
    return f"{100 * x:.1f}%"


# RQ2

def load_run(run):
    """Episodes of a run, by exact folder name or by run name (newest versioned folder)."""
    from .runstore import resolve
    folder = resolve(ROOT / "runs", run)
    eps, errors = [], 0
    if folder is None:
        return eps, errors
    run_cfg = json.loads((folder / "config.json").read_text()) if (folder / "config.json").exists() else {}
    for f in sorted((folder / "episodes").glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("error"):
            errors += 1
            continue
        r["_id"] = f.stem
        r["_run"] = folder.name
        r["_run_partial"] = bool(run_cfg.get("_partial"))
        r["_run_planned"] = run_cfg.get("_planned")
        eps.append(r)
    return eps, errors


def tokens(r, tags=None):
    u = r.get("usage", {})
    return sum(v.get("prompt_tokens", 0) + v.get("completion_tokens", 0)
               for k, v in u.items() if tags is None or k in tags)


def rq2(eps):
    by = {c: {} for c in CONDS}
    for r in eps:
        s = r["spec"]
        by[r["condition"]][(s["task"], s["seed"], s["fault"])] = r
    out = {"n": {c: len(by[c]) for c in CONDS}}

    # success rates
    table = {}
    for c in CONDS:
        row = {}
        for fk in FAULTS + ["faulted", "held_out", "all"]:
            sel = [r for k, r in by[c].items()
                   if (fk == "all") or (fk == "faulted" and k[2] != "none")
                   or (fk == "held_out" and k[2] in HELD_OUT) or k[2] == fk]
            k = sum(r["success"] for r in sel)
            row[fk] = {"k": k, "n": len(sel), "sr": k / len(sel) if sel else None, "ci": wilson(k, len(sel))}
        table[c] = row
    out["success"] = table

    # paired tests on keys present in both conditions
    tests = {}
    pairs_to_test = [("adr", "base"), ("adr", "retry"), ("retry", "base"), ("adr", "adr_v1"), ("adr_oracle", "adr"),
                     ("adr_v1", "retry"), ("adr_v1", "base"),
                     ("adr", "retry_evidence"), ("adr_hints", "adr_untyped"), ("adr", "adr_hints"),
                     ("adr_untyped", "retry_evidence"), ("retry_evidence", "retry"), ("retry_evidence", "base")]
    for a, b in pairs_to_test:
        for scope in ("faulted", "all", "held_out"):
            keys = [k for k in by[a] if k in by[b] and (scope == "all" or (scope == "faulted" and k[2] != "none")
                                                         or (scope == "held_out" and k[2] in HELD_OUT))]
            pairs = [(by[a][k]["success"], by[b][k]["success"]) for k in keys]
            b_only, a_only, p = mcnemar(pairs)
            tests[f"{a} vs {b} ({scope})"] = {"n_pairs": len(pairs), f"{a}_only": a_only,
                                               f"{b}_only": b_only, "p": p,
                                               "sr_a": sum(x for x, _ in pairs) / max(1, len(pairs)),
                                               "sr_b": sum(y for _, y in pairs) / max(1, len(pairs)),
                                               "cluster_ci": cluster_ci(by[a], by[b], keys)}
    out["tests"] = tests
    # completeness: a block is one (task, seed, fault); it is complete when every condition of the run has it
    present = [c for c in CONDS if by[c]]
    blocks = {k for c in present for k in by[c]}
    complete = {k for k in blocks if all(k in by[c] for c in present if not (c == "adr_oracle" and k[2] == "none"))}
    run_partial = any(r.get("_run_partial") for r in eps)
    planned = max((r.get("_run_planned") or 0 for r in eps), default=0)
    out["completeness"] = {"blocks": len(blocks), "complete_blocks": len(complete), "episodes": len(eps),
                           "planned_episodes": planned or None,
                           "partial": run_partial or len(complete) < len(blocks)}
    # the pre-registered hypotheses (primary scope: all episodes), Holm-corrected, on complete blocks only, so
    # every hypothesis is tested on the same episodes even if the run stopped early
    reg = []
    for h, a, b in REGISTERED:
        keys = sorted(k for k in complete if k in by[a] and k in by[b])
        if keys:
            b_only, a_only, p = mcnemar([(by[a][k]["success"], by[b][k]["success"]) for k in keys])
            reg.append((h, a, b, {"p": p, "n_pairs": len(keys), "a_only": a_only, "b_only": b_only,
                                  "cluster_ci": cluster_ci(by[a], by[b], keys)}))
    adj = holm([t["p"] for *_, t in reg])
    out["registered"] = [{"hypothesis": h, "a": a, "b": b, "p": t["p"], "p_holm": q, "n_pairs": t["n_pairs"],
                          "a_only": t["a_only"], "b_only": t["b_only"], "cluster_ci": t["cluster_ci"],
                          "partial_run": out["completeness"]["partial"]} for (h, a, b, t), q in zip(reg, adj)]

    # per-task success, to see where any effect comes from
    tasks = sorted({k[0] for c in CONDS for k in by[c]})
    out["per_task"] = {t: {c: (sum(r["success"] for k, r in by[c].items() if k[0] == t) /
                               max(1, sum(1 for k in by[c] if k[0] == t))) if any(k[0] == t for k in by[c]) else None
                           for c in CONDS} for t in tasks}

    # recovery metrics on episodes where the fault actually fired
    rec = {}
    for c in CONDS:
        fired = [r for k, r in by[c].items() if k[2] != "none" and r["fault"].get("fired")]
        detected = [r for r in fired if any(s.get("fault_related") for s in r["steps"])]
        rec[c] = {
            "fired": len(fired),
            "post_error_success": sum(r["success"] for r in fired) / max(1, len(fired)),
            "error_awareness": (len(detected) / max(1, len(fired))) if c != "base" else None,
            "recovery_rate": (sum(r["success"] for r in detected) / max(1, len(detected))) if c != "base" else None,
            "detected": len(detected),
        }
        per = {}
        for fk in FAULTS[1:]:
            ff = [r for k, r in by[c].items() if k[2] == fk and r["fault"].get("fired")]
            dd = [r for r in ff if any(s.get("fault_related") for s in r["steps"])]
            per[fk] = {"fired": len(ff), "success": sum(r["success"] for r in ff) / max(1, len(ff)),
                       "detected": len(dd) / max(1, len(ff))}
        rec[c]["per_fault"] = per
    out["recovery"] = rec

    # Monitor: detection vs false alarms
    mon_eps = [r for c in CONDS if c not in ("base", "adr_v1") for r in by[c].values()]
    clean_steps = [s for r in mon_eps if r["spec"]["fault"] == "none" for s in r["steps"] if "verdict" in s]
    fa = sum(1 for s in clean_steps if (s.get("verdict") or {}).get("flagged"))
    fired = [r for r in mon_eps if r["spec"]["fault"] != "none" and r["fault"].get("fired")]
    det = sum(1 for r in fired if any(s.get("fault_related") for s in r["steps"]))
    out["monitor"] = {"fault_recall": det / max(1, len(fired)), "faults": len(fired),
                      "false_alarm_per_step": fa / max(1, len(clean_steps)), "clean_steps": len(clean_steps),
                      "paired_discrimination": det / max(1, len(fired)) - fa / max(1, len(clean_steps)),
                      "false_done_vetoes": sum(1 for r in mon_eps for s in r["steps"]
                                               if (s.get("verdict") or {}).get("symptom") == "false_done")}

    # RQ1 live: diagnosis vs injected type
    gold, pred = [], []
    symptoms = []
    for c in ("adr", "adr_oracle", "adr_v1", "adr_hints"):
        for r in by[c].values():
            for s in r["steps"]:
                d = s.get("diagnosis")
                if s.get("fault_related") and d:
                    gold.append(TRUE[r["spec"]["fault"]])
                    pred.append(d.get("model_type") or d.get("type"))
                    symptoms.append((s.get("verdict") or {}).get("symptom"))
    # how far a plain lookup on the Monitor's symptom would get (best type per symptom)
    best = {}
    for sym in set(symptoms):
        best[sym] = Counter(g for g, s_ in zip(gold, symptoms) if s_ == sym).most_common(1)[0][0]
    lookup_acc = sum(best[s_] == g for g, s_ in zip(gold, symptoms)) / max(1, len(gold))
    conf = {g: Counter() for g in TRUE.values()}
    for g, p in zip(gold, pred):
        conf[g][p] += 1
    clean_diag = Counter(s["diagnosis"].get("type") for c in ("adr",) for r in by[c].values()
                         if r["spec"]["fault"] == "none" for s in r["steps"] if s.get("diagnosis"))
    out["rq1_live"] = {"n": len(gold), "accuracy": sum(g == p for g, p in zip(gold, pred)) / max(1, len(gold)),
                       "macro_f1": macro_f1(gold, pred, list(TRUE.values())), "kappa": kappa(gold, pred),
                       "confusion": {g: dict(v) for g, v in conf.items()},
                       "clean_false_alarm_diagnoses": dict(clean_diag),
                       "symptom_lookup_accuracy": lookup_acc, "symptom_lookup_rule": best}

    # cost and effort
    cost = {}
    for c in CONDS:
        rs = list(by[c].values())
        n = max(1, len(rs))
        cost[c] = {"tokens": sum(tokens(r) for r in rs) / n,
                   "agent_tokens": sum(tokens(r, {"agent"}) for r in rs) / n,
                   "wrapper_tokens": sum(tokens(r, {"judge", "diag", "ground"}) for r in rs) / n,
                   "seconds": sum(r.get("seconds", 0) for r in rs) / n,
                   "agent_steps": sum(r.get("agent_steps", 0) for r in rs) / n,
                   "actions": sum(r.get("actions", 0) for r in rs) / n,
                   "recoveries": sum(r.get("recoveries", 0) for r in rs) / n}
    out["cost"] = cost
    # extra tokens per extra successful episode, against the base agent on the same episodes
    cpr = {}
    for c in CONDS:
        keys = [k for k in by[c] if k in by["base"]] if c != "base" else []
        if not keys:
            continue
        extra = sum(tokens(by[c][k]) - tokens(by["base"][k]) for k in keys) / len(keys)
        gain = sum(by[c][k]["success"] - by["base"][k]["success"] for k in keys) / len(keys)
        cpr[c] = {"extra_tokens": extra, "extra_success": gain,
                  "tokens_per_rescue": (extra / gain) if gain > 0 else None}
    out["cost_per_rescue"] = cpr
    out["plans"] = {c: dict(Counter((s.get("plan") or {}).get("kind") for r in by[c].values()
                                    for s in r["steps"] if s.get("plan"))) for c in ("adr", "adr_oracle")}
    # ADR v2: the click check (Prevent) and Check the fix
    v2 = list(by["adr"].values())
    pre = [s["precommit"] for r in v2 for s in r["steps"] if "precommit" in s]
    blocked_eps = [r for r in v2 if any((s.get("precommit") or {}).get("blocked") for s in r["steps"])]
    out["v2"] = {"precommit_checks": len(pre), "judged_final": sum(p.get("is_final") is True for p in pre),
                 "blocks": sum(p["blocked"] for p in pre), "blocked_episodes": len(blocked_eps),
                 "blocked_episode_success": sum(r["success"] for r in blocked_eps) / max(1, len(blocked_eps)),
                 "recovery_checks": sum(1 for r in v2 for s in r["steps"] if "recovery_check" in s),
                 "escalations": sum(1 for r in v2 for s in r["steps"] if s.get("escalation")),
                 "votes": sum(1 for r in v2 for s in r["steps"] if (s.get("diagnosis") or {}).get("votes")),
                 "generic_fallbacks": sum(1 for r in v2 for s in r["steps"]
                                          if (s.get("plan") or {}).get("kind") == "generic_fallback")}
    return out, by


# RQ1 offline

def rq1_offline(name):
    """An offline RQ1 run by name; a stopped run is read from its saved cases."""
    from .runstore import CaseStore, resolve
    folder = resolve(ROOT / "runs" / "rq1", name)
    if folder is None:
        return None
    p = folder / "results.json"
    if p.exists():
        rs = json.loads(p.read_text())
    elif (folder / "cases").exists():
        rs = CaseStore(folder).all()
    else:
        return None
    from .config import Config
    from .rq1_offline import human_type
    from .taxonomy import Taxonomy
    tax = Taxonomy(Config().path(Config().taxonomy))
    for r in rs:   # always score with the current mapping in the taxonomy file
        r["human_type"] = human_type(tax, r["tag"])
    sc = [r for r in rs if r["human_type"] and "error" not in r["diag"]]
    gold, pred = [r["human_type"] for r in sc], [r["pred_type"] for r in sc]
    fam = [r for r in rs if r["human_family"] in "PGRS" and "error" not in r["diag"]]
    conf = {g: Counter() for g in TYPES}
    for g, p_ in zip(gold, pred):
        conf[g][p_] += 1
    by_model = {}
    for m in sorted({r["model"] for r in sc}):
        x = [r for r in sc if r["model"] == m]
        f = [r for r in fam if r["model"] == m]
        by_model[m] = {"n": len(x), "type_acc": sum(r["pred_type"] == r["human_type"] for r in x) / max(1, len(x)),
                       "family_acc": sum(r["pred_family"] == r["human_family"] for r in f) / max(1, len(f))}
    hi = [r for r in sc if r["confidence"] == "high"]
    return {"name": name, "n": len(sc), "excluded": len(rs) - len(sc),
            "type_accuracy": sum(g == p_ for g, p_ in zip(gold, pred)) / max(1, len(sc)),
            "macro_f1": macro_f1(gold, pred, TYPES[:-1]), "kappa": kappa(gold, pred),
            "family_accuracy": sum(r["pred_family"] == r["human_family"] for r in fam) / max(1, len(fam)),
            "family_kappa": kappa([r["human_family"] for r in fam], [r["pred_family"] for r in fam]),
            "high_conf_type_accuracy": sum(r["pred_type"] == r["human_type"] for r in hi) / max(1, len(hi)),
            "gold_dist": dict(Counter(gold)), "pred_dist": dict(Counter(pred)),
            "confusion": {g: dict(v) for g, v in conf.items() if v}, "by_model": by_model}


def main(argv=None):
    from . import report
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="final_mini,final_luna",
                    help="run:agent pairs or names from COMBINED, one RQ2 section each")
    ap.add_argument("--rq1", default="full,twoframe")
    a = ap.parse_args(argv)
    runs, summary = [], {}
    for item in a.runs.split(","):
        run, _, agent = item.partition(":")
        if run in COMBINED:
            agent, sources = COMBINED[run]
            eps, errors = [], 0
            for src, mapping in sources:
                if not (ROOT / "runs" / src / "episodes").exists():
                    continue
                e, er = load_run(src)
                errors += er
                for r in e:
                    if r["condition"] in mapping:
                        r["condition"] = mapping[r["condition"]]
                        eps.append(r)
        elif (ROOT / "runs" / run / "episodes").exists():
            eps, errors = load_run(run)
        else:
            continue
        res, by = rq2(eps)
        res["errors"] = errors
        runs.append((agent or run, run, res, by))
        summary[run] = res
        s = res["success"]
        print(f"[{run} / {agent}] episodes {sum(res['n'].values())} (errors skipped {errors})")
        for c in CONDS:
            if not s[c]["all"]["n"]:
                continue
            print(f"  {c:11s} faulted {s[c]['faulted']['k']}/{s[c]['faulted']['n']} = {pct(s[c]['faulted']['sr'] or 0)}"
                  f"   clean {s[c]['none']['k']}/{s[c]['none']['n']}")
        for k, v in res["tests"].items():
            if "faulted" in k:
                print(f"  {k}: p={v['p']:.3g} pairs={v['n_pairs']}")
        print("  monitor", {k: round(v, 3) if isinstance(v, float) else v for k, v in res["monitor"].items()})
        print("  rq1 live", {k: v for k, v in res["rq1_live"].items() if k != "confusion"})
    rq1s = [x for x in (rq1_offline(n) for n in a.rq1.split(",") if n) if x]
    ag = ROOT / "runs" / "rq1" / "agentic" / "results.json"
    if ag.exists():
        from .rq1_agentic import summarize
        from .config import Config
        from .rq1_offline import human_type
        from .taxonomy import Taxonomy
        tax = Taxonomy(Config().path(Config().taxonomy))
        ars = json.loads(ag.read_text())
        for r in ars:
            r["human_type"] = human_type(tax, r["tag"])
        summary["rq1_agentic"] = summarize(ars)
        summary["rq1_agentic"]["claude"] = summarize([r for r in ars if r["model"].startswith("claude")])
        print("  rq1 agentic", summary["rq1_agentic"])
    summary["rq1_offline"] = rq1s
    for q in rq1s:
        print("  rq1", q["name"], {k: q[k] for k in ("n", "type_accuracy", "family_accuracy", "kappa", "macro_f1")})
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    report.build(runs, rq1s, summary.get("rq1_agentic"))
    print("wrote results/report.html and results/summary.json")


if __name__ == "__main__":
    main()
