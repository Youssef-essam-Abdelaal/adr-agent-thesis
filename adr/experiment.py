"""
Parallel experiment runner. One browser per worker process.

A run lives in runs/<name>__<fingerprint>. The fingerprint changes with any code,
taxonomy or setting that can change a result, so a new implementation always gets a
new folder and never mixes with old episodes. Every episode is saved (atomically) the
moment it finishes, so running the same command again continues where it stopped:
finished episodes are skipped, unreadable files are redone, and episodes that crashed
are retried once.

Before starting, the runner prints how many episodes are left and an estimate of the
tokens they need, and asks to continue (skip with --yes). A cap (default 1.5 x the
estimate) stops the run gracefully: no new episodes start, running ones finish and
are saved, and the run can be resumed later. A stopped run is marked partial; the
analysis then uses only complete sets (all conditions finished for the same task,
seed and fault).

    python -m adr.experiment --name main --tasks click-option,login-user \
        --seeds 1,2 --faults none,grounding --conditions base,adr --workers 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .config import ROOT, Config
from .env.faults import Fault
from .runstore import (LIVE_FILES, atomic_write, fingerprint, keep_awake, other_versions, read_result,
                       run_folder)

_PW = None
_BROWSER = None

# tokens per episode (thousands), measured in earlier runs; used only for the estimate
EST_K = {"base": 3.8, "retry": 8.6, "retry_evidence": 8.8, "adr_untyped": 11.5, "adr_hints": 11.0,
         "adr": 12.6, "adr_oracle": 12.0}
USD_PER_MTOK = (1.0, 3.0)   # rough assumption for the dollar range; check the OpenAI usage page for real cost


def _browser():
    global _PW, _BROWSER
    if _BROWSER is None or not _BROWSER.is_connected():
        from playwright.sync_api import sync_playwright
        if _PW is None:
            _PW = sync_playwright().start()
        _BROWSER = _PW.chromium.launch(channel="chrome", headless=True)
    return _BROWSER


def spec_id(s: dict) -> str:
    return f"{s['condition']}__{s['task']}__{s['fault']}__s{s['seed']}"


def run_spec(spec: dict, cfg_dict: dict, out_root: str, crash_retry: bool = False) -> dict:
    from .env.miniwob import MiniWoBEnv
    from .wrapper import ADRWrapper
    cfg = Config(**cfg_dict)
    out = Path(out_root)
    path = out / "episodes" / f"{spec_id(spec)}.json"
    fault = Fault(kind=spec["fault"], at=spec["at"], variant=spec["seed"])
    env = None
    try:
        if cfg.env_kind == "osworld":
            from .env.osworld import OSWorldEnv
            env = OSWorldEnv(cfg)
        else:
            env = MiniWoBEnv(cfg, _browser())
        if cfg.adr_version >= 3:
            from .v3.wrapper import ADRWrapperV3 as ADRWrapper   # noqa: N806
        w = ADRWrapper(cfg, spec["condition"])
        frames = out / "frames" / spec_id(spec)
        if frames.exists() and any(frames.iterdir()):   # an earlier attempt was cut off: keep its screenshots apart
            stale = out / "frames" / "_stale" / f"{spec_id(spec)}__{int(time.time())}"
            stale.parent.mkdir(parents=True, exist_ok=True)
            frames.rename(stale)
        res = w.run(env, spec["task"], spec["seed"], fault, frames)
    except Exception as e:
        res = {**spec, "success": False, "error": f"{type(e).__name__}: {e}",
               "trace": traceback.format_exc()[-1500:]}
    finally:
        if env is not None:
            env.close()
    res["spec"] = spec
    if crash_retry:
        res["crash_retry"] = True
    atomic_write(path, json.dumps(res, indent=1, default=str))
    return res


def build_specs(tasks, seeds, faults, conditions, type_tasks=None):
    specs = []
    for task in tasks:
        for seed in seeds:
            for fk in faults:
                if fk == "wrong_value" and type_tasks is not None and task not in type_tasks:
                    continue
                for cond in conditions:
                    if cond == "adr_oracle" and fk == "none":
                        continue
                    specs.append({"task": task, "seed": seed, "fault": fk, "condition": cond,
                                  "at": 1 if fk == "wrong_value" else 1 + (seed % 2)})
    return specs


def plan(specs, out: Path):
    """Split specs into done, to run, and to retry after a crash."""
    done, todo, retry = [], [], []
    for s in specs:
        r = read_result(out / "episodes" / f"{spec_id(s)}.json")
        if r is None:
            todo.append(s)
        elif r.get("error") and not r.get("crash_retry"):
            retry.append(s)
        else:
            done.append((s, r))
    return done, todo, retry


def estimate_tokens(specs, cfg: Config, done) -> float:
    seen = defaultdict(list)
    for _, r in done:
        if not r.get("error"):
            seen[r.get("condition")].append(sum(v.get("prompt_tokens", 0) + v.get("completion_tokens", 0)
                                                for v in (r.get("usage") or {}).values()))
    total = 0.0
    for s in specs:
        c = s["condition"]
        if len(seen[c]) >= 20:
            total += sum(seen[c]) / len(seen[c])
            continue
        k = EST_K.get(c, 12.0)
        if cfg.adr_version == 1 and c in ("adr", "adr_oracle"):
            k = 8.8
        if c in ("retry", "retry_evidence") and cfg.precommit and cfg.precommit_scope == "all":
            k += 3.7
        total += k * 1000
    return total


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--type-tasks", default="", help="tasks where a wrong_value fault can fire")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--faults", default="none,grounding,stagnation,occlusion,wrong_value")
    ap.add_argument("--conditions", default="base,retry,adr,adr_oracle")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--experiment", default="", help="a label stored in the run's config.json")
    ap.add_argument("--max-tokens", type=float, default=None,
                    help="stop starting new episodes after this many tokens (default 1.5 x estimate; 0 = no cap)")
    ap.add_argument("--yes", action="store_true", help="do not ask before starting")
    ap.add_argument("--legacy-folder", action="store_true",
                    help="use runs/<name> without a fingerprint (only to resume runs made before versioned folders)")
    a = ap.parse_args(argv)

    cfg = Config.load(a.config, a.set)
    if a.workers:
        cfg.workers = a.workers
    fp = fingerprint(LIVE_FILES, cfg.to_dict())
    out = ROOT / "runs" / a.name if a.legacy_folder else run_folder(ROOT / "runs", a.name, fp)
    olds = [] if a.legacy_folder else other_versions(ROOT / "runs", a.name, fp)
    if olds and not out.exists():
        print(f"Code or settings differ from earlier versions of '{a.name}' ({', '.join(p.name for p in olds)}). "
              f"Starting a new folder; earlier results are kept.", flush=True)
    out.mkdir(parents=True, exist_ok=True)

    type_tasks = set(a.type_tasks.split(",")) if a.type_tasks else None
    specs = build_specs(a.tasks.split(","), [int(s) for s in a.seeds.split(",")],
                        a.faults.split(","), a.conditions.split(","), type_tasks)
    atomic_write(out / "specs.json", json.dumps(specs))
    done, todo, retry = plan(specs, out)
    est = estimate_tokens(todo + retry, cfg, done)
    cap = est * 1.5 if a.max_tokens is None else (a.max_tokens or float("inf"))
    print(f"Run folder: {out}\n"
          f"Episodes: {len(specs)} planned, {len(done)} done, {len(todo)} to run, {len(retry)} to retry after a crash.\n"
          f"Estimate for what is left: about {est / 1e6:.1f}M tokens "
          f"(roughly ${est / 1e6 * USD_PER_MTOK[0]:.0f} to ${est / 1e6 * USD_PER_MTOK[1]:.0f}). "
          f"Cap for this session: {'none' if cap == float('inf') else f'{cap / 1e6:.1f}M tokens'}.", flush=True)
    if not todo and not retry:
        print("Nothing left to run.", flush=True)
    elif not a.yes and sys.stdin.isatty():
        if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Not started.")
            return 1

    from .provenance import code_version
    cpath = out / "config.json"
    meta = read_result(cpath) or {**cfg.to_dict(), "_provenance": code_version(), "_fingerprint": fp,
                                  "_experiment": a.experiment, "_sessions": []}
    if a.experiment:
        meta["_experiment"] = a.experiment
    session = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "provenance": code_version(),
               "command": " ".join(sys.argv), "done_before": len(done), "to_run": len(todo) + len(retry)}
    meta.setdefault("_sessions", []).append(session)
    atomic_write(cpath, json.dumps(meta, indent=1))

    keep_awake()
    t0, used, finished, stopped = time.time(), 0.0, 0, ""
    tally = defaultdict(lambda: [0, 0])
    jobs = [(s, False) for s in todo] + [(s, True) for s in retry]
    try:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futs = {ex.submit(run_spec, s, cfg.to_dict(), str(out), cr): s for s, cr in jobs}
            for f in as_completed(futs):
                s = futs[f]
                if f.cancelled():
                    continue
                try:
                    r = f.result()
                except Exception as e:
                    r = {"success": False, "error": str(e)}
                finished += 1
                used += sum(v.get("prompt_tokens", 0) + v.get("completion_tokens", 0)
                            for v in (r.get("usage") or {}).values())
                t = tally[s["condition"]]
                t[0] += int(bool(r.get("success")))
                t[1] += 1
                err = f" ERROR {str(r.get('error'))[:80]}" if r.get("error") else ""
                line = " ".join(f"{c}={v[0]}/{v[1]}" for c, v in sorted(tally.items()))
                print(f"[{len(done) + finished}/{len(specs)} {time.time() - t0:5.0f}s {used / 1e6:.2f}M tok] "
                      f"{spec_id(s)} {'OK ' if r.get('success') else 'FAIL'}{err} | {line}", flush=True)
                if used > cap and not stopped:
                    stopped = f"token cap reached ({used / 1e6:.1f}M)"
                    n = sum(1 for g in futs if g.cancel())
                    print(f"{stopped}: {n} episodes not started; running ones will finish. "
                          f"Run the same command again to continue.", flush=True)
    except KeyboardInterrupt:
        stopped = "interrupted by the user"
        print("\nInterrupted. Finished episodes are saved; run the same command again to continue.", flush=True)

    done_now, todo_now, retry_now = plan(specs, out)
    crashed = sum(1 for _, r in done_now if r.get("error"))
    session.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), tokens=used, stopped=stopped,
                   done_after=len(done_now))
    meta.update(_finished=session["finished"], _episodes=len(done_now), _planned=len(specs),
                _crashed=crashed, _partial=bool(todo_now or retry_now))
    atomic_write(cpath, json.dumps(meta, indent=1))
    left = len(todo_now) + len(retry_now)
    print(f"{'Stopped (' + stopped + ')' if stopped and left else 'Finished'} after {round(time.time() - t0)} s: {len(done_now)} of {len(specs)} "
          f"episodes saved, {crashed} crashed twice, {left} left.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
