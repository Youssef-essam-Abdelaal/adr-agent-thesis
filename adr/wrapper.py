"""
The wrapper: the loop the agent runs inside.

    observe -> agent decides -> Prevent -> act -> Verify -> (Diagnose -> Recover -> check the fix) -> repeat

The agent is not modified and does not know the wrapper is there. Conditions
used for RQ2 differ only in what happens after a flag:

    base            Base, agent alone: no wrapper
    retry           generic retry: Monitor + one fixed reflect-and-retry message
    retry_evidence  generic retry + Monitor's evidence: the same message plus what the Monitor saw
    adr_untyped     free-form diagnosis: Monitor + the Diagnostician's model and evidence, but no taxonomy;
                    the agent gets the advice, the wrapper does not act (ablation)
    adr_hints       taxonomy diagnosis told to the agent: Monitor + typed diagnosis, the agent is told the fix
                    and does it itself (ablation)
    adr             ADR: Monitor + Diagnostician + Strategist, which carries out the fix and checks it
    adr_oracle      ADR with oracle diagnosis: as adr, but on flags caused by the injected fault the true type
                    replaces the Diagnostician's (upper bound: perfect diagnosis, same recovery)

Read as a chain: retry_evidence -> adr_untyped adds a model's free-form diagnosis;
adr_untyped -> adr_hints adds the taxonomy (same model, same evidence, advice only);
adr_hints -> adr adds acting on the agent's behalf.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .agent import make_agent
from .config import Config
from .diagnostician import Diagnostician, UntypedDiagnostician
from .env.faults import Fault
from .llm import reset_usage, usage_snapshot
from .monitor import Monitor, is_commit
from .strategist import Plan, Strategist, evidence_plan, generic_plan, untyped_plan
from .taxonomy import Taxonomy

CONDITIONS = ["base", "retry", "retry_evidence", "adr_untyped", "adr_hints", "adr", "adr_oracle"]


class ADRWrapper:
    def __init__(self, cfg: Config, condition: str = "adr", taxonomy: Optional[Taxonomy] = None):
        assert condition in CONDITIONS, condition
        self.cfg, self.condition = cfg, condition
        tax = taxonomy or Taxonomy(cfg.path(cfg.taxonomy))
        self.agent = make_agent(cfg)
        self.monitor = Monitor(cfg) if condition != "base" else None
        typed = condition in ("adr", "adr_oracle", "adr_hints")
        self.diag = (Diagnostician(cfg, tax) if typed else
                     UntypedDiagnostician(cfg, tax) if condition == "adr_untyped" else None)
        self.strat = Strategist(cfg, tax) if typed else None
        # the click check belongs to every ADR variant; the retry baselines get it only with
        # precommit_scope=all, so that all wrapped conditions share identical detection
        self.prevent = bool(cfg.precommit) and condition != "base" and (
            condition.startswith("adr") or cfg.precommit_scope == "all")

    def run(self, env, task: str, seed: int, fault: Optional[Fault] = None,
            out_dir: Optional[Path] = None) -> dict:
        cfg = self.cfg
        reset_usage()
        t0 = time.time()
        fault = fault or Fault()
        goal = env.reset(task, seed, fault)
        if self.monitor:
            self.monitor.reset()
        if self.strat:
            self.strat.reset()
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        history, frames, steps = [], [], []
        feedback, recoveries, n_actions, end = None, 0, 0, "max_steps"
        blocks = 0
        fault_attributed = False
        shot = env.screenshot()
        self._save(out_dir, shot, "00")

        for step in range(cfg.max_steps):
            prop = self.agent.act(shot, goal, history, feedback)
            feedback = None
            rec = {"step": step, "thought": prop.thought, "proposed": prop.action}

            if prop.done:
                if self.monitor and cfg.done_check and recoveries < cfg.max_recoveries:
                    v = self.monitor.check_done(goal, shot, prop.thought)
                    rec["verdict"] = v
                    if v["flagged"]:
                        plan, d = self._recover(goal, history, v, shot, shot, prop.thought,
                                                prop.action, frames, fault, step, False)
                        rec.update(diagnosis=d, plan=self._plan_dict(plan))
                        recoveries += 1
                        if plan.feedback and not plan.stop:
                            feedback = plan.feedback
                            steps.append(rec)
                            continue
                end = "declared_done"
                steps.append(rec)
                if hasattr(env, "on_done"):     # OSWorld scores the final DONE or FAIL claim
                    env.on_done(prop.action)
                break

            # Prevent: the click check runs before an action that finalizes the task
            candidate = (prop.action.get("type") == "click" if cfg.precommit_mode == "clicks" else False) \
                or is_commit(prop.thought, prop.action, cfg.precommit_words)
            if self.prevent and blocks < cfg.precommit_max_blocks and candidate:
                pc = self.monitor.check_commit(goal, shot, prop.thought, prop.action, history)
                rec["precommit"] = pc
                if pc["blocked"]:
                    blocks += 1
                    feedback = (f"Do not finalize yet. Problem: {pc['problem']} Fix this first: {pc['fix']} "
                                f"Then finalize.")
                    steps.append(rec)
                    continue

            before = shot
            info = env.execute(prop.action, "agent", step)
            n_actions += 1
            after = env.screenshot()
            executed = info["executed"]
            fp = fault.params
            if info["fault_fired"] and fp.get("consistent") and fp.get("original"):
                # a real typo is in the agent's reasoning too: no tell-tale mismatch
                prop.thought = prop.thought.replace(str(fp["original"]), str(fp["corrupted"]))
                rec["thought"] = prop.thought
            rec.update(executed=executed, fault_fired=info["fault_fired"], error=info["error"])
            entry = {"action": executed, "thought": prop.thought}
            history.append(entry)
            frames.append(before)
            st = env.status()
            if st["done"]:
                end = "env_done"
                steps.append(rec)
                self._save(out_dir, after, f"{step + 1:02d}")
                break

            if self.monitor:
                v = self.monitor.check(goal, before, after, prop.thought, executed, history)
                rec["verdict"] = entry["verdict"] = v
                related = (fault.fired and not fault_attributed and v["flagged"]
                           and fault.fired_step <= step <= fault.fired_step + 2)
                if related:
                    fault_attributed = True
                rec["fault_related"] = related
                if v["flagged"] and recoveries < cfg.max_recoveries:
                    self._save(out_dir, after, f"{step:02d}_flag")
                    plan, d = self._recover(goal, history[:-1], v, before, after, prop.thought,
                                            executed, frames[:-1], fault, step, related)
                    rec.update(diagnosis=d, plan=self._plan_dict(plan))
                    recoveries += 1
                    done_n, blocked, blocks = self._execute(plan, env, rec, history, step, goal, blocks)
                    n_actions += done_n
                    if blocked:
                        plan = Plan(f"{plan.kind}_blocked", feedback=(
                            f"The supervisor stopped before finalizing for you, because: {blocked['problem']} "
                            f"Fix this first: {blocked['fix']}"))
                        rec["plan_after_block"] = self._plan_dict(plan)
                    if plan.actions:
                        rec_before, after = after, env.screenshot()
                        if cfg.verify_recovery and not env.status()["done"]:
                            last_rec = rec["recovery_executed"][-1] if rec.get("recovery_executed") else executed
                            chk = self.monitor.check(goal, rec_before, after, f"supervisor recovery ({plan.kind})",
                                                     last_rec, history)
                            rec["recovery_check"] = chk
                            if chk["flagged"]:
                                plan = self.strat.escalate(plan, chk, executed)
                                rec["escalation"] = self._plan_dict(plan)
                                done_n, blocked, blocks = self._execute(plan, env, rec, history, step, goal, blocks)
                                n_actions += done_n
                                if blocked:
                                    plan = Plan(f"{plan.kind}_blocked", feedback=(
                                        f"The supervisor stopped before finalizing for you, because: "
                                        f"{blocked['problem']} Fix this first: {blocked['fix']}"))
                                if plan.actions:
                                    after = env.screenshot()
                    feedback = plan.feedback
                    if plan.stop:
                        end = "escalated"
                        steps.append(rec)
                        break
                    if env.status()["done"]:
                        end = "env_done"
                        steps.append(rec)
                        break

            shot = after
            steps.append(rec)
            self._save(out_dir, shot, f"{step + 1:02d}")
            if n_actions >= cfg.max_actions:
                end = "max_actions"
                break

        st = env.status(final=True)
        return {
            "task": task, "seed": seed, "condition": self.condition, "goal": goal,
            "fault": fault.to_dict(), "success": bool(st["success"]), "env_done": bool(st["done"]),
            "raw_reward": st.get("raw"), "reason": st.get("reason"), "end": end,
            "agent_steps": len(steps), "actions": n_actions, "recoveries": recoveries,
            "seconds": round(time.time() - t0, 1), "usage": usage_snapshot(), "steps": steps,
        }

    def _execute(self, plan, env, rec, history, step, goal, blocks):
        """
        Run a plan's actions. The agent must see what was done on its behalf, or it
        repeats it. With precommit_recovery=1 the wrapper's own clicks get the same
        click check as the agent's: the wrapper may not finalize wrongly either.
        Returns (actions executed, the blocking check or None, updated block count).
        """
        cfg, n = self.cfg, 0
        for pa in plan.actions:
            if pa.get("wait"):
                time.sleep(pa["wait"])
            a = pa["action"]
            if (self.prevent and cfg.precommit_recovery and blocks < cfg.precommit_max_blocks
                    and (a.get("type") == "click" or (a.get("type") == "key" and str(a.get("key")).lower() == "enter"))):
                pc = self.monitor.check_commit(goal, env.screenshot(), f"supervisor recovery ({plan.kind})", a, history)
                rec.setdefault("recovery_precommit", []).append(pc)
                if pc["blocked"]:
                    return n, pc, blocks + 1
            r = env.execute(a, "recovery", step)
            n += 1
            rec.setdefault("recovery_executed", []).append(r["executed"])
            history.append({"action": r["executed"], "by": "supervisor",
                            "thought": f"done for you by the supervisor ({plan.kind})"})
            if env.status()["done"]:
                break
        return n, None, blocks

    def _recover(self, goal, history, v, before, after, thought, action, frames, fault, step, related):
        if self.condition == "retry":
            return generic_plan(v), None
        if self.condition == "retry_evidence":
            return evidence_plan(v), None
        if self.condition == "adr_untyped":
            a = self.diag.advise(goal, history, v, before, after, thought, action, frames)
            return untyped_plan(a, v), a
        d = self.diag.diagnose(goal, history, v, before, after, thought, action, frames)
        if self.condition == "adr_oracle" and related:
            d = dict(d, model_type=d.get("type"), type=fault.true_type,
                     subtype="dismissable_overlay" if fault.true_type == "III" else d.get("subtype", ""),
                     oracle=True)
        return self.strat.plan(d, v, goal, action, after, hints_only=self.condition == "adr_hints"), d

    @staticmethod
    def _plan_dict(plan: Plan) -> dict:
        return {"kind": plan.kind, "actions": [a["action"] for a in plan.actions],
                "feedback": plan.feedback, "stop": plan.stop, "note": plan.note}

    @staticmethod
    def _save(out_dir, img, name):
        if out_dir is not None:
            img.save(out_dir / f"{name}.jpg", quality=72)
