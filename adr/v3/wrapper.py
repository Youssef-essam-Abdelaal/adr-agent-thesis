"""
ADR v3 loop (v2: adr/wrapper.py, unchanged). Same conditions and the same order of stages; the v3 parts are used for
every wrapped condition, so retry and ADR share the same detection (as in v2). Changes in the loop:

1. Second look: when the step judge says "no effect", the wrapper waits (second_look_s), takes a fresh screenshot and
   asks again; only a "no effect" that is still there counts, because late page updates were read as dead clicks.
   The judge is asked again only if the screen changed while waiting (Layer 1's test); otherwise it would see the same
   picture, and its first answer stands.
2. The done-check gets the start screen and the recent actions.
3. wait_check (adr/v3/strategist.py): after the wait the screen is compared with the one before the agent's action;
   the action is repeated only if nothing changed. A fix that ran no action is not checked afterwards.

The loop also handles these switches, all off by default (see adr/config.py): done_always, with "cannot be done"
reported as infeasible because OSWorld scores infeasible tasks by FAIL; drop_blocked, since an agent that still sees
its blocked reply proposes the same click again until the block cap lets it through; n_free; fix_report, so that a fix
that only opened a list is not reported as a selection; replan_gap; use_axes, for which the wrapper passes the step
budget (termination axis) and adds the axes' sentences to the agent's message; replan_strikes.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from ..diagnostician import UntypedDiagnostician
from ..env.faults import Fault
from ..llm import reset_usage, usage_snapshot
from ..monitor import is_commit
from ..strategist import Plan
from ..taxonomy import Taxonomy
from ..vision import diff_info
from ..wrapper import ADRWrapper
from .diagnostician import DiagnosticianV3
from .monitor import MonitorV3, choosing_in_open_list
from .strategist import LATE, REPEATED, StrategistV3


class ADRWrapperV3(ADRWrapper):
    def __init__(self, cfg, condition: str = "adr", taxonomy: Optional[Taxonomy] = None):
        super().__init__(cfg, condition, taxonomy)
        tax = taxonomy or Taxonomy(cfg.path(cfg.taxonomy))
        if self.monitor:
            self.monitor = MonitorV3(cfg)
        if self.diag and not isinstance(self.diag, UntypedDiagnostician):
            self.diag = DiagnosticianV3(cfg, tax)
        if self.strat:
            self.strat = StrategistV3(cfg, tax)

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
        blocks, done_vetoes = 0, 0
        last_replan = None                   # the step of the last "try another way" message
        last_flag_step = None                # the last step the Monitor flagged
        fault_attributed = False
        shot = env.screenshot()
        first = shot
        self._save(out_dir, shot, "00")

        for step in range(cfg.max_steps):
            prop = self.agent.act(shot, goal, history, feedback)
            rec = {"step": step, "thought": prop.thought, "proposed": prop.action}
            if feedback:                     # the message the agent actually got (a fix can change its planned one)
                rec["feedback_received"] = feedback
            feedback = None

            if prop.done:
                # with done_always the check runs on every "done" but vetoes at most max_done_vetoes times in all
                may_veto = (done_vetoes < cfg.max_done_vetoes if cfg.done_always
                            else recoveries < cfg.max_recoveries)
                if self.monitor and cfg.done_check and (may_veto or cfg.done_always):
                    v = self.monitor.check_done(goal, shot, prop.thought, first=first, history=history)
                    rec["verdict"] = v
                    insists = (cfg.done_always and not may_veto      # "cannot be done" again after the last veto
                               and (v.get("judge") or {}).get("agent_says_impossible") is True)
                    if (v.get("impossible") or insists) and not prop.action.get("infeasible"):
                        prop.action = dict(prop.action, infeasible=True)     # report "cannot be done" as FAIL
                        rec["reported_infeasible"] = True
                    elif v["flagged"] and may_veto:
                        plan, d = self._recover(goal, history, v, shot, shot, prop.thought,
                                                prop.action, frames, fault, step, False)
                        rec.update(diagnosis=d, plan=self._plan_dict(plan))
                        recoveries += 1
                        done_vetoes += 1
                        if plan.feedback and not plan.stop:
                            feedback = plan.feedback
                            steps.append(rec)
                            continue
                end = "declared_done"
                steps.append(rec)
                if hasattr(env, "on_done"):
                    env.on_done(prop.action)
                break

            candidate = (prop.action.get("type") == "click" if cfg.precommit_mode == "clicks" else False) \
                or is_commit(prop.thought, prop.action, cfg.precommit_words)
            if candidate and cfg.click_check == "v4" and choosing_in_open_list(history, prop.action):
                candidate = False                        # choosing an item in a list that just opened
                rec["precommit_skipped"] = "choice in a list the previous step opened"
            if self.prevent and blocks < cfg.precommit_max_blocks and candidate:
                pc = self.monitor.check_commit(goal, shot, prop.thought, prop.action, history)
                rec["precommit"] = pc
                if pc["blocked"]:
                    blocks += 1
                    if cfg.drop_blocked and getattr(self.agent, "turns", None):
                        self.agent.turns.pop()           # the agent does not see its blocked reply again
                        rec["dropped_from_agent_history"] = True
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
                if (v["flagged"] and v["symptom"] == "no_effect" and cfg.second_look_s > 0
                        and recoveries < cfg.max_recoveries):
                    time.sleep(cfg.second_look_s)            # second look: a late update is not a dead click
                    fresh = env.screenshot()
                    rec["first_look"] = v
                    moved = diff_info(after, fresh, cfg.scale)["changed_px"]
                    rec["second_look"] = {"changed_px": moved, "asked": moved >= cfg.no_change_px}
                    if moved >= cfg.no_change_px:            # only a screen that changed can change the answer
                        after = fresh
                        v = self.monitor.recheck(goal, before, after, prop.thought, executed, history)
                    if env.status()["done"]:
                        rec["verdict"] = entry["verdict"] = v
                        end = "env_done"
                        steps.append(rec)
                        self._save(out_dir, after, f"{step + 1:02d}")
                        break
                rec["verdict"] = entry["verdict"] = v
                related = (fault.fired and not fault_attributed and v["flagged"]
                           and fault.fired_step <= step <= fault.fired_step + 2)
                if related:
                    fault_attributed = True
                rec["fault_related"] = related
                if v["flagged"] and recoveries < cfg.max_recoveries:
                    self._save(out_dir, after, f"{step:02d}_flag")
                    if self.strat is not None:                     # the retry baselines have no Strategist
                        self.strat.progress = (step, cfg.max_steps)  # the step budget, for the termination axis
                    plan, d = self._recover(goal, history[:-1], v, before, after, prop.thought,
                                            executed, frames[:-1], fault, step, related)
                    dropped = False
                    if plan.kind == "replan":
                        if (cfg.replan_strikes >= 2 and v.get("symptom") != "loop"
                                and last_flag_step != step - 1):
                            rec["replan_held"] = self._plan_dict(plan)    # one flag is not enough to change course
                            plan, dropped = Plan("none", note="replan held back: a single flag"), True
                        elif cfg.replan_gap and last_replan is not None and step - last_replan < cfg.replan_gap:
                            rec["replan_dropped"] = self._plan_dict(plan)   # no chains of replans
                            plan, dropped = Plan("none", note="replan dropped: too soon after the last one"), True
                        else:
                            last_replan = step
                    rec.update(diagnosis=d, plan=self._plan_dict(plan))
                    if not dropped and not (cfg.n_free and plan.kind == "none"):  # with n_free a cancelled alarm
                        recoveries += 1                                           # uses no fix
                    done_n, blocked, blocks = self._execute3(plan, env, rec, history, step, goal, blocks, before)
                    n_actions += done_n
                    if blocked:
                        plan = Plan(f"{plan.kind}_blocked", feedback=(
                            f"The supervisor stopped before finalizing for you, because: {blocked['problem']} "
                            f"Fix this first: {blocked['fix']}"))
                        rec["plan_after_block"] = self._plan_dict(plan)
                    if done_n:
                        rec_before, after = after, env.screenshot()
                        if cfg.verify_recovery and not env.status()["done"]:
                            last_rec = rec["recovery_executed"][-1] if rec.get("recovery_executed") else executed
                            chk = self.monitor.recheck(goal, rec_before, after, f"supervisor recovery ({plan.kind})",
                                                       last_rec, history)
                            rec["recovery_check"] = chk
                            if cfg.fix_report and not chk["flagged"] and plan.feedback:
                                seen = str((chk.get("judge") or {}).get("change") or "").strip()
                                if seen and seen.lower() != "nothing":   # say what the fix really did
                                    plan.feedback += f" After the fix, the screen shows: {seen}"
                            if chk["flagged"]:
                                plan = self.strat.escalate(plan, chk, executed)
                                rec["escalation"] = self._plan_dict(plan)
                                done_n, blocked, blocks = self._execute3(plan, env, rec, history, step, goal,
                                                                         blocks, before)
                                n_actions += done_n
                                if blocked:
                                    plan = Plan(f"{plan.kind}_blocked", feedback=(
                                        f"The supervisor stopped before finalizing for you, because: "
                                        f"{blocked['problem']} Fix this first: {blocked['fix']}"))
                                if done_n:
                                    after = env.screenshot()
                    elif plan.actions:
                        after = env.screenshot()                 # wait_check found the screen had changed
                    feedback = plan.feedback
                    if cfg.use_axes and any(getattr(self.strat, "axis_note", ("", ""))):
                        pre, post = self.strat.axis_note               # what the axes add to the message
                        feedback = " ".join(x for x in (pre, feedback, post) if x)
                        rec["axis_note"] = {"before": pre, "after": post}
                    if plan.stop:
                        end = "escalated"
                        steps.append(rec)
                        break
                    if env.status()["done"]:
                        end = "env_done"
                        steps.append(rec)
                        break
                if v["flagged"]:
                    last_flag_step = step                    # for the two-strikes rule

            shot = after
            steps.append(rec)
            self._save(out_dir, shot, f"{step + 1:02d}")
            if n_actions >= cfg.max_actions:
                end = "max_actions"
                break

        st = env.status(final=True)
        return {
            "task": task, "seed": seed, "condition": self.condition, "goal": goal, "adr_version": 3,
            "fault": fault.to_dict(), "success": bool(st["success"]), "env_done": bool(st["done"]),
            "raw_reward": st.get("raw"), "reason": st.get("reason"), "end": end,
            "agent_steps": len(steps), "actions": n_actions, "recoveries": recoveries,
            "seconds": round(time.time() - t0, 1), "usage": usage_snapshot(), "steps": steps,
        }

    def _execute3(self, plan, env, rec, history, step, goal, blocks, before):
        """As v2's _execute, plus wait_check: an action marked only_if_unchanged runs only if, after its wait, the
        screen still looks as it did before the agent's action; otherwise it is skipped and the agent is told so."""
        cfg, n = self.cfg, 0
        for pa in plan.actions:
            if pa.get("wait"):
                time.sleep(pa["wait"])
            if pa.get("only_if_unchanged"):
                changed = diff_info(before, env.screenshot(), cfg.scale)["changed_px"] >= cfg.no_change_px
                rec["wait_check"] = "changed" if changed else "unchanged"
                if changed:
                    plan.feedback = LATE
                    continue
                plan.feedback = REPEATED
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
