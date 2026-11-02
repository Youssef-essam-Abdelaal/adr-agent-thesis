"""
Fast tests: no browser, no API key. Model calls are replaced by canned replies.

    .venv/bin/python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adr import runstore  # noqa: E402
from adr.agent import Agent  # noqa: E402
from adr.analyze import kappa, macro_f1, mcnemar, wilson  # noqa: E402
from adr.analyze import rq2 as _rq2  # noqa: E402
from adr.config import Config  # noqa: E402
from adr.env.faults import HELD_OUT, TRUE_TYPE, Fault, corrupt, corrupt_random, randomize  # noqa: E402
from adr.experiment import plan as _plan, spec_id as _spec_id  # noqa: E402
from adr.monitor import Monitor, is_commit, same_action  # noqa: E402
from adr.strategist import Plan, Strategist, evidence_plan, generic_plan, untyped_plan  # noqa: E402
from adr.taxonomy import Taxonomy  # noqa: E402
from adr.vision import diff_info, screen_hash  # noqa: E402
from adr.wrapper import CONDITIONS, ADRWrapper  # noqa: E402


class FakeLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def ask(self, system, text, images=(), tag="misc", **kw):
        self.calls.append((system, text, tag))
        return self.replies.pop(0) if self.replies else {}


def img(color=(255, 255, 255), dot=None):
    im = Image.new("RGB", (480, 630), color)
    if dot:
        ImageDraw.Draw(im).rectangle([dot[0], dot[1], dot[0] + 60, dot[1] + 60], fill=(0, 0, 0))
    return im


CFG = Config()
TAX = Taxonomy(CFG.path(CFG.taxonomy))


class TestTaxonomy(unittest.TestCase):
    def test_one_recovery_per_type(self):
        self.assertEqual(TAX.recovery("I"), "reground")
        self.assertEqual(TAX.recovery("II"), "wait_retry")
        self.assertEqual(TAX.recovery("III", "dismissable_overlay"), "dismiss_retry")
        self.assertEqual(TAX.recovery("III", "access_gate"), "escalate")
        self.assertEqual(TAX.recovery("N"), "none")

    def test_render_has_procedure_and_all_types(self):
        text = TAX.render()
        for t in ["I ", "II ", "III ", "IV ", "V ", "VI ", "N "]:
            self.assertIn(f"- {t}", text)
        self.assertIn("DECISION PROCEDURE", text)

    def test_every_cuadebug_code_maps_once(self):
        seen = {}
        for t in TAX.data["types"]:
            for c in t.get("cuadebug", []):
                self.assertNotIn(c, seen, f"{c} mapped twice")
                seen[c] = t["id"]


class TestFaults(unittest.TestCase):
    def test_corrupt_always_changes_value(self):
        for v in range(6):
            for s in ["Kanesha", "gyJ8Q", "12/31/2016", "ab"]:
                self.assertNotEqual(corrupt(s, v), s)

    def test_eligibility(self):
        self.assertTrue(Fault("wrong_value").eligible({"type": "type", "x": 1, "y": 1, "text": "abc"}))
        self.assertFalse(Fault("wrong_value").eligible({"type": "click", "x": 1, "y": 1}))
        self.assertTrue(Fault("stagnation").eligible({"type": "key", "key": "Enter"}))
        self.assertFalse(Fault("occlusion").eligible({"type": "key", "key": "Enter"}))
        self.assertEqual(Fault("occlusion").true_type, "III")


class TestVision(unittest.TestCase):
    def test_diff_and_hash(self):
        a, b = img(), img(dot=(100, 100))
        self.assertEqual(diff_info(a, a, 3)["changed_px"], 0)
        d = diff_info(a, b, 3)
        self.assertGreater(d["changed_px"], 100)
        self.assertIsNotNone(d["bbox"])
        self.assertEqual(screen_hash(a), screen_hash(a.copy()))


class TestMonitor(unittest.TestCase):
    def make(self, *replies, **over):
        cfg = Config()
        for k, v in over.items():
            setattr(cfg, k, v)
        m = Monitor(cfg)
        m.llm = FakeLLM(*replies)
        return m

    def test_judge_flag(self):
        m = self.make({"ok": False, "symptom": "wrong_effect", "confidence": 0.9, "evidence": "x"})
        v = m.check("task", img(), img(dot=(10, 10)), "t", {"type": "click", "x": 5, "y": 5}, [{"action": {}}])
        self.assertTrue(v["flagged"])
        self.assertEqual(v["symptom"], "wrong_effect")

    def test_low_confidence_not_flagged(self):
        m = self.make({"ok": False, "symptom": "no_effect", "confidence": 0.3})
        v = m.check("task", img(), img(), "t", {"type": "click", "x": 5, "y": 5}, [])
        self.assertFalse(v["flagged"])

    def test_cheap_mode_skips_model_when_screen_changes(self):
        m = self.make(judge_policy="suspicious_only")
        v = m.check("task", img(), img(dot=(10, 10)), "t", {"type": "click", "x": 5, "y": 5}, [])
        self.assertFalse(v["flagged"])
        self.assertEqual(m.llm.calls, [])

    def test_repeating_a_click_with_progress_is_not_a_loop(self):
        m = self.make(*[{"ok": True}] * 3)
        a = {"type": "click", "x": 48, "y": 242}
        shots = [img(), img(dot=(0, 0)), img(dot=(200, 0)), img(dot=(0, 300))]
        for i in range(3):
            v = m.check("task", shots[i], shots[i + 1], "t", a, [])
        self.assertFalse(v["flagged"])

    def test_dead_repeated_click_is_a_loop(self):
        m = self.make(*[{"ok": True}] * 3)
        a = {"type": "click", "x": 48, "y": 242}
        for _ in range(2):
            v = m.check("task", img(), img(), "t", a, [])
        self.assertTrue(v["flagged"])
        self.assertEqual(v["symptom"], "loop")

    def test_same_action(self):
        self.assertTrue(same_action({"type": "click", "x": 10, "y": 10}, {"type": "click", "x": 15, "y": 12}))
        self.assertFalse(same_action({"type": "click", "x": 10, "y": 10}, {"type": "click", "x": 60, "y": 10}))


class TestPrevention(unittest.TestCase):
    def test_commit_detection(self):
        w = Config().precommit_words
        self.assertTrue(is_commit("Now I click Submit.", {"type": "click", "x": 1, "y": 1}, w))
        self.assertTrue(is_commit("", {"type": "key", "key": "Enter"}, w))
        self.assertFalse(is_commit("Type the username", {"type": "type", "x": 1, "y": 1, "text": "a"}, w))
        self.assertFalse(is_commit("Check the box for Kiwi", {"type": "click", "x": 1, "y": 1}, w))
        self.assertFalse(is_commit("Look at the list", {"type": "click", "x": 1, "y": 1}, w))  # 'ok' inside 'look'

    def test_commit_check_blocks_only_when_sure(self):
        m = Monitor(Config())
        m.llm = FakeLLM({"ready": False, "problem": "Banana is checked", "fix": "uncheck it", "confidence": 0.9},
                        {"ready": False, "confidence": 0.2})
        a = {"type": "click", "x": 5, "y": 5}
        self.assertTrue(m.check_commit("t", img(), "submit", a, [])["blocked"])
        self.assertFalse(m.check_commit("t", img(), "submit", a, [])["blocked"])


class TestStrategist(unittest.TestCase):
    def make(self, *replies):
        s = Strategist(Config(), TAX)
        s.llm = FakeLLM(*replies)
        return s

    def test_reground_repeats_action_on_found_target(self):
        s = self.make({"found": True, "x": 100, "y": 200})
        p = s.plan({"type": "I", "intended_target": "Submit button"}, {"symptom": "no_effect"},
                   "task", {"type": "click", "x": 100, "y": 266}, img())
        self.assertEqual(p.kind, "reground")
        self.assertEqual(p.actions[0]["action"], {"type": "click", "x": 100, "y": 200})

    def test_wait_retry_and_escalation_to_replan(self):
        s = self.make()
        last = {"type": "click", "x": 1, "y": 1}
        p1 = s.plan({"type": "II"}, {"symptom": "no_effect"}, "t", last, img())
        self.assertEqual(p1.kind, "wait_retry")
        s.plan({"type": "II"}, {"symptom": "no_effect"}, "t", last, img())
        p3 = s.plan({"type": "II"}, {"symptom": "no_effect"}, "t", last, img())
        self.assertEqual(p3.kind, "replan")

    def test_dismiss_then_repeat(self):
        s = self.make({"found": True, "x": 400, "y": 300})
        last = {"type": "type", "x": 10, "y": 10, "text": "abc"}
        p = s.plan({"type": "III", "subtype": "dismissable_overlay"}, {"symptom": "blocked"}, "t", last, img())
        self.assertEqual([a["action"]["type"] for a in p.actions], ["click", "type"])

    def test_access_gate_escalates(self):
        p = self.make().plan({"type": "III", "subtype": "access_gate"}, {"symptom": "blocked"}, "t", {}, img())
        self.assertTrue(p.stop)

    def test_fix_value_retypes(self):
        p = self.make().plan({"type": "IV", "correct_value": "Kanesha"}, {"symptom": "wrong_effect"}, "t",
                             {"type": "type", "x": 5, "y": 5, "text": "Kanesh"}, img())
        self.assertEqual(p.actions[0]["action"]["text"], "Kanesha")

    def test_false_done_always_vetoes_even_if_diagnosed_no_failure(self):
        p = self.make().plan({"type": "N"}, {"symptom": "false_done", "unmet": ["field filled"]}, "t", {}, img())
        self.assertEqual(p.kind, "veto_done")
        self.assertIn("NOT complete", p.feedback)

    def test_no_failure_suppresses(self):
        p = self.make().plan({"type": "N"}, {"symptom": "loop"}, "t", {"type": "click", "x": 1, "y": 1}, img())
        self.assertEqual(p.kind, "none")
        self.assertFalse(p.actions)

    def test_irreversible_is_never_retried(self):
        p = self.make().plan({"type": "II", "recoverability": "irreversible"}, {"symptom": "no_effect"}, "t",
                             {"type": "click", "x": 1, "y": 1}, img())
        self.assertFalse(p.actions)

    def test_fix_target_undoes_and_selects(self):
        s = self.make({"found": True, "x": 50, "y": 300})
        p = s.plan({"type": "IV", "correct_target": "Kiwi checkbox", "undo_by_reclick": True},
                   {"symptom": "wrong_effect"}, "t", {"type": "click", "x": 50, "y": 240}, img())
        self.assertEqual([a["action"]["y"] for a in p.actions], [240, 300])

    def test_escalation_ladder(self):
        s = self.make()
        last = {"type": "click", "x": 1, "y": 1}
        p = s.escalate(s.plan({"type": "II"}, {"symptom": "no_effect"}, "t", last, img()), {"evidence": "e"}, last)
        self.assertEqual(p.kind, "wait_retry_longer")
        self.assertEqual(p.actions[0]["wait"], Config().wait_s * 2)
        p = s.escalate(Plan("reground"), {"evidence": "still missed"}, last)
        self.assertFalse(p.actions)
        self.assertIn("did not work", p.feedback)

    def test_uncertain_diagnosis_falls_back_to_generic(self):
        p = self.make().plan({"type": "IV", "uncertain": True}, {"symptom": "wrong_effect"}, "t",
                             {"type": "click", "x": 1, "y": 1}, img())
        self.assertEqual(p.kind, "generic_fallback")
        self.assertFalse(p.actions)

    def test_generic_baseline_is_one_message(self):
        p = generic_plan({"symptom": "wrong_effect"})
        self.assertFalse(p.actions)
        self.assertIn("try again", p.feedback)


class TestVoting(unittest.TestCase):
    def make(self, *replies):
        from adr.diagnostician import Diagnostician
        d = Diagnostician(Config(), TAX)
        d.llm = FakeLLM(*replies)
        return d

    def args(self):
        v = {"symptom": "wrong_effect", "signals": {}, "judge": {}}
        return ("task", [], v, img(), img(), "t", {"type": "click", "x": 5, "y": 5}, [])

    def test_confident_diagnosis_is_not_revoted(self):
        d = self.make({"type": "I", "confidence": 0.95})
        self.assertEqual(d.diagnose(*self.args())["type"], "I")
        self.assertEqual(len(d.llm.calls), 1)

    def test_majority_wins(self):
        d = self.make({"type": "VI", "confidence": 0.4}, {"type": "I", "confidence": 0.8}, {"type": "I", "confidence": 0.7})
        res = d.diagnose(*self.args())
        self.assertEqual(res["type"], "I")
        self.assertEqual(res["votes"], ["VI", "I", "I"])

    def test_no_majority_is_uncertain(self):
        d = self.make({"type": "VI", "confidence": 0.4}, {"type": "I", "confidence": 0.5}, {"type": "IV", "confidence": 0.5})
        self.assertTrue(d.diagnose(*self.args()).get("uncertain"))


class TestVersions(unittest.TestCase):
    def test_v1_switches_off_v2_parts(self):
        c = Config.load(None, ["adr_version=1"])
        self.assertEqual((c.precommit, c.verify_recovery, c.votes), (0, 0, 1))
        self.assertTrue(c.taxonomy.endswith("taxonomy_v4.0.json"))

    def test_v1_wrong_target_is_a_hint_only(self):
        s = Strategist(Config.load(None, ["adr_version=1"]), TAX)
        s.llm = FakeLLM({"found": True, "x": 50, "y": 300})
        p = s.plan({"type": "IV", "correct_target": "Kiwi", "undo_by_reclick": True}, {"symptom": "wrong_effect"},
                   "t", {"type": "click", "x": 50, "y": 240}, img())
        self.assertEqual(p.kind, "fix_target")
        self.assertFalse(p.actions)

    def test_both_taxonomy_files_load(self):
        for f in ("taxonomy/taxonomy_v4.0.json", "taxonomy/taxonomy_v4.json"):
            self.assertIn("I", Taxonomy(Config().path(f)).types)


class TestAgentParsing(unittest.TestCase):
    def test_clamps_and_rejects(self):
        p = Agent._parse({"thought": "t", "action": {"type": "click", "x": 9999, "y": -5}}, 480, 630)
        self.assertEqual((p.action["x"], p.action["y"]), (479, 0))
        self.assertEqual(Agent._parse({"action": {"type": "fly"}}, 480, 630).action, {"type": "wait"})
        self.assertEqual(Agent._parse({"action": {"type": "click"}}, 480, 630).action, {"type": "wait"})


class TestStats(unittest.TestCase):
    def test_mcnemar(self):
        _, _, p = mcnemar([(True, False)] * 10)
        self.assertLess(p, 0.01)
        self.assertEqual(mcnemar([(True, True)] * 5)[2], 1.0)

    def test_kappa_and_f1(self):
        self.assertAlmostEqual(kappa(list("ABAB"), list("ABAB")), 1.0)
        self.assertAlmostEqual(macro_f1(list("AB"), list("AB"), ["A", "B"]), 1.0)
        lo, hi = wilson(5, 10)
        self.assertTrue(lo < 0.5 < hi)


class TestNewFaults(unittest.TestCase):
    def test_randomize_is_reproducible_and_varied(self):
        a = randomize("stagnation", "login-user", 6)
        self.assertEqual(a, randomize("stagnation", "login-user", 6))
        values = {randomize("stagnation", "login-user", s)["ms"] for s in range(6, 30)}
        self.assertGreater(len(values), 10)
        self.assertTrue(all(1000 <= v <= 8000 for v in values))
        g = randomize("grounding", "click-option", 7)
        self.assertTrue(10 <= (g["dx"] ** 2 + g["dy"] ** 2) ** 0.5 <= 40)

    def test_held_out_kinds_have_types(self):
        self.assertEqual([TRUE_TYPE[k] for k in HELD_OUT], ["II", "I"])
        self.assertTrue(Fault("loading").eligible({"type": "key", "key": "Enter"}))
        self.assertFalse(Fault("layout_shift").eligible({"type": "key", "key": "Enter"}))

    def test_random_typos_always_change_value(self):
        for v in range(12):
            for s in ["Kanesha", "gyJ8Q", "12/31/2016", "ab", "Zurich"]:
                self.assertNotEqual(corrupt_random(s, v), s)


class TestNewPlans(unittest.TestCase):
    def test_evidence_baseline_carries_evidence_but_no_type(self):
        p = evidence_plan({"symptom": "blocked", "evidence": "a cookie banner covers the form"})
        self.assertIn("cookie banner", p.feedback)
        self.assertFalse(p.actions)
        self.assertNotIn("III", p.feedback)

    def test_untyped_plan(self):
        p = untyped_plan({"what_went_wrong": "it missed", "advice": "click Submit again"}, {"symptom": "no_effect"})
        self.assertIn("click Submit again", p.feedback)
        self.assertFalse(p.actions)
        self.assertEqual(untyped_plan({"no_failure": True}, {"symptom": "loop"}).kind, "untyped_none")

    def test_hints_mode_never_acts_or_calls_a_model(self):
        s = Strategist(Config(), TAX)
        s.llm = FakeLLM()
        last = {"type": "click", "x": 5, "y": 5}
        for d, v in [({"type": "I", "intended_target": "Submit"}, {"symptom": "no_effect"}),
                     ({"type": "II"}, {"symptom": "no_effect"}),
                     ({"type": "III", "subtype": "dismissable_overlay", "dismiss_target": "Accept"}, {"symptom": "blocked"}),
                     ({"type": "IV", "correct_target": "Kiwi"}, {"symptom": "wrong_effect"})]:
            p = s.plan(d, v, "t", last, img(), hints_only=True)
            self.assertFalse(p.actions, p.kind)
            self.assertTrue(p.feedback)
        self.assertEqual(s.llm.calls, [])


class FakeEnv:
    """Two-step task: click A, then Submit. The screen changes colour after each click."""

    def __init__(self, fault_kind="none"):
        self.fault_kind = fault_kind

    def reset(self, task, seed, fault):
        self.fault, self.clicks, self.done, self.ok = fault, [], False, False
        return "Click A, then Submit."

    def screenshot(self):
        return img((255, 255 - 60 * len(self.clicks), 255))

    def execute(self, action, source="agent", step=-1):
        fired = False
        a = dict(action)
        if source == "agent" and self.fault.kind == "stagnation" and not self.fault.fired:
            self.fault.fired, self.fault.fired_step, fired = True, step, True
            return {"executed": a, "fault_fired": True, "error": ""}      # swallowed
        if a.get("type") == "click":
            self.clicks.append((a["x"], a["y"]))
            if a["y"] > 400:
                self.done, self.ok = True, len(self.clicks) >= 2
        return {"executed": a, "fault_fired": fired, "error": ""}

    def status(self, final=False):
        return {"done": self.done, "raw": 1 if self.ok else 0, "reason": None, "success": self.done and self.ok}


class TestWrapperLoop(unittest.TestCase):
    """Every condition runs end to end on the fake environment with scripted model replies."""

    def run_condition(self, condition, fault_kind):
        cfg = Config()
        w = ADRWrapper(cfg, condition)
        agent_replies = [{"thought": "click A", "action": {"type": "click", "x": 100, "y": 100}},
                         {"thought": "click A again", "action": {"type": "click", "x": 100, "y": 100}},
                         {"thought": "click Submit", "action": {"type": "click", "x": 200, "y": 500}}] * 3
        w.agent.llm = FakeLLM(*agent_replies)
        judge = {"ok": False, "symptom": "no_effect", "confidence": 0.9, "evidence": "nothing changed"}
        w.monitor.llm = FakeLLM(*([{"ok": True, "is_final": False, "ready": True, "confidence": 0.9}] + [judge]
                                  + [{"ok": True, "is_final": False, "ready": True, "confidence": 0.9}] * 20))
        if w.diag:
            w.diag.llm = FakeLLM(*[{"type": "II", "confidence": 0.95, "what_went_wrong": "dead click",
                                    "advice": "click A again"}] * 5)
        if w.strat:
            w.strat.llm = FakeLLM(*[{"found": True, "x": 100, "y": 100}] * 5)
        return w.run(FakeEnv(fault_kind), "fake", 1, Fault(fault_kind))

    def test_all_conditions_complete(self):
        for c in [c for c in CONDITIONS if c != "base"]:
            r = self.run_condition(c, "stagnation")
            self.assertIn(r["end"], ("env_done", "max_steps", "max_actions", "declared_done"), c)
            self.assertTrue(r["steps"], c)
            plans = [s["plan"]["kind"] for s in r["steps"] if s.get("plan")]
            if c in ("adr_hints", "adr_untyped", "retry", "retry_evidence"):
                self.assertFalse(any(s.get("recovery_executed") for s in r["steps"]), c)
            if c == "adr":
                self.assertIn("wait_retry", plans)

    def test_done_check_switch(self):
        for on in (1, 0):
            cfg = Config()
            cfg.done_check = on
            w = ADRWrapper(cfg, "adr")
            w.agent.llm = FakeLLM({"thought": "all done", "action": {"type": "done"}, "done": True})
            w.monitor.llm = FakeLLM({"complete": True, "confidence": 0.9, "criteria": []})
            r = w.run(FakeEnv(), "fake", 1, Fault("none"))
            self.assertEqual(r["end"], "declared_done")
            self.assertEqual("verdict" in r["steps"][0], bool(on))


class TestRunStore(unittest.TestCase):
    def test_fingerprint_follows_results_not_workers(self):
        c = Config().to_dict()
        fp = runstore.fingerprint(runstore.LIVE_FILES, c)
        self.assertEqual(fp, runstore.fingerprint(runstore.LIVE_FILES, dict(c, workers=3)))
        self.assertNotEqual(fp, runstore.fingerprint(runstore.LIVE_FILES, dict(c, max_recoveries=2)))
        self.assertNotEqual(fp, runstore.fingerprint(runstore.LIVE_FILES, c, {"dataset": "other"}))

    def test_atomic_write_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            runstore.atomic_write(p, '{"a": 1}')
            self.assertEqual(runstore.read_result(p), {"a": 1})
            p.write_text('{"a": ')                       # a half-written file from a hard crash
            self.assertIsNone(runstore.read_result(p))     # treated as missing, so it is redone

    def test_case_store_resume(self):
        with tempfile.TemporaryDirectory() as d:
            s = runstore.CaseStore(Path(d))
            s.save("claude/chrome/1", {"id": 1})
            self.assertTrue(s.has("claude/chrome/1"))
            self.assertFalse(s.has("claude/chrome/2"))
            self.assertEqual(len(runstore.CaseStore(Path(d)).all()), 1)

    def test_plan_skips_done_and_retries_crashes_once(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            specs = [{"task": "t", "seed": s, "fault": "none", "condition": "base", "at": 1} for s in range(4)]
            ep = out / "episodes"
            runstore.atomic_write(ep / f"{_spec_id(specs[0])}.json", json.dumps({"success": True}))
            runstore.atomic_write(ep / f"{_spec_id(specs[1])}.json", json.dumps({"error": "boom"}))
            runstore.atomic_write(ep / f"{_spec_id(specs[2])}.json", json.dumps({"error": "boom", "crash_retry": True}))
            done, todo, retry = _plan(specs, out)
            self.assertEqual((len(done), len(todo), len(retry)), (2, 1, 1))


class TestPartialRuns(unittest.TestCase):
    @staticmethod
    def ep(cond, task, seed, ok):
        return {"condition": cond, "success": ok, "spec": {"task": task, "seed": seed, "fault": "grounding"},
                "fault": {"fired": True}, "steps": [], "usage": {}}

    def test_registered_tests_use_complete_blocks_only(self):
        eps = []
        for seed in range(12):
            for c in ("retry_evidence", "adr_untyped", "adr_hints", "adr"):
                if seed >= 10 and c == "adr":
                    continue                               # run stopped before these finished
                eps.append(self.ep(c, f"task{seed % 3}", seed, c == "adr" or seed % 2 == 0))
        res, _ = _rq2(eps)
        self.assertTrue(res["completeness"]["partial"])
        self.assertEqual(res["completeness"]["complete_blocks"], 10)
        self.assertTrue(all(r["n_pairs"] == 10 for r in res["registered"]))


class TestOpenModels(unittest.TestCase):
    """UI-TARS adapter (adr/agent_uitars.py) and where models run (adr/llm.py)."""

    def test_smart_resize_matches_osworld_log(self):
        from adr.agent_uitars import smart_resize
        self.assertEqual(smart_resize(1080, 1920), (1092, 1932))      # OSWorld screen
        rh, rw = smart_resize(630, 480)                               # MiniWoB screenshot
        self.assertEqual((rh % 28, rw % 28), (0, 0))

    def test_click_scaled_back_like_pyautogui(self):
        from adr.agent_uitars import parse
        th, a = parse("Thought: close the pop-up.\nAction: click(start_box='(1134,184)')", 1920, 1080)
        self.assertEqual(th, "close the pop-up.")
        self.assertEqual((a["type"], a["x"], a["y"]), ("click", 1126, 181))   # OSWorld: pyautogui (1126.96, 181.98)

    def test_coordinate_formats(self):
        from adr.agent_uitars import parse
        for call in ("click(start_box='<|box_start|>(238,308)<|box_end|>')", "click(point='<point>238 308</point>')",
                     "click(start_box='[228,298,248,318]')"):
            _, a = parse(f"Thought: x\nAction: {call}", 476, 616)
            self.assertEqual((a["type"], a["x"], a["y"]), ("click", 238, 308), call)
        _, a = parse("Thought: x\nAction: left_double(start_box='(10,20)')", 476, 616)
        self.assertTrue(a["double"])

    def test_other_actions(self):
        from adr.agent_uitars import parse
        _, a = parse("Thought: x\nAction: type(content='Leonie\\n')", 480, 630)
        self.assertEqual(a, {"type": "type", "text": "Leonie", "submit": True})
        _, a = parse("Thought: x\nAction: type(content='it\\'s')", 480, 630)
        self.assertEqual(a, {"type": "type", "text": "it's"})
        self.assertEqual(parse("Action: hotkey(key='enter')", 480, 630)[1], {"type": "key", "key": "Enter"})
        self.assertEqual(parse("Action: hotkey(key='ctrl a')", 480, 630)[1], {"type": "key", "key": "Control+a"})
        _, a = parse("Action: scroll(start_box='(100,200)', direction='down')", 480, 630)
        self.assertEqual((a["type"], a["dy"] > 0), ("scroll", True))
        self.assertEqual(parse("Action: finished(content='done')", 480, 630)[1], {"type": "done"})
        self.assertEqual(parse("Action: drag(start_box='(1,1)', end_box='(5,5)')", 480, 630)[1], {"type": "wait"})
        self.assertEqual(parse("no idea", 480, 630)[1], {"type": "wait"})

    def test_agent_keeps_history_and_resets(self):
        import os
        from adr.agent_uitars import UITarsAgent
        cfg = Config.load(str(Path(__file__).resolve().parent.parent / "configs" / "open_models.json"))
        os.environ["ADR_FAKE_LLM"] = "1"
        try:
            ag = UITarsAgent(cfg)
            p1 = ag.act(img(), "Click OK", [], None)
            ag.act(img(), "Click OK", [{"action": p1.action, "thought": p1.thought}], "try again")
            self.assertEqual(len(ag.turns), 2)
            ag.act(img(), "New task", [], None)
            self.assertEqual(len(ag.turns), 1)
        finally:
            os.environ.pop("ADR_FAKE_LLM", None)

    def test_no_openai_guard(self):
        import os
        from adr.llm import LLM
        os.environ["ADR_NO_OPENAI"] = "1"
        try:
            with self.assertRaises(RuntimeError):
                LLM("gpt-x").client
            self.assertIsNotNone(LLM("ui-tars", base_url="http://127.0.0.1:8000/v1").client)
        finally:
            os.environ.pop("ADR_NO_OPENAI", None)

    def test_roles_follow_config(self):
        from adr.llm import LLM
        cfg = Config.load(str(Path(__file__).resolve().parent.parent / "configs" / "open_models.json"))
        j = LLM.for_role(cfg, "judge")
        self.assertTrue(j.ollama)
        self.assertEqual((j.model, j.ctx), ("qwen3-vl:32b-instruct-q4_K_M", 8192))
        self.assertEqual(LLM.for_role(Config(), "judge").base_url, "")      # default: OpenAI, unchanged


class TestOSWorldActions(unittest.TestCase):
    """Our actions as pyautogui commands for the OSWorld VM (adr/env/osworld.py); no VM needed."""

    def test_commands(self):
        from adr.env.osworld import to_pyautogui
        self.assertEqual(to_pyautogui({"type": "click", "x": 10, "y": 20}), "pyautogui.click(10, 20)")
        self.assertEqual(to_pyautogui({"type": "click", "x": 10, "y": 20, "double": True}), "pyautogui.doubleClick(10, 20)")
        self.assertEqual(to_pyautogui({"type": "type", "text": "Boston", "submit": True}),
                         'pyautogui.typewrite("Boston", interval=0.02); pyautogui.press(\'enter\')')
        agent = to_pyautogui({"type": "type", "x": 5, "y": 6, "text": "a"}, "agent")
        fix = to_pyautogui({"type": "type", "x": 5, "y": 6, "text": "a"}, "recovery")
        self.assertNotIn("hotkey", agent)                 # the agent types where its cursor is
        self.assertNotIn("click", agent)                  # no click first (it broke typing in the address bar)
        self.assertEqual(to_pyautogui({"type": "click", "x": 1, "y": 2, "button": "right"}), "pyautogui.rightClick(1, 2)")
        self.assertIn("pyautogui.hotkey('ctrl', 'a')", fix)  # a retype by the wrapper replaces the field
        self.assertEqual(to_pyautogui({"type": "key", "key": "Control+a"}), 'pyautogui.hotkey("ctrl", "a")')
        self.assertEqual(to_pyautogui({"type": "key", "key": "Enter"}), 'pyautogui.press("enter")')
        self.assertEqual(to_pyautogui({"type": "scroll", "x": 1, "y": 2, "dy": 300}), "pyautogui.scroll(-5, x=1, y=2)")
        self.assertEqual(to_pyautogui({"type": "wait"}), "WAIT")

    def test_call_user_is_infeasible(self):
        from adr.agent_uitars import parse
        self.assertEqual(parse("Action: call_user()", 1920, 1080)[1], {"type": "done", "infeasible": True})

    def test_right_click_is_marked(self):
        from adr.agent_uitars import parse
        a = parse("Action: right_single(start_box='<|box_start|>(500,300)<|box_end|>')", 1920, 1080)[1]
        self.assertEqual((a["type"], a.get("button")), ("click", "right"))
        self.assertNotIn("button", parse("Action: click(start_box='<|box_start|>(500,300)<|box_end|>')", 1920, 1080)[1])


class TestJudgeBench(unittest.TestCase):
    """Judge benchmark adapters (adr/judge_bench.py): parsing and scoring only, no data or model needed."""

    def test_reflection_actions(self):
        from adr.judge_bench import reflection_action
        self.assertEqual(reflection_action("CLICK[[284, 841]]", (1440, 3120)), {"type": "click", "x": 409, "y": 2624})
        s = reflection_action("SCROLL[[500, 800, 500, 200]]", (1000, 2000))
        self.assertEqual((s["x"], s["y"]), (500, 1600))
        self.assertTrue(s["type"].startswith("swipe from (500, 1600) to (500, 400)"))

    def test_critic_actions(self):
        from adr.judge_bench import critic_action
        self.assertEqual(critic_action("Action: {'action': 'click', 'coordinate': [12, 34]}"),
                         {"type": "click", "x": 12, "y": 34})
        self.assertEqual(critic_action('Action: {"name": "computer_use", "arguments": {"action": "key", "keys": ["enter"]}}'),
                         {"type": "key", "key": "enter"})
        self.assertEqual(critic_action("Action: {'action': 'terminate', 'status': 'success'}")["type"], "done")
        self.assertEqual(critic_action("Action: {'action': 'type', 'text': 'hi'}"), {"type": "type 'hi'"})

    def test_cua_actions(self):
        from adr.judge_bench import cua_action
        a = cua_action("pyautogui.click(x=0.5, y=0.25)", (1920, 1080), None)
        self.assertEqual((a["type"], a["x"], a["y"]), ("click", 960, 270))
        t = cua_action("pyautogui.write(message='cleanmymac')\npyautogui.press('enter')", (1920, 1080), (960, 270))
        self.assertEqual((t["type"], t["text"], t["x"]), ("type", "cleanmymac", 960))
        self.assertEqual(cua_action("computer.terminate(status='success')", (10, 10), None)["type"], "done")
        self.assertEqual(cua_action("pyautogui.press('enter')", (10, 10), None)["key"], "enter")
        self.assertEqual(cua_action(None, (10, 10), None)["type"], "none")
        self.assertEqual(cua_action("CLICK [[500,500]]", (1000, 2000), None), {"type": "click", "x": 500, "y": 1000})
        self.assertEqual(cua_action("mobile_use.click(coordinate=[544.3, 1731.2])", (1440, 3200), None),
                         {"type": "click", "x": 544, "y": 1731})
        self.assertEqual(cua_action("mouse_click(976.6, 681.8, 'left')", (1280, 720), None),
                         {"type": "click", "x": 977, "y": 682})
        self.assertEqual(cua_action("fill('145', 'MIT')", (1280, 720), None), {"type": "fill('145', 'MIT')"})
        self.assertNotIn("x", cua_action("scroll(0, 500)", (1280, 720), None))
        for done in ('send_msg_to_user("found it")', "COMPLETE", "INCOMPLETE", "mobile_use.terminate(status='success')"):
            self.assertEqual(cua_action(done, (10, 10), None)["type"], "done", done)
        self.assertEqual(cua_action("keyboard_type('Miami, FL')", (1280, 720), (5, 6)),
                         {"type": "type", "x": 5, "y": 6, "text": "Miami, FL"})
        self.assertEqual(cua_action("TEXT 'Healthy lunch'", (10, 10), None), {"type": "type 'Healthy lunch'"})

    def test_scores(self):
        from adr.judge_bench import prf, score
        m = prf([True, True, False, False], [True, False, True, False])
        self.assertEqual((m["precision"], m["recall"], m["accuracy"]), (0.5, 0.5, 0.5))
        s = score("gui_reflection", [{"correct_pos": True, "correct_neg": False}, {"correct_pos": True, "correct_neg": True}])
        self.assertEqual(s["accuracy"], 0.75)
        c = score("cuastepbench", [{"benchmark": "x", "labels": ["good", "bad", "bad"], "pred_bad": [False, True, False],
                                    "routes": ["judge"] * 3, "success": False, "pred_success": False}])
        self.assertEqual((c["step_error"]["precision"], c["step_error"]["recall"]), (1.0, 0.5))


if __name__ == "__main__":
    unittest.main()
