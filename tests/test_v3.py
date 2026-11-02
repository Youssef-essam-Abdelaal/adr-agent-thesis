"""
ADR v3 (adr/v3/): fast tests with canned model replies; no browser, no API key.

    .venv/bin/python -m unittest tests.test_v3 -v
"""
import sys
import unittest
from pathlib import Path

from PIL import ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_core import FakeEnv, FakeLLM, img  # noqa: E402

from adr.config import Config  # noqa: E402
from adr.env.faults import Fault  # noqa: E402
from adr.strategist import Plan  # noqa: E402
from adr.v3.diagnostician import SYSTEM_V3, DiagnosticianV3  # noqa: E402
from adr.v3.monitor import (DONE_SYSTEM_V4, DONE_SYSTEM_V5, PRECOMMIT_READ_SYSTEM, PRECOMMIT_V4_SYSTEM,  # noqa: E402
                            PRECOMMIT_V5_SYSTEM, MonitorV3, choosing_in_open_list)
from adr.v3.strategist import LATE, REPEATED, RETYPED, StrategistV3  # noqa: E402
from adr.v3.wrapper import ADRWrapperV3  # noqa: E402

CLICK = {"type": "click", "x": 110, "y": 110}
TYPE = {"type": "type", "text": "Kanesha", "x": 235, "y": 204}


def monitor(*replies):
    m = MonitorV3(Config.load(None, ["adr_version=3"]))
    m.llm = FakeLLM(*replies)
    return m


class TestMonitorV3(unittest.TestCase):
    def test_closer_to_the_task_is_not_a_failure(self):
        m = monitor({"ok": False, "symptom": "wrong_effect", "task_progress": "closer", "intent_met": False,
                     "confidence": 0.9})
        v = m.check("task", img(), img(dot=(100, 100)), "open the list", CLICK, [{"action": CLICK}])
        self.assertFalse(v["flagged"])
        self.assertEqual(v["judge"]["overruled"], "task_progress_closer")

    def test_further_from_the_task_is_flagged(self):
        m = monitor({"ok": False, "symptom": "wrong_effect", "task_progress": "further", "confidence": 0.9})
        v = m.check("task", img(), img(dot=(100, 100)), "t", CLICK, [{"action": CLICK}])
        self.assertTrue(v["flagged"])

    def test_no_effect_is_not_asked_again(self):
        m = monitor({"ok": False, "symptom": "no_effect", "confidence": 0.9}, {"ok": True})
        v = m.check("task", img(), img(dot=(100, 100)), "t", CLICK, [{"action": CLICK}])
        self.assertEqual(len(m.llm.calls), 1)          # no re-ask, even though the pixels changed
        self.assertTrue(v["flagged"])
        self.assertIn("Judge against the TASK", m.llm.calls[0][0])

    def test_recheck_leaves_the_loop_state_alone(self):
        m = monitor({"ok": True}, {"ok": True})
        m.recheck("task", img(), img(), "t", CLICK, [])
        self.assertEqual(m.actions, [])

    def test_done_check_vetoes_only_what_is_visibly_not_met(self):
        unsure = monitor({"criteria": [{"criterion": "the closed tab is back", "status": "cannot_tell"}],
                          "complete": False, "confidence": 0.8})
        v = unsure.check_done("reopen the closed tab", img(), "done", first=img(dot=(0, 0)), history=[])
        self.assertFalse(v["flagged"])
        self.assertIn("STARTED", unsure.llm.calls[0][1])
        wrong = monitor({"criteria": [{"criterion": "folder is Bookmarks bar", "status": "not_met"}],
                         "complete": False, "confidence": 0.8})
        v = wrong.check_done("save to the bookmarks bar", img(), "done")
        self.assertTrue(v["flagged"])
        self.assertEqual(v["unmet"], ["folder is Bookmarks bar"])

    def test_click_check_is_told_the_action_has_not_happened(self):
        m = monitor({"is_final": True, "ready": False, "problem": "folder is wrong", "confidence": 0.9})
        pc = m.check_commit("task", img(), "save", CLICK, [])
        self.assertTrue(pc["blocked"])
        self.assertIn("NOT done yet", m.llm.calls[0][1])

    def test_click_check_v2_in_v3(self):
        """click_check=v2 runs v2's click check (its prompt, one marked screenshot) inside v3."""
        from adr.monitor import PRECOMMIT_SYSTEM as V2_PRECOMMIT
        seen = []

        class Spy(FakeLLM):
            def ask(self, system, text, images=(), tag="misc", **kw):
                seen.append(len(list(images)))
                return super().ask(system, text, images, tag, **kw)
        m = MonitorV3(Config.load(None, ["adr_version=3", "click_check=v2"]))
        m.llm = Spy({"is_final": True, "ready": False, "problem": "the field is empty", "confidence": 0.9})
        pc = m.check_commit("task", img(), "submit", CLICK, [])
        self.assertTrue(pc["blocked"])
        self.assertEqual(m.llm.calls[0][0], V2_PRECOMMIT)
        self.assertNotIn("NOT done yet", m.llm.calls[0][1])
        self.assertEqual(seen, [1])

    def test_judge_without_the_box(self):
        """Ablation judge_box=0: the AFTER screenshot is unmarked and the judge is not told about a box."""
        seen = []

        class Spy(FakeLLM):
            def ask(self, system, text, images=(), tag="misc", **kw):
                seen.append(list(images))
                return super().ask(system, text, images, tag, **kw)
        before, after = img(), img(dot=(200, 300))
        for flag in (1, 0):
            m = MonitorV3(Config.load(None, ["adr_version=3", f"judge_box={flag}"]))
            m.llm = Spy({"ok": True, "symptom": "none", "confidence": 0.9})
            m.check("task", before, after, "click A", CLICK, [])
            system, text, _ = m.llm.calls[0]
            self.assertEqual("blue box" in system, bool(flag))
            self.assertEqual("blue box" in text, bool(flag))
            self.assertEqual(seen[-1][1].tobytes() == after.tobytes(), not flag)   # unmarked AFTER only without the box

    def test_click_check_reads_the_fields_first(self):
        """click_check=read is v2's check with a reading step; one marked screenshot."""
        seen = []

        class Spy(FakeLLM):
            def ask(self, system, text, images=(), tag="misc", **kw):
                seen.append(len(list(images)))
                return super().ask(system, text, images, tag, **kw)
        m = MonitorV3(Config.load(None, ["adr_version=3", "click_check=read"]))
        m.llm = Spy({"is_final": True, "fields": [{"field": "text field", "content": "empty"}], "ready": False,
                     "problem": "the text field is empty", "confidence": 0.9})
        pc = m.check_commit("Enter Kanesha and press Submit", img(), "submit", CLICK, [])
        self.assertTrue(pc["blocked"])
        self.assertEqual(m.llm.calls[0][0], PRECOMMIT_READ_SYSTEM)
        self.assertIn("About to finalize", m.llm.calls[0][1])
        self.assertEqual(seen, [1])
        self.assertIn("the task text shown above the form is not part of any field", PRECOMMIT_READ_SYSTEM)

    def test_click_check_setting_is_checked(self):
        with self.assertRaises(ValueError):
            Config.load(None, ["adr_version=3", "click_check=v9"])


class TestRecoverV3(unittest.TestCase):
    def test_votes_are_off_in_v3(self):
        self.assertEqual(Config.load(None, ["adr_version=3"]).votes, 1)
        self.assertEqual(Config.load(None, []).votes, 3)

    def test_diagnostician_rules(self):
        self.assertIn("Choose II only if the screen did not change at all", SYSTEM_V3)
        d = DiagnosticianV3(Config.load(None, ["adr_version=3"]))
        self.assertIn("answer N", d.system)

    def test_wait_check_and_toggles(self):
        s = StrategistV3(Config.load(None, ["adr_version=3"]))
        p = s._wait_retry({}, {}, "task", CLICK, img())
        self.assertEqual(p.kind, "wait_check")
        self.assertTrue(p.actions[0]["only_if_unchanged"])
        t = s._wait_retry({"undo_by_reclick": True}, {}, "task", CLICK, img())
        self.assertEqual((t.kind, t.actions), ("look_again", []))
        e = s.escalate(p, {"evidence": "still nothing"}, CLICK)
        self.assertEqual((e.kind, e.actions), ("wait_check_failed", []))


class TestRetypeV33(unittest.TestCase):
    def test_typing_judged_no_effect_is_typed_again(self):
        s = StrategistV3(Config.load(None, ["adr_version=3", "retype=1"]))
        p = s._wait_retry({}, {}, "task", TYPE, img())
        self.assertEqual((p.kind, p.feedback), ("retype", RETYPED))
        self.assertNotIn("only_if_unchanged", p.actions[0])        # no screen check before typing again
        self.assertEqual(p.actions[0]["action"], TYPE)
        self.assertEqual(s._wait_retry({}, {}, "task", CLICK, img()).kind, "wait_check")   # clicks: as in v3

    def test_without_the_setting_typing_keeps_the_v3_fix(self):
        s = StrategistV3(Config.load(None, ["adr_version=3"]))
        self.assertEqual(s._wait_retry({}, {}, "task", TYPE, img()).kind, "wait_check")

    def test_a_failed_retype_asks_the_agent_to_type_it(self):
        s = StrategistV3(Config.load(None, ["adr_version=3", "retype=1"]))
        e = s.escalate(Plan("retype", [{"action": TYPE}], RETYPED), {"evidence": "the field is still empty"}, TYPE)
        self.assertEqual((e.kind, e.actions), ("retype_failed", []))
        self.assertIn("Click into the field yourself", e.feedback)


class TestWrapperV3(unittest.TestCase):
    def run_v3(self, fault_kind, monitor_replies):
        cfg = Config.load(None, ["adr_version=3", "second_look_s=0.01", "wait_s=0.01"])
        w = ADRWrapperV3(cfg, "adr")
        w.agent.llm = FakeLLM(*([{"thought": "click A", "action": {"type": "click", "x": 100, "y": 100}},
                                 {"thought": "click A again", "action": {"type": "click", "x": 100, "y": 100}},
                                 {"thought": "click Submit", "action": {"type": "click", "x": 200, "y": 500}}] * 3))
        w.monitor.llm = FakeLLM(*monitor_replies)
        w.diag.llm = FakeLLM(*[{"type": "II", "confidence": 0.95, "evidence": "dead click"}] * 5)
        return w.run(FakeEnv(fault_kind), "fake", 1, Fault(fault_kind))

    def test_dead_click_is_looked_at_twice_then_repeated_once(self):
        no = {"ok": False, "symptom": "no_effect", "confidence": 0.9, "task_progress": "same"}
        ok = {"ok": True, "symptom": "none", "confidence": 0.9, "is_final": False, "ready": True}
        r = self.run_v3("stagnation", [ok, no] + [ok] * 20)
        flagged = [s for s in r["steps"] if s.get("plan")]
        self.assertTrue(flagged)
        s = flagged[0]
        self.assertIn("first_look", s)
        self.assertEqual(s["second_look"], {"changed_px": 0, "asked": False})   # nothing moved: no second judge call
        self.assertEqual(s["verdict"]["symptom"], "no_effect")
        self.assertEqual((s["plan"]["kind"], s.get("wait_check")), ("wait_check", "unchanged"))
        self.assertEqual(len(s.get("recovery_executed", [])), 1)
        self.assertEqual(r["adr_version"], 3)
        told = [x["feedback_received"] for x in r["steps"] if x.get("feedback_received")]
        self.assertEqual(told[0], REPEATED)                  # the next step records what the agent was actually told

    def test_unchanged_screen_costs_no_second_judge_call(self):
        no = {"ok": False, "symptom": "no_effect", "confidence": 0.9, "task_progress": "same"}
        ok = {"ok": True, "symptom": "none", "confidence": 0.9, "is_final": False, "ready": True}
        cfg = Config.load(None, ["adr_version=3", "second_look_s=0.01", "wait_s=0.01"])
        w = ADRWrapperV3(cfg, "adr")
        w.agent.llm = FakeLLM({"thought": "click A", "action": {"type": "click", "x": 100, "y": 100}})
        w.monitor.llm = FakeLLM(ok, no, ok, ok, ok, ok, ok, ok)
        w.diag.llm = FakeLLM(*[{"type": "II", "confidence": 0.95, "evidence": "dead click"}] * 5)
        w.cfg.max_steps = 1
        w.run(FakeEnv("stagnation"), "fake", 1, Fault("stagnation"))
        judged = [c for c in w.monitor.llm.calls if "This step. Reasoning: click A" in c[1]]
        self.assertEqual(len(judged), 1)                                      # the step judge ran once, not twice

    def test_a_late_page_update_is_not_a_dead_click(self):
        class LateEnv(FakeEnv):
            """Shows the effect of a click only one screenshot later."""
            lag = 0

            def execute(self, action, source="agent", step=-1):
                self.lag = 1
                return super().execute(action, source, step)

            def screenshot(self):
                if self.lag:
                    self.lag -= 1
                    return img((255, 255 - 60 * max(0, len(self.clicks) - 1), 255))
                return super().screenshot()

        cfg = Config.load(None, ["adr_version=3", "second_look_s=0.01", "wait_s=0.01"])
        w = ADRWrapperV3(cfg, "adr")
        w.agent.llm = FakeLLM({"thought": "click A", "action": {"type": "click", "x": 100, "y": 100}},
                              {"thought": "click Submit", "action": {"type": "click", "x": 200, "y": 500}})
        ok = {"ok": True, "symptom": "none", "confidence": 0.9, "is_final": False, "ready": True}
        w.monitor.llm = FakeLLM(ok, {"ok": False, "symptom": "no_effect", "confidence": 0.9}, ok, ok, ok, ok)
        r = w.run(LateEnv("none"), "fake", 1, Fault("none"))
        first = r["steps"][0]
        self.assertIn("first_look", first)
        self.assertTrue(first["second_look"]["asked"])                        # the page changed late: asked again
        self.assertFalse(first["verdict"]["flagged"])
        self.assertNotIn("plan", first)
        self.assertTrue(LATE and REPEATED)


class TestV34(unittest.TestCase):
    """click_check=v4, no_effect_max_frac, done_always, n_free and a choice in a list that just opened."""

    def test_click_check_v4_uses_the_task_only_prompt_with_the_date(self):
        m = MonitorV3(Config.load(None, ["adr_version=3", "click_check=v4"]))
        m.llm = FakeLLM({"is_final": True, "ready": False, "problem": "x", "fix": "y", "confidence": 0.9})
        pc = m.check_commit("Search flights for tomorrow", img(), "click Search", CLICK, [])
        system, text, _ = m.llm.calls[0]
        self.assertEqual(system, PRECOMMIT_V4_SYSTEM)
        self.assertIn("Never list this action itself or the result it will produce", system)
        self.assertIn("Today is", text)
        self.assertIn("NOT done yet", text)
        self.assertTrue(pc["blocked"])

    def test_a_large_change_overrules_no_effect(self):
        m = MonitorV3(Config.load(None, ["adr_version=3", "no_effect_max_frac=0.1"]))
        m.llm = FakeLLM({"ok": False, "symptom": "no_effect", "confidence": 0.9, "task_progress": "same"})
        after = img()
        ImageDraw.Draw(after).rectangle([0, 0, 300, 400], fill=(0, 0, 0))
        v = m.check("task", img(), after, "scroll down", CLICK, [{"action": CLICK}])
        self.assertFalse(v["flagged"])
        self.assertEqual(v["judge"].get("overruled"), "pixels_changed")

    def test_a_small_change_keeps_no_effect(self):
        m = MonitorV3(Config.load(None, ["adr_version=3", "no_effect_max_frac=0.1"]))
        m.llm = FakeLLM({"ok": False, "symptom": "no_effect", "confidence": 0.9, "task_progress": "same"})
        v = m.check("task", img(), img(dot=(400, 560)), "click", CLICK, [{"action": CLICK}])
        self.assertTrue(v["flagged"])

    def test_done_check_reports_a_plausible_cannot_be_done_claim(self):
        m = MonitorV3(Config.load(None, ["adr_version=3", "done_always=1"]))
        m.llm = FakeLLM({"agent_says_impossible": True, "impossible_plausible": True,
                         "criteria": [{"criterion": "language set", "status": "not_met"}], "complete": False,
                         "confidence": 0.9})
        v = m.check_done("Change the language to Xenothian", img(), "This cannot be done; inform the user.")
        self.assertTrue(v["impossible"])
        self.assertFalse(v["flagged"])
        self.assertEqual(m.llm.calls[0][0], DONE_SYSTEM_V4)
        self.assertIn("Today is", m.llm.calls[0][1])

    def test_done_check_vetoes_an_implausible_cannot_be_done_claim(self):
        m = MonitorV3(Config.load(None, ["adr_version=3", "done_always=1"]))
        m.llm = FakeLLM({"agent_says_impossible": True, "impossible_plausible": False,
                         "criteria": [{"criterion": "toggle on", "status": "not_met"}], "complete": False,
                         "confidence": 0.9})
        v = m.check_done("Enable Do Not Track", img(), "The option does not exist.")
        self.assertFalse(v["impossible"])
        self.assertTrue(v["flagged"])

    def test_cannot_be_done_is_reported_as_infeasible_even_after_the_budget(self):
        cfg = Config.load(None, ["adr_version=3", "done_always=1", "max_recoveries=0",
                                 "second_look_s=0.01", "wait_s=0.01"])
        w = ADRWrapperV3(cfg, "adr")
        w.agent.llm = FakeLLM({"thought": "The feature does not exist, so this cannot be done.",
                               "action": {"type": "done"}})
        w.monitor.llm = FakeLLM({"agent_says_impossible": True, "impossible_plausible": True, "criteria": [],
                                 "complete": False, "confidence": 0.9})

        class DoneEnv(FakeEnv):
            reported = None

            def on_done(self, action):
                self.reported = action

        env = DoneEnv()
        r = w.run(env, "fake", 1, Fault("none"))
        self.assertTrue(r["steps"][0].get("reported_infeasible"))
        self.assertTrue(env.reported and env.reported.get("infeasible"))

    def test_done_vetoes_stop_at_the_cap_and_a_repeated_cannot_be_done_is_reported(self):
        cfg = Config.load(None, ["adr_version=3", "done_always=1", "max_recoveries=10", "max_done_vetoes=2",
                                 "second_look_s=0.01", "wait_s=0.01"])
        w = ADRWrapperV3(cfg, "adr")
        says = {"thought": "Chrome is not installed, so this cannot be done.", "action": {"type": "done"}}
        w.agent.llm = FakeLLM(says, says, says)
        doubt = {"agent_says_impossible": True, "impossible_plausible": False,
                 "criteria": [{"criterion": "language set", "status": "not_met"}], "complete": False,
                 "confidence": 0.9}
        w.monitor.llm = FakeLLM(doubt, doubt, doubt)
        w.diag.llm = FakeLLM({"type": "I", "confidence": 0.9, "evidence": "Chrome is open"},
                             {"type": "I", "confidence": 0.9, "evidence": "Chrome is open"})
        r = w.run(FakeEnv("none"), "fake", 1, Fault("none"))
        self.assertEqual([bool(s.get("plan")) for s in r["steps"]], [True, True, False])
        self.assertTrue(r["steps"][2].get("reported_infeasible"))
        self.assertEqual(r["end"], "declared_done")

    def test_a_cancelled_alarm_uses_no_fix(self):
        cfg = Config.load(None, ["adr_version=3", "n_free=1", "second_look_s=0.01", "wait_s=0.01"])
        w = ADRWrapperV3(cfg, "adr")
        w.agent.llm = FakeLLM({"thought": "click A", "action": {"type": "click", "x": 100, "y": 100}},
                              {"thought": "click Submit", "action": {"type": "click", "x": 200, "y": 500}})
        ok = {"ok": True, "symptom": "none", "confidence": 0.9, "is_final": False, "ready": True}
        wrong = {"ok": False, "symptom": "wrong_effect", "confidence": 0.9, "task_progress": "same"}
        w.monitor.llm = FakeLLM(ok, wrong, ok, ok, ok)
        w.diag.llm = FakeLLM({"type": "N", "confidence": 0.95, "evidence": "the step was fine"})
        r = w.run(FakeEnv("none"), "fake", 1, Fault("none"))
        self.assertEqual([s["plan"]["kind"] for s in r["steps"] if s.get("plan")], ["none"])
        self.assertEqual(r["recoveries"], 0)

    def test_a_choice_in_a_list_the_last_step_opened_is_not_checked(self):
        opened = {"action": CLICK, "verdict": {"judge": {"change": "The folder dropdown in the bookmark dialog is now "
                  "open, showing options including 'Bookmarks bar', 'Other bookmarks'."}, "signals": {"bbox": [1030, 8, 1834, 994]}}}
        self.assertTrue(choosing_in_open_list([opened], {"type": "click", "x": 1689, "y": 319}))
        self.assertFalse(choosing_in_open_list([opened], {"type": "click", "x": 500, "y": 319}))     # outside the list
        dialog = {"action": CLICK, "verdict": {"judge": {"change": "A 'Bookmark added' dialog appeared with 'Other "
                  "bookmarks' selected in the Folder dropdown."}, "signals": {"bbox": [742, 30, 1834, 380]}}}
        self.assertFalse(choosing_in_open_list([dialog], {"type": "click", "x": 1765, "y": 338}))    # Done stays checked

    def test_a_click_that_fixes_the_problem_is_not_blocked(self):
        m = MonitorV3(Config.load(None, ["adr_version=3", "click_check=v4"]))
        m.llm = FakeLLM({"is_final": True, "ready": False, "fixes_problem": True, "problem": "folder", "confidence": 0.9})
        self.assertFalse(m.check_commit("Save to the bookmarks bar", img(), "pick Bookmarks bar", CLICK, [])["blocked"])


class TestV35(unittest.TestCase):
    """replan_gap, click_check=v5 and done_pairs."""

    def _flag_every_step(self, gap):
        cfg = Config.load(None, ["adr_version=3", f"replan_gap={gap}", "max_steps=3", "second_look_s=0", "wait_s=0.01"])
        w = ADRWrapperV3(cfg, "adr")
        scroll = {"thought": "scroll down", "action": {"type": "scroll", "x": 100, "y": 100, "direction": "down"}}
        w.agent.llm = FakeLLM(scroll, scroll, scroll)
        w.monitor.check = lambda *a, **k: {"flagged": True, "symptom": "wrong_effect", "source": "judge",
                                           "confidence": 0.9, "evidence": "nothing new", "judge": {}, "signals": {}}
        w._recover = lambda *a, **k: (Plan("replan", feedback="Try another way."), {"type": "V"})
        return w.run(FakeEnv("none"), "fake", 1, Fault("none"))

    def test_a_replan_right_after_a_replan_is_dropped_and_uses_no_fix(self):
        r = self._flag_every_step(3)
        self.assertEqual([s["plan"]["kind"] for s in r["steps"]], ["replan", "none", "none"])
        self.assertEqual([bool(s.get("replan_dropped")) for s in r["steps"]], [False, True, True])
        self.assertEqual(r["recoveries"], 1)

    def test_without_a_gap_every_replan_is_sent(self):
        r = self._flag_every_step(0)
        self.assertEqual([s["plan"]["kind"] for s in r["steps"]], ["replan", "replan", "replan"])
        self.assertEqual(r["recoveries"], 3)

    def test_click_check_v5_blocks_an_unmet_condition_even_if_the_reply_says_ready(self):
        m = MonitorV3(Config.load(None, ["adr_version=3", "click_check=v5"]))
        m.llm = FakeLLM({"is_final": True, "on_right_element": True,
                         "conditions": [{"task_says": "economy", "screen_shows": "Business", "met": False}],
                         "fixes_problem": False, "ready": True, "problem": "", "fix": "", "confidence": 0.9})
        pc = m.check_commit("Book an economy seat", img(), "click Book", CLICK, [])
        self.assertEqual(m.llm.calls[0][0], PRECOMMIT_V5_SYSTEM)
        self.assertIn("task_says", PRECOMMIT_V5_SYSTEM)
        self.assertTrue(pc["blocked"])

    def test_click_check_v5_does_not_block_a_click_that_fixes_the_condition(self):
        m = MonitorV3(Config.load(None, ["adr_version=3", "click_check=v5"]))
        m.llm = FakeLLM({"is_final": True, "conditions": [{"task_says": "economy", "screen_shows": "none", "met": False}],
                         "fixes_problem": True, "ready": False, "confidence": 0.9})
        self.assertFalse(m.check_commit("Book an economy seat", img(), "click Economy", CLICK, [])["blocked"])

    def test_done_pairs_uses_the_paired_prompt(self):
        m = MonitorV3(Config.load(None, ["adr_version=3", "done_always=1", "done_pairs=1"]))
        m.llm = FakeLLM({"agent_says_impossible": False, "complete": False, "confidence": 0.9,
                         "criteria": [{"criterion": "page in the bookmarks bar", "task_says": "bookmarks bar",
                                       "shown": "a note names another folder", "status": "not_met"}]})
        v = m.check_done("Save this page to the bookmarks bar", img(), "Done.", history=[])
        self.assertEqual(m.llm.calls[0][0], DONE_SYSTEM_V5)
        self.assertIn("a different value is not_met", DONE_SYSTEM_V5)
        self.assertTrue(v["flagged"])


class TestV36Axes(unittest.TestCase):
    """use_axes: every axis of taxonomy v4 changes the fix."""

    def _s(self):
        return StrategistV3(Config.load(None, ["adr_version=3", "use_axes=1"]))

    def _d(self, **kw):
        d = {"type": "II", "subtype": "dead_click", "evidence": "nothing changed", "screen_truth": "the Save button is visible",
             "awareness": "unaware", "behaviour": "progressing", "recoverability": "recoverable",
             "feasibility": "feasible", "self_verification": "verified", "confidence": 0.9}
        d.update(kw)
        return d

    V = {"flagged": True, "symptom": "no_effect", "evidence": "nothing changed"}

    def test_an_aware_progressing_agent_gets_no_advice_and_uses_no_fix(self):
        p = self._s().plan(self._d(type="V", subtype="bad_plan", awareness="aware"), self.V, "task", CLICK, img())
        self.assertEqual(p.kind, "none")

    def test_an_unaware_agent_is_told_that_its_step_failed(self):
        s = self._s()
        s.plan(self._d(), self.V, "task", CLICK, img())
        self.assertIn("although your reasoning assumes it did", s.axis_note[0])

    def test_a_loop_turns_a_repeat_into_a_replan_that_names_the_action(self):
        s = self._s()
        p = s.plan(self._d(behaviour="looping"), self.V, "task", CLICK, img())
        self.assertEqual(p.kind, "replan")
        self.assertIn("without progress. Do not do it again.", s.axis_note[0])

    def test_feasibility_says_whether_the_task_can_be_done(self):
        s = self._s()
        s.plan(self._d(type="V", subtype="knowledge_gap", feasibility="falsely_infeasible"), self.V, "task", CLICK, img())
        self.assertIn("The task can be done: the Save button is visible", s.axis_note[1])
        s.plan(self._d(type="V", subtype="knowledge_gap", feasibility="infeasible"), self.V, "task", CLICK, img())
        self.assertIn("report that the task cannot be done", s.axis_note[1])

    def test_an_unverified_value_fix_asks_the_agent_to_check(self):
        s = self._s()
        s.plan(self._d(type="IV", subtype="wrong_value", self_verification="not_verified", correct_value="12"),
               self.V, "task", {"type": "type", "text": "13", "x": 10, "y": 10}, img())
        self.assertIn("check on the screen that the value really changed", s.axis_note[1])

    def test_the_step_budget_is_named_near_the_end_of_a_loop(self):
        s = self._s()
        s.progress = (8, 10)
        s.plan(self._d(behaviour="looping"), self.V, "task", CLICK, img())
        self.assertIn("Only 1 step is left.", s.axis_note[1])
        s.progress = (2, 10)
        s.plan(self._d(behaviour="looping"), self.V, "task", CLICK, img())
        self.assertNotIn("left", s.axis_note[1])

    def test_without_the_switch_nothing_changes(self):
        s = StrategistV3(Config.load(None, ["adr_version=3"]))
        p = s.plan(self._d(type="V", subtype="bad_plan", awareness="aware"), self.V, "task", CLICK, img())
        self.assertEqual(p.kind, "replan")
        self.assertEqual(s.axis_note, ("", ""))

    def test_the_diagnostician_asks_for_the_two_new_axes(self):
        on = DiagnosticianV3(Config.load(None, ["adr_version=3", "use_axes=1"]))
        off = DiagnosticianV3(Config.load(None, ["adr_version=3"]))
        for field in ('"feasibility": "feasible" | "falsely_infeasible"', '"self_verification": "verified" | "not_verified"',
                      "feasibility (feasible | falsely_infeasible"):
            self.assertIn(field, on.system)
            self.assertNotIn(field, off.system)

    def test_the_agent_gets_the_axes_sentences_with_the_fix(self):
        cfg = Config.load(None, ["adr_version=3", "use_axes=1", "max_steps=2", "second_look_s=0", "wait_s=0.01"])
        w = ADRWrapperV3(cfg, "adr")
        scroll = {"thought": "scroll down", "action": {"type": "scroll", "x": 100, "y": 100, "direction": "down"}}
        w.agent.llm = FakeLLM(scroll, scroll)
        w.monitor.check = lambda *a, **k: {"flagged": True, "symptom": "wrong_effect", "source": "judge",
                                           "confidence": 0.9, "evidence": "nothing new", "judge": {}, "signals": {}}
        w.diag.llm = FakeLLM(self._d(type="V", subtype="bad_plan", awareness="unaware"))
        r = w.run(FakeEnv("none"), "fake", 1, Fault("none"))
        self.assertIn("although your reasoning assumes it did", r["steps"][0]["axis_note"]["before"])
        self.assertTrue(r["steps"][1]["feedback_received"].startswith("Your last step did not do what you intended"))


class TestV37(unittest.TestCase):
    """replan_strikes and gate_dismiss_first: two strikes before "try another way"; close an access gate once
    before handing the task over."""

    def _run(self, flags, symptom="wrong_effect"):
        cfg = Config.load(None, ["adr_version=3", "replan_strikes=2", f"max_steps={len(flags)}", "second_look_s=0",
                                 "wait_s=0.01"])
        w = ADRWrapperV3(cfg, "adr")
        scroll = {"thought": "scroll down", "action": {"type": "scroll", "x": 100, "y": 100, "direction": "down"}}
        w.agent.llm = FakeLLM(*[scroll] * len(flags))
        it = iter(flags)
        w.monitor.check = lambda *a, **k: ({"flagged": True, "symptom": symptom, "source": "judge", "confidence": 0.9,
                                            "evidence": "x", "judge": {}, "signals": {}} if next(it) else
                                           {"flagged": False, "symptom": "none", "source": "judge", "confidence": 0.9,
                                            "evidence": "", "judge": {}, "signals": {}})
        w._recover = lambda *a, **k: (Plan("replan", feedback="Try another way."), {"type": "V"})
        return w.run(FakeEnv("none"), "fake", 1, Fault("none"))

    def test_a_single_flag_holds_the_replan_back_and_a_second_in_a_row_sends_it(self):
        r = self._run([True, True, False, True])
        self.assertEqual([(s.get("plan") or {}).get("kind") for s in r["steps"]], ["none", "replan", None, "none"])
        self.assertTrue(r["steps"][0].get("replan_held"))
        self.assertEqual(r["recoveries"], 1)

    def test_the_loop_rule_counts_as_the_second_strike(self):
        r = self._run([True], symptom="loop")
        self.assertEqual(r["steps"][0]["plan"]["kind"], "replan")

    def test_an_access_gate_is_closed_once_before_the_task_is_handed_over(self):
        s = StrategistV3(Config.load(None, ["adr_version=3", "gate_dismiss_first=1"]))
        s.locate = lambda img, desc: (100, 200)
        d = {"type": "III", "subtype": "access_gate", "evidence": "a sign-in window", "confidence": 0.9}
        v = {"flagged": True, "symptom": "blocked"}
        self.assertEqual(s.plan(d, v, "task", CLICK, img()).kind, "dismiss_retry")
        self.assertEqual(s.plan(d, v, "task", CLICK, img()).kind, "escalate")
        s.reset()
        s.locate = lambda img, desc: None
        self.assertEqual(s.plan(d, v, "task", CLICK, img()).kind, "escalate")

    def test_without_the_switch_a_gate_is_handed_over_at_once(self):
        s = StrategistV3(Config.load(None, ["adr_version=3"]))
        s.locate = lambda img, desc: (100, 200)
        d = {"type": "III", "subtype": "access_gate", "evidence": "a sign-in window", "confidence": 0.9}
        self.assertEqual(s.plan(d, {"flagged": True, "symptom": "blocked"}, "task", CLICK, img()).kind, "escalate")


class TestRetryBaselineV3(unittest.TestCase):
    """The retry baselines have no Strategist, so a flagged step must not touch it (retry_evidence once crashed on
    its first flagged step: 'NoneType' object has no attribute 'progress')."""

    def test_retry_evidence_runs_with_the_v37_switches(self):
        cfg = Config.load(None, ["adr_version=3", "use_axes=1", "replan_strikes=2", "gate_dismiss_first=1", "max_steps=3",
                                 "second_look_s=0", "wait_s=0.01"])
        w = ADRWrapperV3(cfg, "retry_evidence")
        self.assertIsNone(w.strat)
        scroll = {"thought": "scroll down", "action": {"type": "scroll", "x": 100, "y": 100, "direction": "down"}}
        w.agent.llm = FakeLLM(*[scroll] * 3)
        w.monitor.check = lambda *a, **k: {"flagged": True, "symptom": "wrong_effect", "source": "judge",
                                           "confidence": 0.9, "evidence": "the list did not move", "judge": {},
                                           "signals": {}}
        r = w.run(FakeEnv("none"), "fake", 1, Fault("none"))
        self.assertFalse(r.get("error"))
        plans = [s.get("plan") for s in r["steps"] if s.get("plan")]
        self.assertTrue(plans)
        self.assertTrue(all(not p.get("actions") for p in plans))     # the baseline only talks, it never acts


if __name__ == "__main__":
    unittest.main()
