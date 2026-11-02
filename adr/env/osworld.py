"""
Live OSWorld environment: a real Ubuntu virtual machine in VMware with real apps and websites.

Same contract as MiniWoBEnv: reset(task, seed, fault) -> goal, screenshot() -> image, execute(action) -> info,
status() for the benchmark's own check, plus on_done(action) for the agent's final claim. Our actions become
pyautogui commands run inside the VM; success comes from OSWorld's own checker (DesktopEnv.evaluate()), run once at
the end of the episode (status(final=True)). No faults are injected: OSWorld failures are natural.

One VM per process, started once and reverted to its snapshot by every reset, so run with workers=1.
Needs the OSWorld code in cfg.osworld_dir, run from its venv (third_party/OSWorld/.venv), and vmrun on PATH.

The episode timing follows OSWorld's own runner (lib_run_single.py): 60 s after a reset for the environment to be
ready, and 20 s before scoring for apps to settle (Chrome writes changed settings to disk seconds later). The agent's
own typing does not click the field first: in Chrome's address bar that second click dropped the selection, so a
typed URL was appended to the old one.
"""
from __future__ import annotations

import atexit
import io
import json
import os
import sys
import time
from typing import Optional

from PIL import Image

from ..config import Config
from .faults import Fault

KEYS = {"enter": "enter", "tab": "tab", "backspace": "backspace", "space": "space", "escape": "esc",
        "delete": "delete", "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right",
        "pageup": "pageup", "pagedown": "pagedown", "home": "home", "end": "end", "control": "ctrl",
        "meta": "win", "alt": "alt", "shift": "shift"}

_DESKTOP = None
RESET_WAIT = 60          # seconds after a reset, as OSWorld's lib_run_single.py ("wait for the environment to be ready")
SCORE_WAIT = 20          # seconds before the checker runs, as lib_run_single.py ("wait for the environment to settle")


def _patch_vm_ip(root: str, wait: int = 120) -> None:
    """When VMware Tools do not start in the guest after a revert to the snapshot, vmrun cannot report the guest's IP
    and OSWorld's provider waits for it forever. Ask vmrun for up to `wait` seconds, then use the last IP that worked,
    but only if OSWorld's server in the VM answers there (port 5000); otherwise keep trying both, as OSWorld did.
    Nothing changes while VMware Tools run."""
    import urllib.request
    from desktop_env.providers.vmware import provider as P
    cache = os.path.join(root, "osworld_vm_ip")

    def vmrun_ip(path_to_vm):
        try:
            out = P.VMwareProvider._execute_command(
                ["vmrun"] + P.get_vmrun_type(return_list=True) + ["getGuestIPAddress", path_to_vm], return_output=True)
        except Exception:
            return ""
        return out if out and out[0].isdigit() else ""

    def answers(ip):
        try:
            with urllib.request.urlopen(f"http://{ip}:5000/screenshot", timeout=10) as r:
                return r.status == 200
        except Exception:
            return False

    def get_ip_address(self, path_to_vm):
        deadline = time.time() + wait
        while True:
            ip = vmrun_ip(path_to_vm)
            if ip:
                with open(cache, "w") as f:
                    f.write(ip)
                return ip
            if time.time() > deadline and os.path.exists(cache):
                last = open(cache).read().strip()
                if last and answers(last):
                    print(f"VMware Tools not running: using the last IP that worked, {last}", flush=True)
                    return last
            time.sleep(5)

    P.VMwareProvider.get_ip_address = get_ip_address


def _desktop(cfg: Config):
    """The process's one DesktopEnv (starting the VM takes about a minute)."""
    global _DESKTOP
    if _DESKTOP is None:
        root = str(cfg.path(cfg.osworld_dir))
        if root not in sys.path:
            sys.path.insert(0, root)
        cwd = os.getcwd()
        os.chdir(root)                       # OSWorld resolves vmware_vm_data and cache paths from its own folder
        try:
            _patch_vm_ip(root)
            from desktop_env.desktop_env import DesktopEnv
            vm = cfg.osworld_vm
            if vm and not os.path.isabs(vm):           # relative to the OSWorld folder
                vm = str(cfg.path(cfg.osworld_dir) / vm)
            if vm:
                # OSWorld checks "is my VM running?" by comparing this path with the paths `vmrun list` prints, without
                # resolving symlinks, so a path through a symlink never matched and OSWorld started a VM that was
                # already running. Use the real path, as `vmrun list` prints it.
                vm = os.path.realpath(vm)
            _DESKTOP = DesktopEnv(provider_name="vmware", path_to_vm=vm or None, action_space="pyautogui",
                                  screen_size=(1920, 1080), headless=True, require_a11y_tree=False, os_type="Ubuntu")
        finally:
            os.chdir(cwd)
        atexit.register(_DESKTOP.close)
    return _DESKTOP


def _q(s: str) -> str:
    return json.dumps(str(s))


def to_pyautogui(a: dict, source: str = "agent") -> str:
    """One of our actions as a pyautogui command for the VM. The agent's typing goes where the focus is, without a
    click (UI-TARS clicks a field in its own step, as in OSWorld's UI-TARS harness; the position attached to the type
    action is only for the judge's red circle). A retype by the wrapper, as a fix, clicks the field and replaces its
    content."""
    t = a.get("type")
    if t == "click":
        fn = "doubleClick" if a.get("double") else "rightClick" if a.get("button") == "right" else "click"
        return f"pyautogui.{fn}({int(a['x'])}, {int(a['y'])})"
    if t == "type":
        cmd = []
        if "x" in a and source != "agent":
            cmd += [f"pyautogui.click({int(a['x'])}, {int(a['y'])})",
                    "pyautogui.hotkey('ctrl', 'a')", "pyautogui.press('backspace')"]
        cmd.append(f"pyautogui.typewrite({_q(a.get('text', ''))}, interval=0.02)")
        if a.get("submit"):
            cmd.append("pyautogui.press('enter')")
        return "; ".join(cmd)
    if t == "key":
        keys = [KEYS.get(k.lower(), k.lower()) for k in str(a.get("key", "Enter")).split("+")]
        return f"pyautogui.press({_q(keys[0])})" if len(keys) == 1 else "pyautogui.hotkey(" + ", ".join(_q(k) for k in keys) + ")"
    if t == "scroll":
        clicks = -5 if float(a.get("dy", 100)) > 0 else 5
        return f"pyautogui.scroll({clicks}, x={int(a.get('x', 960))}, y={int(a.get('y', 540))})"
    return "WAIT"


class OSWorldEnv:
    def __init__(self, cfg: Config, browser=None):
        self.cfg = cfg
        self.env = _desktop(cfg)
        self.task_cfg: dict = {}
        self.goal = ""
        self.fault = Fault()
        self.agent_actions = 0
        self._shot: Optional[bytes] = None
        self.score: Optional[float] = None
        self.rect = (0.0, 0.0, 1920.0, 1080.0)

    def reset(self, task: str, seed: int, fault: Optional[Fault] = None) -> str:
        path = self.cfg.path(self.cfg.osworld_dir) / "evaluation_examples" / "examples" / self.cfg.osworld_domain / f"{task}.json"
        self.task_cfg = json.loads(path.read_text())
        cwd = os.getcwd()
        os.chdir(str(self.cfg.path(self.cfg.osworld_dir)))
        try:
            self.env.reset(task_config=self.task_cfg)
        finally:
            os.chdir(cwd)
        time.sleep(RESET_WAIT)
        self._shot = None
        self.goal = self.task_cfg["instruction"]
        self.fault = fault or Fault()          # unused: failures in OSWorld are natural
        self.agent_actions = 0
        self.score = None
        return self.goal

    def close(self) -> None:
        pass                                   # the VM stays up for the next episode; it is stopped at exit

    def screenshot(self) -> Image.Image:
        if self._shot is None:
            self._shot = self.env._get_obs()["screenshot"]
        return Image.open(io.BytesIO(self._shot)).convert("RGB")

    def size(self):
        return 1920, 1080

    def status(self, final: bool = False) -> dict:
        """OSWorld has no 'done' signal during the episode; its checker runs once, at the end."""
        if not final:
            return {"done": False, "raw": None, "reason": None, "success": False}
        if self.score is None:
            time.sleep(SCORE_WAIT)
            cwd = os.getcwd()
            os.chdir(str(self.cfg.path(self.cfg.osworld_dir)))
            try:
                self.score = float(self.env.evaluate())
            except Exception as e:             # a checker crash counts as a failure, with the reason kept
                self.score, reason = 0.0, f"evaluate failed: {type(e).__name__}: {str(e)[:150]}"
                return {"done": True, "raw": 0.0, "reason": reason, "success": False}
            finally:
                os.chdir(cwd)
        return {"done": True, "raw": self.score, "reason": None, "success": self.score >= 0.999}

    def on_done(self, action: dict) -> None:
        """The agent's final claim: DONE, or FAIL when it says the task cannot be done (OSWorld scores infeasible
        tasks by it)."""
        self.env.step("FAIL" if action.get("infeasible") else "DONE", pause=0.5)

    def execute(self, action: dict, source: str = "agent", step: int = -1) -> dict:
        a = dict(action)
        if source == "agent":
            self.agent_actions += 1
        err = ""
        try:
            obs, _, _, _ = self.env.step(to_pyautogui(a, source), pause=self.cfg.osworld_pause)
            self._shot = obs.get("screenshot")
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
            self._shot = None
        return {"executed": a, "fault_fired": False, "error": err}
