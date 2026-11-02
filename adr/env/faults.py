"""
Fault injection: real GUI failure conditions, each with a known true type.

Modelled on the anomaly types of GUI-Robust (arXiv 2506.14477): pop-ups and
cookie banners, unresponsive clicks and loading delays, plus the two agent-side
execution slips (mis-located click, wrong value typed). Each fault fires once,
on the k-th eligible agent action, so its step is known exactly. That is what
gives RQ1 clean ground truth.

    kind          true type   what happens
    grounding     I           mis-click: the click lands `grounding_offset` px away from where it was aimed
    stagnation    II          ignored input: the page ignores input for `stagnation_ms` (dead click / not ready)
    occlusion     III         pop-up: a banner appears over the target just as the agent acts
    wrong_value   IV          typo: one character of the typed value is wrong (dropped, swapped or changed)

Held-out kinds, never used while the system was developed (for the confirmatory run):
    loading       II          loading cover: a "Loading" cover blocks the page for a while, then clears by itself
    layout_shift  I           layout shift: a banner pushes the content down just before the click, so it lands too high

With `fault_randomize=1` every setting is drawn per episode (seeded by task, seed and
kind, so it is reproducible): freeze length, offset size and direction, pop-up style,
size, position and close-button text, typo kind. The typo is then also written into
the agent's own reasoning, so it does not leave a tell-tale mismatch.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field

TRUE_TYPE = {"grounding": "I", "stagnation": "II", "occlusion": "III", "wrong_value": "IV",
             "loading": "II", "layout_shift": "I"}
HELD_OUT = ["loading", "layout_shift"]


@dataclass
class Fault:
    kind: str = "none"
    at: int = 1           # fire on the k-th eligible agent action (1-based)
    variant: int = 0      # picks the pop-up style / offset direction / typo kind
    fired: bool = False
    fired_step: int = -1  # agent step index at which it fired
    detail: str = ""
    params: dict = field(default_factory=dict)   # randomised settings, empty = fixed defaults

    @property
    def true_type(self) -> str:
        return TRUE_TYPE.get(self.kind, "")

    def eligible(self, action: dict) -> bool:
        t = action.get("type")
        if self.kind == "grounding":
            return t in ("click", "type") and "x" in action
        if self.kind == "stagnation":
            return t in ("click", "type", "key")
        if self.kind == "occlusion":
            return t in ("click", "type") and "x" in action
        if self.kind == "wrong_value":
            return t == "type" and len(str(action.get("text", ""))) >= 2
        if self.kind == "loading":
            return t in ("click", "type", "key")
        if self.kind == "layout_shift":
            return t in ("click", "type") and "x" in action
        return False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["true_type"] = self.true_type
        return d


def randomize(kind: str, task: str, seed: int) -> dict:
    """Per-episode fault settings, reproducible from (task, seed, kind)."""
    rng = random.Random(f"{task}|{seed}|{kind}")
    if kind == "grounding":
        r, a = rng.uniform(10, 40), rng.uniform(0, 2 * math.pi)
        return {"dx": r * math.cos(a), "dy": r * math.sin(a)}
    if kind == "stagnation":
        return {"ms": rng.randint(1000, 8000)}
    if kind == "occlusion":
        return {"style": rng.randrange(6), "h": rng.randint(60, 140), "corner": rng.choice(["tr", "tl", "br", "bl"]),
                "full": rng.random() < 0.3}
    if kind == "wrong_value":
        return {"variant": rng.randrange(6), "consistent": True}
    if kind == "loading":
        return {"ms": rng.randint(2000, 7000)}
    if kind == "layout_shift":
        return {"px": rng.randint(18, 40)}
    return {}


def corrupt(text: str, variant: int) -> str:
    """A small, realistic slip: drop, swap or change one character."""
    s = str(text)
    if len(s) < 2:
        return s + "x"
    v = variant % 3
    if v == 0:
        out = s[:-1]
    elif v == 1:
        i = max(0, len(s) // 2 - 1)
        out = s[:i] + s[i + 1] + s[i] + s[i + 2:]
    else:
        i = len(s) // 2
        c = s[i]
        if c.isdigit():
            r = str((int(c) + 1) % 10)
        elif c.isalpha():
            r = "e" if c.lower() != "e" else "a"
            r = r.upper() if c.isupper() else r
        else:
            r = "x"
        out = s[:i] + r + s[i + 1:]
    if out == s:
        out = s[:-1]
    return out


def corrupt_random(text: str, variant: int) -> str:
    """Six slip kinds for the randomised mode (the fixed mode keeps `corrupt`)."""
    s = str(text)
    if variant % 6 < 3 or len(s) < 2:
        return corrupt(s, variant % 3)
    i = len(s) // 2
    v = variant % 6
    if v == 3:                                   # a character typed twice
        out = s[:i] + s[i] + s[i:]
    elif v == 4:                                 # wrong letter case
        j = next((k for k, c in enumerate(s) if c.isalpha()), None)
        out = s if j is None else s[:j] + s[j].swapcase() + s[j + 1:]
    else:                                        # neighbouring key on the keyboard
        rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm", "1234567890"]
        c = s[i].lower()
        row = next((r for r in rows if c in r), None)
        if row:
            k = row.index(c)
            n = row[k + 1] if k + 1 < len(row) else row[k - 1]
            out = s[:i] + (n.upper() if s[i].isupper() else n) + s[i + 1:]
        else:
            out = s[:i] + "x" + s[i + 1:]
    return out if out != s else corrupt(s, 0)


# Installed once per episode. Hides MiniWoB's own click markers and score panel,
# and adds the stagnation guard (inactive until __adrBlockUntil is set).
SETUP_JS = r"""
() => {
  const st = document.createElement('style');
  st.textContent = '#click-canvas{display:none !important} #reward-display{display:none !important}';
  document.head.appendChild(st);
  window.__adrBlockUntil = 0;
  const guard = (e) => {
    if (Date.now() < window.__adrBlockUntil) { e.preventDefault(); e.stopImmediatePropagation(); }
  };
  ['mousedown','mouseup','click','dblclick','keydown','keypress','keyup','beforeinput','input']
    .forEach(t => window.addEventListener(t, guard, true));
}
"""

STAGNATE_JS = "(ms) => { window.__adrBlockUntil = Date.now() + ms; }"

# Randomised pop-up: six styles, random size, close control in a random corner away
# from the agent's click; sometimes it covers the whole task area.
OVERLAY_RANDOM_JS = r"""
([x, y, p]) => {
  const wrap = document.getElementById('wrap').getBoundingClientRect();
  const area = document.getElementById('area').getBoundingClientRect();
  const h = p.full ? (wrap.bottom - area.top) : p.h;
  const top = p.full ? area.top : Math.max(area.top, Math.min(y - h / 2, wrap.bottom - h));
  const styles = [
    ['#f4f4f4', 'We use cookies to improve your experience.', 'Accept'],
    ['#fff8dc', 'Get 10% off! Subscribe to our newsletter.', '\u00d7'],
    ['#e8f0ff', 'Special offer just for you. Upgrade today!', 'No thanks'],
    ['#eef7ee', 'Need help? Chat with us now.', 'Close'],
    ['#fdecea', 'Rate your experience with us!', 'Maybe later'],
    ['#f0eefc', 'New features are available. See what is new.', 'Got it'],
  ];
  const [bg, msg, label] = styles[p.style % styles.length];
  const box = document.createElement('div');
  box.id = 'adr-overlay';
  box.style.cssText = `position:fixed;left:${wrap.left}px;top:${top}px;width:${wrap.width}px;height:${h}px;` +
    `z-index:99999;box-sizing:border-box;padding:6px;font:10px Arial;color:#222;background:${bg};` +
    'box-shadow:0 0 6px rgba(0,0,0,.45);';
  const t = document.createElement('div');
  t.textContent = msg;
  box.appendChild(t);
  const b = document.createElement('button');
  b.textContent = label;
  b.id = 'adr-dismiss';
  let corner = p.corner;
  const bx = corner.includes('r') ? wrap.right - 30 : wrap.left + 30;
  const by = corner.includes('t') ? top + 8 : top + h - 8;
  if (Math.abs(bx - x) < 30 && Math.abs(by - y) < 20) {   // never under the agent's click
    corner = (corner.includes('t') ? 'b' : 't') + (corner.includes('r') ? 'l' : 'r');
  }
  b.style.cssText = 'position:absolute;font:10px Arial;padding:2px 6px;cursor:pointer;' +
    (corner.includes('t') ? 'top:4px;' : 'bottom:4px;') + (corner.includes('r') ? 'right:4px;' : 'left:4px;');
  // keep the text clear of the close control, as a real banner would
  t.style.cssText = 'line-height:13px;margin:0 4px;' + (corner.includes('t') ? 'padding-top:22px;' : 'padding-bottom:22px;');
  b.addEventListener('click', (e) => { e.stopPropagation(); box.remove(); });
  box.appendChild(b);
  box.addEventListener('mousedown', e => e.stopPropagation());
  box.addEventListener('click', e => e.stopPropagation());
  document.body.appendChild(box);
}
"""

# Held-out II: a visible loading cover that blocks input and clears by itself.
LOADING_JS = r"""
(ms) => {
  const area = document.getElementById('area').getBoundingClientRect();
  const wrap = document.getElementById('wrap').getBoundingClientRect();
  const c = document.createElement('div');
  c.id = 'adr-loading';
  c.style.cssText = `position:fixed;left:${wrap.left}px;top:${area.top}px;width:${wrap.width}px;` +
    `height:${wrap.bottom - area.top}px;z-index:99998;background:rgba(255,255,255,.85);` +
    'display:flex;align-items:center;justify-content:center;font:11px Arial;color:#444;';
  c.textContent = 'Loading\u2026';
  c.addEventListener('mousedown', e => { e.stopPropagation(); e.preventDefault(); }, true);
  c.addEventListener('click', e => { e.stopPropagation(); e.preventDefault(); }, true);
  document.body.appendChild(c);
  window.__adrBlockUntil = Date.now() + ms;
  setTimeout(() => c.remove(), ms);
}
"""

# Held-out I: a late-loading banner pushes the task content down just before the click.
SHIFT_JS = r"""
(px) => {
  const area = document.getElementById('area');
  const s = document.createElement('div');
  s.id = 'adr-shift';
  s.textContent = 'Sponsored';
  s.style.cssText = `height:${px}px;line-height:${px}px;background:#e9e9e9;color:#777;` +
    'font:9px Arial;text-align:center;margin-bottom:2px;';
  area.insertBefore(s, area.firstChild);
}
"""

# A pop-up placed over the point the agent is about to click. The dismiss control
# sits away from that point, so clicking the same spot again never clears it.
OVERLAY_JS = r"""
([x, y, variant]) => {
  const wrap = document.getElementById('wrap').getBoundingClientRect();
  const area = document.getElementById('area').getBoundingClientRect();
  const h = 92;
  let top = Math.max(area.top, Math.min(y - h / 2, wrap.bottom - h));
  const box = document.createElement('div');
  box.id = 'adr-overlay';
  box.style.cssText = `position:fixed;left:${wrap.left}px;top:${top}px;width:${wrap.width}px;height:${h}px;` +
    'z-index:99999;box-sizing:border-box;padding:6px;font:10px Arial;color:#222;' +
    'box-shadow:0 0 6px rgba(0,0,0,.45);';
  const styles = [
    ['#f4f4f4', 'We use cookies to improve your experience. By continuing you agree to our cookie policy.', 'Accept'],
    ['#fff8dc', 'Get 10% off your first order! Subscribe to our newsletter.', '×'],
    ['#e8f0ff', 'Special offer just for you. Upgrade to premium today!', 'No thanks'],
  ];
  const [bg, msg, label] = styles[variant % 3];
  box.style.background = bg;
  const p = document.createElement('div');
  p.textContent = msg;
  p.style.cssText = 'margin-right:22px;line-height:13px;';
  box.appendChild(p);
  const b = document.createElement('button');
  b.textContent = label;
  b.id = 'adr-dismiss';
  const near = (y - top) > h / 2;   // keep the button away from the agent's click
  b.style.cssText = 'position:absolute;font:10px Arial;padding:2px 6px;cursor:pointer;' +
    (label === '×' ? 'right:3px;top:3px;font-size:13px;padding:0 5px;'
                        : (near ? 'right:6px;top:6px;' : 'right:6px;bottom:6px;'));
  b.addEventListener('click', (e) => { e.stopPropagation(); box.remove(); });
  box.appendChild(b);
  box.addEventListener('mousedown', e => e.stopPropagation());
  box.addEventListener('click', e => e.stopPropagation());
  document.body.appendChild(box);
}
"""
