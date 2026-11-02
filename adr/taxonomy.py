"""Loads the taxonomy file and renders it for the Diagnostician prompt."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

TYPE_IDS = ["I", "II", "III", "IV", "V", "VI", "N"]


class Taxonomy:
    def __init__(self, path: Path):
        self.data = json.loads(Path(path).read_text())
        self.types: Dict[str, dict] = {t["id"]: t for t in self.data["types"]}

    def recovery(self, type_id: str, subtype: str = "") -> str:
        t = self.types.get(type_id)
        if not t:
            return "replan"
        return t.get("subtype_recovery", {}).get(subtype, t.get("recovery", "replan"))

    def render(self) -> str:
        out = ["FAILURE TYPES (choose exactly one id):"]
        for t in self.data["types"]:
            out.append(f"- {t['id']} {t['name']}: {t['definition']} Subtypes: {', '.join(t['subtypes'])}.")
        out.append("")
        out.append("DECISION PROCEDURE (answer in order; the first 'yes' decides):")
        out += [f"  {q}" for q in self.data["procedure"]]
        out.append("TIE-BREAKERS:")
        out += [f"  - {r}" for r in self.data["tie_breakers"]]
        out.append("AXES:")
        for k, v in self.data["axes"].items():
            out.append(f"  - {k} ({' | '.join(v['values'])}): {v['rule']}")
        return "\n".join(out)
