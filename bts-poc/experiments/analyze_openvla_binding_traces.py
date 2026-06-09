from __future__ import annotations

"""Analyze OpenVLA/BTS rollout traces for object-binding trigger design.

This script consumes JSON files written by libero_object_rollout_diagnostics.py and computes
per-rollout features useful for designing evidence/belief-selective gates:

- success / first_contact / wrong-object-type metrics
- endpoint-nearest object statistics from logged actions + object positions
- pre-target-contact endpoint_wrong_rate
- top endpoint-nearest objects

Example:

    python bts-poc/experiments/analyze_openvla_binding_traces.py \
      bts-poc/experiments/runs/openvla_object_mixed_actionpos120.json
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def stem(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return re.sub(r"_\d+$", "", name)


def dist(a: Iterable[float], b: Iterable[float]) -> float:
    aa = list(a)
    bb = list(b)
    return math.sqrt(sum((aa[i] - bb[i]) ** 2 for i in range(3)))


def endpoint_features(rollout: Dict[str, Any]) -> Dict[str, Any]:
    target_stem = stem(rollout.get("target_key_name") or rollout.get("target_object"))
    target_contacted = False
    total = 0
    wrong = 0
    target = 0
    nearest = Counter()
    gated = 0
    debug_rows = 0

    for step in rollout.get("trace", []):
        contact = step.get("contact_object")
        contact_name = contact.get("object") if contact else None
        if contact_name and stem(contact_name) == target_stem:
            target_contacted = True
        if target_contacted:
            continue

        debug = step.get("policy_debug") or {}
        if debug:
            debug_rows += 1
            if debug.get("gate_triggered"):
                gated += 1

        action = step.get("action")
        eef = step.get("robot0_eef_pos")
        objects = step.get("object_positions") or {}
        if not action or not eef or not objects:
            continue

        endpoint = [float(eef[i]) + float(action[i]) for i in range(3)]
        best_name, _best_pos = min(objects.items(), key=lambda kv: dist(endpoint, kv[1]))
        best_stem = stem(best_name)
        nearest[best_name] += 1
        total += 1
        if best_stem == target_stem:
            target += 1
        else:
            wrong += 1

    return {
        "pre_target_endpoint_steps": total,
        "endpoint_wrong_count": wrong,
        "endpoint_target_count": target,
        "endpoint_wrong_rate": (wrong / total) if total else None,
        "endpoint_nearest_top5": nearest.most_common(5),
        "policy_debug_rows": debug_rows,
        "gate_triggered_count": gated,
        "gate_triggered_rate": (gated / debug_rows) if debug_rows else None,
    }


def summarize_rollout(rollout: Dict[str, Any]) -> Dict[str, Any]:
    ts = rollout.get("trace_summary", {})
    first_contact = ts.get("first_contact_object") or {}
    out = {
        "task_id": rollout.get("task_id"),
        "init_id": rollout.get("init_id"),
        "target": rollout.get("target_key_name") or rollout.get("target_object"),
        "success": bool(rollout.get("success_seen")),
        "first_contact": first_contact.get("object"),
        "first_wrong_type": bool(ts.get("first_contact_is_wrong_type")),
        "any_wrong_type": bool(ts.get("any_wrong_type_contact")),
        "any_target_contact": bool(ts.get("any_target_contact")),
        "nearest_target_fraction": ts.get("nearest_target_fraction"),
        "target_dist_drop": ts.get("target_dist_drop"),
    }
    out.update(endpoint_features(rollout))
    return out


def load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_float(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON summary output")
    args = ap.parse_args()

    all_rows: List[Dict[str, Any]] = []
    for path in args.json:
        data = load(path)
        rows = [summarize_rollout(r) for r in data.get("rollouts", [])]
        all_rows.extend({"file": str(path), **r} for r in rows)

        n = len(rows)
        succ = sum(int(r["success"]) for r in rows)
        wrong_any = sum(int(r["any_wrong_type"]) for r in rows)
        first_wrong = sum(int(r["first_wrong_type"]) for r in rows)
        target_contact = sum(int(r["any_target_contact"]) for r in rows)
        endpoint_rates = [r["endpoint_wrong_rate"] for r in rows if r["endpoint_wrong_rate"] is not None]
        gate_rates = [r["gate_triggered_rate"] for r in rows if r["gate_triggered_rate"] is not None]

        print("FILE", path)
        print(
            "SUMMARY",
            "n", n,
            "success", f"{succ}/{n}",
            "any_wrong", f"{wrong_any}/{n}",
            "first_wrong", f"{first_wrong}/{n}",
            "target_contact", f"{target_contact}/{n}",
            "mean_endpoint_wrong", fmt_float(sum(endpoint_rates) / len(endpoint_rates) if endpoint_rates else None),
            "mean_gate_rate", fmt_float(sum(gate_rates) / len(gate_rates) if gate_rates else None),
        )
        for r in rows:
            print(
                f"  t{r['task_id']}i{r['init_id']}",
                "target", r["target"],
                "succ", int(r["success"]),
                "first", r["first_contact"],
                "wrong", int(r["any_wrong_type"]),
                "endpoint_wrong", fmt_float(r["endpoint_wrong_rate"]),
                "gate", fmt_float(r["gate_triggered_rate"]),
                "top", r["endpoint_nearest_top5"][:3],
            )
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
