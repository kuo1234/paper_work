from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def summarize_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rollouts = data.get("rollouts", [])
    summaries = [r.get("trace_summary", {}) for r in rollouts]

    def vals(key):
        return [s.get(key) for s in summaries if s.get(key) is not None]

    def mean(key):
        v = vals(key)
        return float(np.mean(v)) if v else None

    def rate_bool(key):
        v = [bool(s.get(key)) for s in summaries]
        return float(np.mean(v)) if v else None

    first_contacts = [s.get("first_contact_object") for s in summaries]
    contact_count = sum(1 for x in first_contacts if x)
    first_contact_target_rate = rate_bool("first_contact_is_target") if contact_count else None

    return {
        "file": str(path),
        "suite": data.get("suite") or (rollouts[0].get("suite") if rollouts else None),
        "policy": data.get("policy"),
        "n_rollouts": data.get("n_rollouts", len(rollouts)),
        "success_count": data.get("success_count"),
        "mean_target_dist_drop": data.get("mean_target_dist_drop", mean("target_dist_drop")),
        "mean_nearest_target_fraction": data.get("mean_nearest_target_fraction", mean("nearest_target_fraction")),
        "first_nearest_target_rate": data.get("first_nearest_target_rate", rate_bool("first_nearest_is_target")),
        "contact_count": contact_count,
        "first_contact_target_rate": first_contact_target_rate,
        "mean_target_dist_final": mean("target_dist_final"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = [summarize_file(p) for p in args.paths]
    header = [
        "file", "suite", "policy", "n", "success", "drop", "nearest_frac",
        "first_nearest", "contacts", "contact_target", "final_dist",
    ]
    print("\t".join(header))
    for r in rows:
        print("\t".join([
            Path(r["file"]).name,
            str(r.get("suite")),
            str(r.get("policy")),
            str(r.get("n_rollouts")),
            str(r.get("success_count")),
            f"{r['mean_target_dist_drop']:.4f}" if r.get("mean_target_dist_drop") is not None else "NA",
            f"{r['mean_nearest_target_fraction']:.4f}" if r.get("mean_nearest_target_fraction") is not None else "NA",
            f"{r['first_nearest_target_rate']:.4f}" if r.get("first_nearest_target_rate") is not None else "NA",
            str(r.get("contact_count")),
            f"{r['first_contact_target_rate']:.4f}" if r.get("first_contact_target_rate") is not None else "NA",
            f"{r['mean_target_dist_final']:.4f}" if r.get("mean_target_dist_final") is not None else "NA",
        ]))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
