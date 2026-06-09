from __future__ import annotations

"""Build candidate-conditioned evidence rows from LIBERO object metadata.

The failed image-only classifier showed the correct evidence problem is candidate-conditioned:
score whether a visible candidate object matches the target specified by language. This script
turns frame-level metadata into one row per object candidate.

Example:

    python bts-poc/experiments/build_libero_candidate_evidence_rows.py \
      --metadata bts-poc/experiments/runs/libero_object_evidence_v0/metadata.jsonl \
      --out bts-poc/experiments/runs/libero_object_candidate_evidence_v0.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional


def stem(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return re.sub(r"_\d+$", "", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, default=Path("runs/libero_object_evidence_v0/metadata.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("runs/libero_object_candidate_evidence_v0.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_pos = 0
    n_total = 0
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            target_stem = stem(row.get("target_key_name") or row.get("parsed_target_object"))
            for cand_name, cand_pos in sorted((row.get("object_positions") or {}).items()):
                cand_stem = stem(cand_name)
                is_target = bool(target_stem and cand_stem == target_stem)
                out = {
                    "suite": row.get("suite"),
                    "task_id": row.get("task_id"),
                    "init_id": row.get("init_id"),
                    "image": row.get("image"),
                    "language": row.get("language"),
                    "target_name": row.get("target_key_name"),
                    "target_stem": target_stem,
                    "candidate_name": cand_name,
                    "candidate_stem": cand_stem,
                    "candidate_pos": cand_pos,
                    "target_pos": row.get("target_pos"),
                    "is_target": is_target,
                }
                f.write(json.dumps(out) + "\n")
                n_total += 1
                n_pos += int(is_target)

    summary = {
        "frames": len(rows),
        "candidate_rows": n_total,
        "positive_rows": n_pos,
        "negative_rows": n_total - n_pos,
        "positive_rate": n_pos / n_total if n_total else None,
        "out": str(args.out),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
