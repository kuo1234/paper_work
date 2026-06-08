from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from envs.image_binding import ControlledImageBindingBenchmark, extract_image_candidates, diagnose_prediction


def parse_instruction(instr: str):
    # "pick the red triangle"
    parts = instr.lower().split()
    return parts[-2], parts[-1]


def structured_image_policy(ex):
    target_color, target_shape = parse_instruction(ex.instruction)
    cands = extract_image_candidates(ex.image)
    if not cands:
        return 0, []
    scores = []
    for c in cands:
        scores.append(int(c.color == target_color) + int(c.shape == target_shape))
    pred_cand = int(np.argmax(scores))
    # Map extracted candidate back to nearest ground-truth object for diagnostic only.
    dists = []
    for o in ex.objects:
        dists.append(np.hypot(cands[pred_cand].xy[0] - o.xy[0], cands[pred_cand].xy[1] - o.xy[1]))
    return int(np.argmin(dists)), cands


def eval_split(split: str, n: int, seed: int):
    env = ControlledImageBindingBenchmark(seed=seed)
    counters = Counter()
    det_counts = Counter()
    examples = []
    for i in range(n):
        ex = env.sample(split)
        pred_idx, cands = structured_image_policy(ex)
        diag = diagnose_prediction(ex, pred_idx)
        for k in ["success", "wrong_object", "wrong_color", "wrong_shape"]:
            counters[k] += int(bool(diag[k]))
        det_counts["n_candidates_total"] += len(cands)
        det_counts["n_examples"] += 1
        det_counts["detected_all"] += int(len(cands) == len(ex.objects))
        if i < 5:
            examples.append({
                "instruction": ex.instruction,
                "gt": [o.text for o in ex.objects],
                "det": [c.text for c in cands],
                "target": diag["target"],
                "pred": diag["pred"],
                "success": diag["success"],
            })
    rates = {k: counters[k] / n for k in ["success", "wrong_object", "wrong_color", "wrong_shape"]}
    rates["detected_all"] = det_counts["detected_all"] / n
    rates["avg_candidates"] = det_counts["n_candidates_total"] / max(1, det_counts["n_examples"])
    return {"split": split, "n": n, "rates": rates, "examples": examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/image_binding_detector_v0/summary.json"))
    args = ap.parse_args()
    res = {split: eval_split(split, args.n, args.seed + i) for i, split in enumerate(["train", "id", "ood"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for split, r in res.items():
        rates = r["rates"]
        print(f"{split:>5} success={rates['success']:.3f} wrong_object={rates['wrong_object']:.3f} wrong_color={rates['wrong_color']:.3f} wrong_shape={rates['wrong_shape']:.3f} detected_all={rates['detected_all']:.3f} avg_candidates={rates['avg_candidates']:.2f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
