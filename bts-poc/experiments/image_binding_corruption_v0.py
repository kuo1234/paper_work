from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from envs.image_binding import ControlledImageBindingBenchmark, extract_image_candidates, diagnose_prediction


def parse_instruction(instr: str):
    parts = instr.lower().split()
    return parts[-2], parts[-1]


def corrupt_image(img: np.ndarray, rng: random.Random, noise_std: float = 0.0, occluder: int = 0) -> np.ndarray:
    out = img.astype(np.float32).copy()
    if noise_std > 0:
        np_rng = np.random.default_rng(rng.randrange(2**32 - 1))
        out += np_rng.normal(0.0, noise_std, size=out.shape)
    if occluder > 0:
        h, w = out.shape[:2]
        x = rng.randrange(0, max(1, w - occluder))
        y = rng.randrange(0, max(1, h - occluder))
        out[y:y + occluder, x:x + occluder] = 245
    return np.clip(out, 0, 255).astype(np.uint8)


def structured_image_policy(ex, image: np.ndarray):
    target_color, target_shape = parse_instruction(ex.instruction)
    cands = extract_image_candidates(image)
    if not cands:
        return 0, cands
    scores = [int(c.color == target_color) + int(c.shape == target_shape) for c in cands]
    pred_cand = int(np.argmax(scores))
    dists = [np.hypot(cands[pred_cand].xy[0] - o.xy[0], cands[pred_cand].xy[1] - o.xy[1]) for o in ex.objects]
    return int(np.argmin(dists)), cands


def eval_setting(split: str, n: int, seed: int, noise_std: float, occluder: int):
    env = ControlledImageBindingBenchmark(seed=seed)
    rng = random.Random(seed + 999)
    counters = Counter()
    det_all = 0
    cand_total = 0
    for _ in range(n):
        ex = env.sample(split)
        img = corrupt_image(ex.image, rng, noise_std=noise_std, occluder=occluder)
        pred, cands = structured_image_policy(ex, img)
        diag = diagnose_prediction(ex, pred)
        for k in ["success", "wrong_object", "wrong_color", "wrong_shape"]:
            counters[k] += int(bool(diag[k]))
        det_all += int(len(cands) == len(ex.objects))
        cand_total += len(cands)
    return {
        "success": counters["success"] / n,
        "wrong_object": counters["wrong_object"] / n,
        "wrong_color": counters["wrong_color"] / n,
        "wrong_shape": counters["wrong_shape"] / n,
        "detected_all": det_all / n,
        "avg_candidates": cand_total / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("runs/image_binding_corruption_v0/summary.json"))
    args = ap.parse_args()
    settings = [
        {"name": "clean", "noise_std": 0.0, "occluder": 0},
        {"name": "noise20", "noise_std": 20.0, "occluder": 0},
        {"name": "noise40", "noise_std": 40.0, "occluder": 0},
        {"name": "occ12", "noise_std": 0.0, "occluder": 12},
        {"name": "noise20_occ12", "noise_std": 20.0, "occluder": 12},
    ]
    payload = {}
    for setting in settings:
        rows = []
        for seed in range(args.seeds):
            rows.append(eval_setting("ood", args.n, seed, setting["noise_std"], setting["occluder"]))
        agg = {}
        for k in rows[0].keys():
            vals = np.array([r[k] for r in rows], dtype=float)
            agg[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
        payload[setting["name"]] = {"setting": setting, "per_seed": rows, "agg": agg}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, r in payload.items():
        a = r["agg"]
        print(f"{name:>14} success={a['success']['mean']:.3f}±{a['success']['std']:.3f} detected_all={a['detected_all']['mean']:.3f} avg_candidates={a['avg_candidates']['mean']:.2f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
