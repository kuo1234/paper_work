from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from envs.image_binding import (
    ControlledImageBindingBenchmark,
    diagnose_prediction,
    location_prior_policy,
    oracle_language_policy,
)


def eval_policy(env: ControlledImageBindingBenchmark, split: str, n: int, policy_name: str):
    if policy_name == "location_prior":
        policy = location_prior_policy
    elif policy_name == "oracle_language":
        policy = oracle_language_policy
    else:
        raise ValueError(policy_name)

    counters = Counter()
    by_combo = defaultdict(Counter)
    examples = []
    for i in range(n):
        ex = env.sample(split)
        pred = policy(ex)
        diag = diagnose_prediction(ex, pred)
        for k in ["success", "wrong_object", "wrong_color", "wrong_shape"]:
            counters[k] += int(bool(diag[k]))
        by_combo["_".join(ex.target_combo)]["n"] += 1
        by_combo["_".join(ex.target_combo)]["success"] += int(bool(diag["success"]))
        if i < 8:
            examples.append({
                "instruction": ex.instruction,
                "target": diag["target"],
                "pred": diag["pred"],
                "success": diag["success"],
                "objects": [{"text": o.text, "xy": list(o.xy)} for o in ex.objects],
            })
    rates = {k: counters[k] / n for k in ["success", "wrong_object", "wrong_color", "wrong_shape"]}
    combo_rates = {k: {"n": v["n"], "success": v["success"] / max(1, v["n"])} for k, v in by_combo.items()}
    return {"split": split, "policy": policy_name, "n": n, "rates": rates, "by_combo": combo_rates, "examples": examples}


def save_examples(out_dir: Path, seed: int):
    try:
        from PIL import Image, ImageDraw
    except Exception as e:  # pragma: no cover
        print(f"PIL unavailable, skip image save: {e}")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    env = ControlledImageBindingBenchmark(seed=seed)
    saved = []
    for split in ["train", "ood"]:
        for i in range(4):
            ex = env.sample(split)
            img = Image.fromarray(ex.image)
            draw = ImageDraw.Draw(img)
            for j, o in enumerate(ex.objects):
                x, y = o.xy
                label = "T" if j == ex.target_index else str(j)
                draw.text((x - 4, y - 4), label, fill=(0, 0, 0))
            path = out_dir / f"{split}_{i}_{ex.target_combo[0]}_{ex.target_combo[1]}.png"
            img.save(path)
            saved.append(str(path))
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/image_binding_v0"))
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = ControlledImageBindingBenchmark(seed=args.seed)
    results = []
    for policy in ["location_prior", "oracle_language"]:
        for split in ["train", "id", "ood"]:
            results.append(eval_policy(env, split, args.n, policy))
    image_paths = save_examples(args.out_dir / "examples", args.seed) if args.save_images else []
    payload = {"results": results, "image_paths": image_paths}
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for r in results:
        rates = r["rates"]
        print(
            f"{r['policy']:>15} {r['split']:>5} "
            f"success={rates['success']:.3f} wrong_object={rates['wrong_object']:.3f} "
            f"wrong_color={rates['wrong_color']:.3f} wrong_shape={rates['wrong_shape']:.3f}"
        )
    if image_paths:
        print("saved_examples")
        for p in image_paths:
            print(p)
    print("wrote", args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
