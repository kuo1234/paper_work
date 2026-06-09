from __future__ import annotations

"""Collect LIBERO images + object metadata for BTS visual/object evidence training.

This is a lightweight dataset collector for the next BTS/OpenVLA step: learn or diagnose a
visual evidence module that predicts target-object evidence from image + language. It stores
RGB frames plus simulator-derived labels/metadata (object positions, BDDL target instance).

Example on spark native OpenVLA env:

    cd ~/bts-poc/experiments
    MUJOCO_GL=egl PYTHONPATH=~/bts-poc/experiments \
      ~/openvla-spark/.venv/bin/python collect_libero_object_evidence_data.py \
        --suite libero_object --tasks 10 --inits 5 --camera-size 256 \
        --out-dir runs/libero_object_evidence_v0

Output:

    runs/libero_object_evidence_v0/metadata.jsonl
    runs/libero_object_evidence_v0/images/<suite>_tXX_iYY.png
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from libero_object_rollout_diagnostics import parse_bddl_goal, parse_language


def find_obs_key(obs: Dict, obj: str, suffix: str):
    prefix = obj + "_"
    candidates = [k for k in obs.keys() if k.startswith(prefix) and k.endswith(suffix)]
    if candidates:
        return sorted(candidates)[0]
    if obj + suffix in obs:
        return obj + suffix
    return None


def object_positions(obs: Dict) -> Dict[str, List[float]]:
    out = {}
    for k, v in obs.items():
        if k.endswith("_pos") and not k.startswith("robot0") and not k.endswith("to_robot0_eef_pos"):
            out[k[:-4]] = np.asarray(v, dtype=float).tolist()
    return dict(sorted(out.items()))


def rotate_for_openvla(img: np.ndarray) -> np.ndarray:
    """Match OpenVLA LIBERO preprocessing orientation (rotate 180 deg)."""
    return np.ascontiguousarray(img[::-1, ::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--inits", type=int, default=5)
    ap.add_argument("--camera-size", type=int, default=256)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/libero_object_evidence_v0"))
    ap.add_argument("--no-rotate", action="store_true", help="Save raw LIBERO camera orientation instead of OpenVLA-rotated image")
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    out_dir = args.out_dir
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    suite = benchmark.get_benchmark(args.suite)()
    rows = []
    with meta_path.open("w", encoding="utf-8") as f:
        for task_id in range(args.tasks):
            task = suite.get_task(task_id)
            parsed = parse_language(task.language)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            goal_info = parse_bddl_goal(open(bddl, "r", encoding="utf-8", errors="replace").read())
            target_key_name = goal_info.get("goal_object_instance") or parsed.get("target_object")
            receptacle_key_name = goal_info.get("goal_receptacle_instance") or parsed.get("receptacle")

            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=args.camera_size, camera_widths=args.camera_size)
            env.seed(0)
            obs = env.reset()
            init_states = suite.get_task_init_states(task_id)
            max_inits = min(args.inits, len(init_states))
            for init_id in range(max_inits):
                obs = env.set_init_state(init_states[init_id])
                for _ in range(max(0, args.warmup_steps)):
                    obs, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

                img = np.asarray(obs["agentview_image"])
                if not args.no_rotate:
                    img = rotate_for_openvla(img)
                img_name = f"{args.suite}_t{task_id:02d}_i{init_id:02d}.png"
                Image.fromarray(img.astype(np.uint8)).save(img_dir / img_name)

                positions = object_positions(obs)
                target_pos_key = find_obs_key(obs, target_key_name, "_pos") if target_key_name else None
                receptacle_pos_key = find_obs_key(obs, receptacle_key_name, "_pos") if receptacle_key_name else None
                row = {
                    "suite": args.suite,
                    "task_id": task_id,
                    "init_id": init_id,
                    "task_name": task.name,
                    "language": task.language,
                    "parsed_target_object": parsed.get("target_object"),
                    "parsed_receptacle": parsed.get("receptacle"),
                    "goal_target_instance": goal_info.get("goal_object_instance"),
                    "goal_receptacle_instance": goal_info.get("goal_receptacle_instance"),
                    "target_key_name": target_key_name,
                    "receptacle_key_name": receptacle_key_name,
                    "target_pos_key": target_pos_key,
                    "receptacle_pos_key": receptacle_pos_key,
                    "image": str(Path("images") / img_name),
                    "camera_size": args.camera_size,
                    "openvla_rotated": not args.no_rotate,
                    "object_positions": positions,
                    "target_pos": np.asarray(obs[target_pos_key], dtype=float).tolist() if target_pos_key else None,
                    "receptacle_pos": np.asarray(obs[receptacle_pos_key], dtype=float).tolist() if receptacle_pos_key else None,
                }
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                print("wrote", img_name, "target", target_key_name)
            env.close()

    summary = {
        "n": len(rows),
        "suite": args.suite,
        "tasks": args.tasks,
        "inits": args.inits,
        "camera_size": args.camera_size,
        "metadata": str(meta_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("summary", summary)


if __name__ == "__main__":
    main()
