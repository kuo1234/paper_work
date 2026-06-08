from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np


def normalize_obj(s: str) -> str:
    return s.lower().replace(" ", "_").replace("-", "_")


def parse_language(language: str) -> Dict[str, str | None]:
    s = language.lower()
    m = re.search(r"pick up the (.*?) and place it in the (.*)$", s)
    if not m:
        return {"target_object": None, "receptacle": None}
    return {"target_object": normalize_obj(m.group(1)), "receptacle": normalize_obj(m.group(2))}


def find_obs_key(obs: Dict, obj: str, suffix: str):
    prefix = obj + "_"
    candidates = [k for k in obs.keys() if k.startswith(prefix) and k.endswith(suffix)]
    if candidates:
        return sorted(candidates)[0]
    return None


def nearest_object(obs: Dict, object_names: List[str], eef_key: str = "robot0_eef_pos"):
    if eef_key not in obs:
        return None
    eef = np.asarray(obs[eef_key])
    best = None
    for name in object_names:
        k = find_obs_key(obs, name, "_pos")
        if not k:
            continue
        d = float(np.linalg.norm(np.asarray(obs[k]) - eef))
        if best is None or d < best["dist"]:
            best = {"object": name, "obs_key": k, "dist": d}
    return best


def summarize_trace(trace: List[Dict], target: str | None) -> Dict:
    nearest = [t.get("nearest_object") for t in trace if t.get("nearest_object")]
    nearest_names = [n.get("object") for n in nearest if n]
    target_prefix = f"{target}_" if target else None
    is_target_nearest = [bool(name and target_prefix and name.startswith(target_prefix)) for name in nearest_names]
    first_nearest = nearest_names[0] if nearest_names else None
    first_target_nearest_t = None
    for i, ok in enumerate(is_target_nearest):
        if ok:
            first_target_nearest_t = i
            break
    dists = [t.get("target_dist_to_eef") for t in trace if t.get("target_dist_to_eef") is not None]
    return {
        "first_nearest_object": first_nearest,
        "first_nearest_is_target": bool(is_target_nearest[0]) if is_target_nearest else False,
        "nearest_target_fraction": float(sum(is_target_nearest) / len(is_target_nearest)) if is_target_nearest else None,
        "first_target_nearest_t": first_target_nearest_t,
        "target_dist_initial": dists[0] if dists else None,
        "target_dist_final": dists[-1] if dists else None,
        "target_dist_drop": (dists[0] - dists[-1]) if len(dists) >= 2 else None,
    }


def rollout_task(task_id: int, init_id: int, steps: int, policy: str, camera_size: int):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark("libero_object")()
    task = suite.get_task(task_id)
    parsed = parse_language(task.language)
    target = parsed["target_object"]
    receptacle = parsed["receptacle"]
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=camera_size, camera_widths=camera_size)
    obs = env.reset()
    init_states = suite.get_task_init_states(task_id)
    if init_id < len(init_states):
        obs = env.set_init_state(init_states[init_id])

    object_names = []
    for k in obs.keys():
        if k.endswith("_pos") and not k.startswith("robot0") and not k.endswith("to_robot0_eef_pos"):
            object_names.append(k[:-4])
    object_names = sorted(set(object_names))

    rng = np.random.default_rng(task_id * 1000 + init_id)
    trace = []
    success_seen = False
    for t in range(steps):
        target_key = find_obs_key(obs, target, "_pos") if target else None
        receptacle_key = find_obs_key(obs, receptacle, "_pos") if receptacle else None
        target_dist = None
        target_to_receptacle = None
        if target_key and "robot0_eef_pos" in obs:
            target_dist = float(np.linalg.norm(np.asarray(obs[target_key]) - np.asarray(obs["robot0_eef_pos"])))
        if target_key and receptacle_key:
            target_to_receptacle = float(np.linalg.norm(np.asarray(obs[target_key]) - np.asarray(obs[receptacle_key])))
        near = nearest_object(obs, object_names)
        trace.append({
            "t": t,
            "target_dist_to_eef": target_dist,
            "target_dist_to_receptacle": target_to_receptacle,
            "nearest_object": near,
            "agentview_shape": list(obs["agentview_image"].shape) if "agentview_image" in obs else None,
            "wrist_shape": list(obs["robot0_eye_in_hand_image"].shape) if "robot0_eye_in_hand_image" in obs else None,
        })
        if policy == "noop":
            action = [0.0] * 7
        elif policy == "random":
            action = rng.normal(0.0, 0.05, size=7).tolist()
            action[-1] = 0.0
        elif policy in {"target_reach", "target_reach_fast"}:
            # Simple object-state heuristic: move end-effector toward parsed target object.
            # This is diagnostic-only, not a task policy. It verifies that target-distance
            # metrics respond to behavior and that parsed target keys are actionable.
            action = [0.0] * 7
            if target_key and "robot0_eef_pos" in obs:
                delta = np.asarray(obs[target_key]) - np.asarray(obs["robot0_eef_pos"])
                gain = 2.0 if policy == "target_reach" else 5.0
                clip = 0.08 if policy == "target_reach" else 0.20
                action[:3] = np.clip(gain * delta, -clip, clip).tolist()
            action[-1] = 0.0
        else:
            raise ValueError(policy)
        obs, reward, done, info = env.step(action)
        success_seen = success_seen or bool(reward > 0 or done)
        if done:
            break
    env.close()
    trace_summary = summarize_trace(trace, target)
    return {
        "task_id": task_id,
        "init_id": init_id,
        "task_name": task.name,
        "language": task.language,
        "target_object": target,
        "receptacle": receptacle,
        "object_names": object_names,
        "policy": policy,
        "steps_requested": steps,
        "steps_recorded": len(trace),
        "success_seen": success_seen,
        "trace_summary": trace_summary,
        "trace": trace,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--inits", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--policy", choices=["noop", "random", "target_reach", "target_reach_fast"], default="noop")
    ap.add_argument("--camera-size", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("runs/libero_object_rollout_diag.json"))
    args = ap.parse_args()
    rows = []
    for task_id in range(args.tasks):
        for init_id in range(args.inits):
            rows.append(rollout_task(task_id, init_id, args.steps, args.policy, args.camera_size))
            print("done", task_id, init_id)
    drops = [r["trace_summary"].get("target_dist_drop") for r in rows if r["trace_summary"].get("target_dist_drop") is not None]
    nearest_fracs = [r["trace_summary"].get("nearest_target_fraction") for r in rows if r["trace_summary"].get("nearest_target_fraction") is not None]
    first_nearest_hits = [r["trace_summary"].get("first_nearest_is_target") for r in rows]
    summary = {
        "n_rollouts": len(rows),
        "policy": args.policy,
        "success_count": sum(int(r["success_seen"]) for r in rows),
        "mean_target_dist_drop": float(np.mean(drops)) if drops else None,
        "mean_nearest_target_fraction": float(np.mean(nearest_fracs)) if nearest_fracs else None,
        "first_nearest_target_rate": float(np.mean(first_nearest_hits)) if first_nearest_hits else None,
        "rollouts": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "n_rollouts", summary["n_rollouts"],
        "success_count", summary["success_count"],
        "mean_drop", summary["mean_target_dist_drop"],
        "nearest_target_frac", summary["mean_nearest_target_fraction"],
        "first_nearest_target_rate", summary["first_nearest_target_rate"],
    )
    for r in rows:
        ts = r["trace_summary"]
        print(
            r["task_id"], r["init_id"], r["target_object"],
            "first_nearest", ts.get("first_nearest_object"),
            "first_hit", ts.get("first_nearest_is_target"),
            "nearest_frac", ts.get("nearest_target_fraction"),
            "drop", ts.get("target_dist_drop"),
        )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
