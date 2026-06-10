from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from envs.gridworld import (
    AmbiguousSpecGridWorld,
    ACTION_NAMES,
    vectorize_obs,
    vectorize_spec,
    task_idx,
)


def run_expert_episode(env: AmbiguousSpecGridWorld, task: str, spec_mode: str, inject_stay: bool = True,
                       dagger_noise: float = 0.0):
    """收一條軌跡並標記每步正確動作。

    dagger_noise=0（預設，v6 行為）：open-loop——預先算最短路專家動作序列，依序執行。
    dagger_noise>0（v7 DAgger-lite）：closed-loop——每步用 expert_next_action 標記**當前 state**
      的正確動作，但以 dagger_noise 機率改執行隨機探索動作（含 stay），強迫資料覆蓋 off-policy
      偏離態。標的永遠是當前 state 的正確動作 → 修 BC covariate shift。
    """
    spec = env.sample_spec_for_task(task, mode=spec_mode)
    spec_vec = vectorize_spec(spec)
    compat0 = env.compatible_tasks_given_state(spec, env.agent_pos)
    steps = []

    if dagger_noise > 0.0:
        # closed-loop DAgger-lite
        for _ in range(env.horizon):
            obs = env.get_obs()
            label = env.expert_next_action(task, spec=spec)  # 當前 state 的正確動作
            steps.append({
                "obs_vec": vectorize_obs(obs, size=env.size, max_objects=env.n_objects),
                "agent_pos": list(obs["agent_pos"]),
                "oracle_posterior": env.oracle_posterior(spec, obs["agent_pos"]),
                "action": label,
                "action_name": ACTION_NAMES[label],
            })
            # 執行：多數時候照 label，偶爾隨機探索（含移動方向，逼出偏離態）
            if env.rng.random() < dagger_noise:
                exec_a = env.rng.randrange(5)
            else:
                exec_a = label
            _, _, done, _ = env.step(exec_a)
            if done:
                break
    else:
        # open-loop（v6 原行為）
        expert_actions = env.expert_trajectory(task, spec=spec)
        if inject_stay and len(expert_actions) > 0:
            noisy_actions = []
            for a in expert_actions:
                noisy_actions.append(a)
                if env.rng.random() < 0.15:
                    noisy_actions.append(4)
            expert_actions = noisy_actions[: env.horizon]
        for a in expert_actions:
            obs = env.get_obs()
            steps.append(
                {
                    "obs_vec": vectorize_obs(obs, size=env.size, max_objects=env.n_objects),
                    "agent_pos": list(obs["agent_pos"]),
                    "oracle_posterior": env.oracle_posterior(spec, obs["agent_pos"]),
                    "action": a,
                    "action_name": ACTION_NAMES[a],
                }
            )
            _, _, done, _ = env.step(a)
            if done:
                break

    obs = env.get_obs()
    final_obs = {
        "obs_vec": vectorize_obs(obs, size=env.size, max_objects=env.n_objects),
        "agent_pos": list(obs["agent_pos"]),
        "oracle_posterior": env.oracle_posterior(spec, obs["agent_pos"]),
        "is_terminal": True,
    }

    record = {
        "task": task,
        "task_idx": task_idx(task),
        "spec": {"color": spec.color, "shape": spec.shape, "text": spec.text},
        "spec_vec": spec_vec,
        "compatible_tasks_t0": compat0,
        "steps": steps,
        "final_obs": final_obs,
        "horizon": env.horizon,
        "grid_size": env.size,
        # v7: 存 partial-obs mode，供 eval rollout 重建同模式環境
        "observe_object_identity": env.observe_object_identity,
        "hint1_pos": list(env.hint1_pos) if env.hint1_pos is not None else None,
        "hint1_attr_kind": env.hint1_attr_kind,
        "hint1_attr_value": env.hint1_attr_value,
        "hint2_pos": list(env.hint2_pos) if env.hint2_pos is not None else None,
        "hint2_attr_kind": env.hint2_attr_kind,
        "hint2_attr_value": env.hint2_attr_value,
        "objects": [
            {"color": o.color, "shape": o.shape, "pos": list(o.pos), "task_id": o.task_id}
            for o in env.objects
        ],
    }
    return record


def split_by_unseen_task(records: List[Dict], test_holdout_tasks: List[str]):
    holdout = set(test_holdout_tasks)
    train, test = [], []
    for r in records:
        if r["task"] in holdout:
            test.append(r)
        else:
            train.append(r)
    return train, test


def generate_dataset(
    out_dir: Path,
    n_episodes: int = 20000,
    size: int = 7,
    horizon: int = 15,
    n_objects: int = 4,
    seed: int = 0,
    spec_mode: str = "mixed",
    test_holdout_tasks: List[str] | None = None,
    n_hints: int = 2,
    observe_object_identity: bool = True,
    dagger_noise: float = 0.0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    env = AmbiguousSpecGridWorld(
        size=size,
        horizon=horizon,
        n_objects=n_objects,
        train_excluded_tasks=[],
        seed=seed,
        n_hints=n_hints,
        observe_object_identity=observe_object_identity,
    )

    records = []
    for ep in range(n_episodes):
        _ = env.reset()
        task = env.target_task
        if task is None:
            continue
        rec = run_expert_episode(env, task, spec_mode=spec_mode, inject_stay=True, dagger_noise=dagger_noise)
        rec["episode_id"] = ep
        records.append(rec)

    test_holdout_tasks = test_holdout_tasks or []
    train_records, test_records = split_by_unseen_task(records, test_holdout_tasks)

    for name, subset in [("train", train_records), ("test", test_records), ("all", records)]:
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rec in subset:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "n_episodes": n_episodes,
        "size": size,
        "horizon": horizon,
        "n_objects": n_objects,
        "seed": seed,
        "spec_mode": spec_mode,
        "n_hints": n_hints,
        "observe_object_identity": observe_object_identity,
        "dagger_noise": dagger_noise,
        "test_holdout_tasks": test_holdout_tasks,
        "n_train": len(train_records),
        "n_test": len(test_records),
        "example_tasks": sorted(list({r["task"] for r in records}))[:10],
        "example_specs": sorted(list({r["spec"]["text"] for r in records}))[:20],
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved dataset to {out_dir}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="data/gridworld_mixed")
    parser.add_argument("--n-episodes", type=int, default=20000)
    parser.add_argument("--size", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--n-objects", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spec-mode", type=str, default="mixed", choices=["exact", "ambiguous", "mixed"])
    parser.add_argument(
        "--test-holdout-tasks",
        type=str,
        default="green_triangle",
        help="comma-separated holdout tasks, e.g. green_triangle,blue_circle",
    )
    parser.add_argument("--n-hints", type=int, default=2, choices=[1, 2],
                        help="1=single-hint (v3 ablation) / 2=dual-hint (v5)")
    parser.add_argument("--observe-object-identity", action=argparse.BooleanOptionalAction, default=True,
                        help="True(預設)=fully-obs(v6)；--no-observe-object-identity=partial-obs(v7 鄰格揭露)")
    parser.add_argument("--dagger-noise", type=float, default=0.0,
                        help="0(預設)=open-loop v6 行為；>0=closed-loop DAgger-lite，每步以此機率執行隨機動作"
                             "但標記當前 state 正確動作，修 BC covariate shift。建議 0.3。")
    args = parser.parse_args()

    holdout = [x for x in args.test_holdout_tasks.split(",") if x]
    generate_dataset(
        out_dir=Path(args.out_dir),
        n_episodes=args.n_episodes,
        size=args.size,
        horizon=args.horizon,
        n_objects=args.n_objects,
        seed=args.seed,
        spec_mode=args.spec_mode,
        test_holdout_tasks=holdout,
        n_hints=args.n_hints,
        observe_object_identity=args.observe_object_identity,
        dagger_noise=args.dagger_noise,
    )


if __name__ == "__main__":
    main()
