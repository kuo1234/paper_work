#!/usr/bin/env python3
"""路徑 2 診斷：分解 belief vs single 的 rollout 結局，確認「belief 走到了但低效」假設。

對給定的 belief/single checkpoint + test set，逐 episode rollout（只看歧義規格），
分類結局為 success / wrong_object / timeout，並記步數，看 belief 模型的 return 劣勢
來自哪裡（繞路步數多？踩 wrong-object？）。

用法（遠端）：
  python experiments/diagnose_p3.py \
    --belief-ckpt runs/exp/dual_partial_s1_belief/best.pt \
    --single-ckpt runs/exp/dual_partial_s1_single/best.pt \
    --test-jsonl data/exp/dual_partial_s1/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import torch

from train import EpisodeDataset
from envs.gridworld import AmbiguousSpecGridWorld, ObjSpec, Spec, vectorize_obs
from eval.phenomena import load_model


@torch.no_grad()
def rollout_detail(model, item, device, horizon_limit: int = 50, sample: bool = False, temperature: float = 1.0):
    """回傳 dict：outcome(success/wrong_object/timeout)、steps、total_return。
    sample=False: greedy argmax decoding（現狀）。
    sample=True: 從 action softmax 抽樣（multi-modal decoding 判別測試）。"""
    objects = [ObjSpec(o["color"], o["shape"], tuple(o["pos"])) for o in item["objects"]]
    observe_obj_id = item.get("observe_object_identity", True)
    env = AmbiguousSpecGridWorld(
        size=item["grid_size"], horizon=item["horizon"], n_objects=len(objects), seed=0,
        observe_object_identity=observe_obj_id,
    )
    env.set_episode(
        objects=objects, target_task=item["task"],
        hint1_pos=tuple(item["hint1_pos"]) if item.get("hint1_pos") is not None else None,
        hint1_attr_kind=item.get("hint1_attr_kind"), hint1_attr_value=item.get("hint1_attr_value"),
        hint2_pos=tuple(item["hint2_pos"]) if item.get("hint2_pos") is not None else None,
        hint2_attr_kind=item.get("hint2_attr_kind"), hint2_attr_value=item.get("hint2_attr_value"),
        hint1_revealed=False, hint2_revealed=False, observe_object_identity=observe_obj_id,
    )
    spec_vec = torch.tensor([item["spec_vec"]], dtype=torch.float, device=device)
    hist_obs, hist_prev = [], []
    prev_action = None
    total_return = 0.0
    steps = 0
    for t in range(min(item["horizon"], horizon_limit)):
        obs = env.get_obs()
        hist_obs.append(vectorize_obs(obs, size=env.size, max_objects=env.n_objects))
        prev_oh = [0.0] * 5 if prev_action is None else [1.0 if i == prev_action else 0.0 for i in range(5)]
        hist_prev.append(prev_oh)
        out = model(
            spec_vec=spec_vec,
            hist_obs=torch.tensor([hist_obs], dtype=torch.float, device=device),
            hist_prev_actions_onehot=torch.tensor([hist_prev], dtype=torch.float, device=device),
            attention_mask=torch.ones(1, len(hist_obs), dtype=torch.float, device=device),
        )
        if sample:
            logits = out["action_logits"][0] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, 1)[0].item())
        else:
            action = int(torch.argmax(out["action_logits"], dim=-1)[0].item())
        prev_action = action
        _, reward, done, info = env.step(action)
        total_return += float(reward)
        steps += 1
        if done:
            if info["success"]:
                return {"outcome": "success", "steps": steps, "return": total_return}
            if info["wrong_object"]:
                return {"outcome": "wrong_object", "steps": steps, "return": total_return}
            break
    return {"outcome": "timeout", "steps": steps, "return": total_return}


def summarize(model, ds, device, max_items: int = 300, sample: bool = False, temperature: float = 1.0):
    n = min(len(ds), max_items)
    buckets = {"success": [], "wrong_object": [], "timeout": []}
    total = 0
    for i in range(n):
        item = ds[i]
        if len(item["compatible_tasks_t0"]) <= 1:  # 只看歧義（與 P3 eval 一致）
            continue
        total += 1
        d = rollout_detail(model, item, device, sample=sample, temperature=temperature)
        buckets[d["outcome"]].append(d)
    def stat(name):
        rows = buckets[name]
        if not rows:
            return f"{name}: 0"
        avg_steps = sum(r["steps"] for r in rows) / len(rows)
        return f"{name}: {len(rows)} ({100*len(rows)/max(total,1):.1f}%) avg_steps={avg_steps:.2f}"
    succ_steps = [r["steps"] for r in buckets["success"]]
    return {
        "total": total,
        "lines": [stat("success"), stat("wrong_object"), stat("timeout")],
        "success_avg_steps": (sum(succ_steps) / len(succ_steps)) if succ_steps else None,
        "wrong_rate": len(buckets["wrong_object"]) / max(total, 1),
        "timeout_rate": len(buckets["timeout"]) / max(total, 1),
        "success_rate": len(buckets["success"]) / max(total, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--belief-ckpt", required=True)
    ap.add_argument("--single-ckpt", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-items", type=int, default=300)
    ap.add_argument("--sample", action="store_true",
                    help="rollout 用 softmax 抽樣（multi-modal decoding 判別測試），預設 greedy argmax")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device(args.device)
    ds = EpisodeDataset(args.test_jsonl)
    belief, _ = load_model(args.belief_ckpt, device)
    single, _ = load_model(args.single_ckpt, device)

    decode = f"SAMPLE(T={args.temperature})" if args.sample else "GREEDY-argmax"
    print("=" * 64)
    print(f"P3 診斷  test={args.test_jsonl}  decoding={decode}")
    print("=" * 64)
    for name, m in [("BELIEF", belief), ("SINGLE", single)]:
        s = summarize(m, ds, device, args.max_items, sample=args.sample, temperature=args.temperature)
        print(f"\n[{name}]  歧義 episodes={s['total']}")
        for line in s["lines"]:
            print(f"  {line}")
        print(f"  -> success_rate={s['success_rate']:.3f} wrong_rate={s['wrong_rate']:.3f} "
              f"timeout_rate={s['timeout_rate']:.3f} success_avg_steps={s['success_avg_steps']}")

    print("\n" + "-" * 64)
    print("假設檢查：belief 若『走到了但低效』，應見 belief 的 success_avg_steps > single，")
    print("或 belief 的 timeout_rate / wrong_rate 較高（猶豫繞路 / 中途踩錯）。")


if __name__ == "__main__":
    main()
