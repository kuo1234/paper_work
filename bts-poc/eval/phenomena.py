from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend，避免遠端無顯示器時報錯
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    # 本地 smoke 環境可能無 matplotlib；JSON metrics 一律輸出，圖只在有 matplotlib 時畫
    plt = None
    _HAS_PLT = False
import torch
import torch.nn.functional as F

from train import EpisodeDataset, collate_fn
from envs.gridworld import AmbiguousSpecGridWorld, ObjSpec, Spec, vectorize_obs
from models.transformer import TinyBTS, SinglePointBaseline


@torch.no_grad()
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    lp = F.log_softmax(logits, dim=-1)
    return -(p * lp).sum(dim=-1)


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    base = TinyBTS(
        obs_dim=ckpt["obs_dim"],
        spec_dim=ckpt["spec_dim"],
        n_tasks=ckpt["n_tasks"],
        n_actions=ckpt["n_actions"],
        d_model=cfg["d_model"],
        nhead=cfg["heads"],
        num_layers=cfg["layers"],
        dim_feedforward=cfg["d_model"] * 2,
    )
    if cfg.get("single_point_baseline", False):
        model = SinglePointBaseline(base)
    else:
        model = base
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


def phenomenon_entropy_gap(model, dataset, device):
    exact = []
    ambiguous = []
    for i in range(len(dataset)):
        item = dataset[i]
        batch = collate_fn([item])
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            spec_vec=batch["spec_vec"],
            hist_obs=batch["obs"][:, :1],
            hist_prev_actions_onehot=batch["prev_actions_oh"][:, :1],
            attention_mask=batch["attn"][:, :1],
        )
        ent = float(entropy_from_logits(out["belief_logits"])[0].item())
        support = int((batch["oracle_posterior"][:, 0] > 0).sum(dim=-1)[0].item())
        if support == 1:
            exact.append(ent)
        else:
            ambiguous.append(ent)
    return {
        "exact_mean": sum(exact) / max(len(exact), 1),
        "ambiguous_mean": sum(ambiguous) / max(len(ambiguous), 1),
        "exact_all": exact,
        "ambiguous_all": ambiguous,
    }


def phenomenon_entropy_decay(model, dataset, device, max_items: int = 200):
    curves = []
    n = min(len(dataset), max_items)
    for i in range(n):
        item = dataset[i]
        ents = []
        batch = collate_fn([item])
        batch = {k: v.to(device) for k, v in batch.items()}
        T = int(batch["lengths"][0].item())
        # 只看歧義規格
        support = int((batch["oracle_posterior"][:, 0] > 0).sum(dim=-1)[0].item())
        if support == 1:
            continue
        for t in range(T):
            out = model(
                spec_vec=batch["spec_vec"],
                hist_obs=batch["obs"][:, : t + 1],
                hist_prev_actions_onehot=batch["prev_actions_oh"][:, : t + 1],
                attention_mask=batch["attn"][:, : t + 1],
            )
            ent = float(entropy_from_logits(out["belief_logits"])[0].item())
            ents.append(ent)
        curves.append(ents)
    return curves


def rollout_episode(model, item: Dict, device: torch.device, horizon_limit: int = 50):
    # 從資料記錄重建環境，讓模型自己 rollout（不 teacher forcing）
    # 回傳 (success: bool, total_return: float)
    objects = [ObjSpec(o["color"], o["shape"], tuple(o["pos"])) for o in item["objects"]]
    # v7: 用資料記錄的 observe_object_identity 重建同模式環境（fully/partial-obs）
    observe_obj_id = item.get("observe_object_identity", True)
    env = AmbiguousSpecGridWorld(
        size=item["grid_size"], horizon=item["horizon"], n_objects=len(objects), seed=0,
        observe_object_identity=observe_obj_id,
    )
    env.set_episode(
        objects=objects,
        target_task=item["task"],
        hint1_pos=tuple(item["hint1_pos"]) if item.get("hint1_pos") is not None else None,
        hint1_attr_kind=item.get("hint1_attr_kind"),
        hint1_attr_value=item.get("hint1_attr_value"),
        hint2_pos=tuple(item["hint2_pos"]) if item.get("hint2_pos") is not None else None,
        hint2_attr_kind=item.get("hint2_attr_kind"),
        hint2_attr_value=item.get("hint2_attr_value"),
        hint1_revealed=False,
        hint2_revealed=False,
        observe_object_identity=observe_obj_id,
    )
    spec = Spec(**item["spec"])
    spec_vec = torch.tensor([item["spec_vec"]], dtype=torch.float, device=device)

    hist_obs = []
    hist_prev = []
    prev_action = None
    total_return = 0.0
    for t in range(min(item["horizon"], horizon_limit)):
        obs = env.get_obs()
        hist_obs.append(vectorize_obs(obs, size=env.size, max_objects=env.n_objects))
        prev_oh = [0.0] * 5 if prev_action is None else [1.0 if i == prev_action else 0.0 for i in range(5)]
        hist_prev.append(prev_oh)

        hist_obs_t = torch.tensor([hist_obs], dtype=torch.float, device=device)
        hist_prev_t = torch.tensor([hist_prev], dtype=torch.float, device=device)
        attn = torch.ones(1, len(hist_obs), dtype=torch.float, device=device)
        out = model(spec_vec=spec_vec, hist_obs=hist_obs_t, hist_prev_actions_onehot=hist_prev_t, attention_mask=attn)
        action = int(torch.argmax(out["action_logits"], dim=-1)[0].item())
        prev_action = action
        _, reward, done, info = env.step(action)
        total_return += float(reward)
        if done:
            return bool(info["success"]), total_return
    return False, total_return


def phenomenon_single_vs_belief(belief_model, baseline_model, dataset, device, max_items: int = 200):
    # 只統計歧義規格，否則精確規格下兩者本就應接近
    n = min(len(dataset), max_items)
    belief_ok = 0
    single_ok = 0
    belief_return = 0.0
    single_return = 0.0
    total = 0
    for i in range(n):
        item = dataset[i]
        if len(item["compatible_tasks_t0"]) <= 1:
            continue
        total += 1
        b_succ, b_ret = rollout_episode(belief_model, item, device)
        s_succ, s_ret = rollout_episode(baseline_model, item, device)
        belief_ok += int(b_succ)
        single_ok += int(s_succ)
        belief_return += b_ret
        single_return += s_ret
    denom = max(total, 1)
    belief_success_rate = belief_ok / denom
    single_success_rate = single_ok / denom
    belief_avg_return = belief_return / denom
    single_avg_return = single_return / denom
    return {
        "n_eval": total,
        "belief_success_rate": belief_success_rate,
        "single_success_rate": single_success_rate,
        "belief_avg_return": belief_avg_return,
        "single_avg_return": single_avg_return,
        "success_gap": belief_success_rate - single_success_rate,
        "return_gap": belief_avg_return - single_avg_return,  # v4 主判準
        "supports": belief_avg_return > single_avg_return,
    }


def pad_mean(curves: List[List[float]]) -> List[float]:
    if not curves:
        return []
    T = max(len(c) for c in curves)
    out = []
    for t in range(T):
        vals = [c[t] for c in curves if t < len(c)]
        out.append(sum(vals) / len(vals))
    return out


def entropy_decay_metrics(curves: List[List[float]]) -> Dict:
    """
    用穩健的方式判讀「觀察後 entropy 是否下降」：
    - per-episode 用「初始 entropy vs 該 episode 後段平均 entropy」算 drop，
      再對所有歧義 episode 平均，避免 first-vs-last 單格的 survivorship 噪音。
    - 也回報整體 mean_curve 的線性斜率（負代表下降）。
    """
    if not curves:
        return {"n_curves": 0, "supported": False}

    per_ep_drop = []
    for c in curves:
        if len(c) < 2:
            continue
        head = c[0]
        tail_len = max(1, len(c) // 2)
        tail = sum(c[-tail_len:]) / tail_len  # 後半段平均
        per_ep_drop.append(head - tail)  # 正值 = 有下降

    mean_curve = pad_mean(curves)
    # 線性斜率
    n = len(mean_curve)
    if n >= 2:
        xs = list(range(n))
        mx = sum(xs) / n
        my = sum(mean_curve) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, mean_curve))
        den = sum((x - mx) ** 2 for x in xs) or 1.0
        slope = num / den
    else:
        slope = 0.0

    mean_drop = sum(per_ep_drop) / max(len(per_ep_drop), 1)
    return {
        "n_curves": len(curves),
        "init_entropy_mean": mean_curve[0] if mean_curve else None,
        "converged_entropy_mean": (sum(mean_curve[-max(1, n // 2):]) / max(1, n // 2)) if mean_curve else None,
        "per_episode_mean_drop": mean_drop,   # >0 代表收斂
        "slope": slope,                       # <0 代表下降
        "supported": (mean_drop > 0.0) and (slope < 0.0),
        "mean_curve": mean_curve,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--baseline-ckpt", type=str, default="")
    parser.add_argument("--test-jsonl", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="runs/bts_poc/figs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, ckpt = load_model(args.ckpt, device)
    baseline_model = None
    if args.baseline_ckpt:
        baseline_model, _ = load_model(args.baseline_ckpt, device)
    test_ds = EpisodeDataset(args.test_jsonl)

    # 現象1：歧義規格 entropy > 精確規格 entropy
    gap = phenomenon_entropy_gap(model, test_ds, device)
    with (out_dir / "phenomenon1_entropy_gap.json").open("w", encoding="utf-8") as f:
        json.dump(gap, f, ensure_ascii=False, indent=2)

    if _HAS_PLT:
        plt.figure(figsize=(5, 4))
        plt.boxplot([gap["exact_all"], gap["ambiguous_all"]], labels=["exact", "ambiguous"])
        plt.ylabel("belief entropy at t=0")
        plt.title("Phenomenon 1: ambiguity -> higher belief entropy")
        plt.tight_layout()
        plt.savefig(out_dir / "phenomenon1_entropy_gap.png", dpi=150)
        plt.close()
    # 現象2：隨觀察 entropy 下降
    curves = phenomenon_entropy_decay(model, test_ds, device)
    p2 = entropy_decay_metrics(curves)
    mean_curve = p2.get("mean_curve", [])
    with (out_dir / "phenomenon2_entropy_decay.json").open("w", encoding="utf-8") as f:
        json.dump(p2, f, ensure_ascii=False, indent=2)

    if mean_curve and _HAS_PLT:
        plt.figure(figsize=(6, 4))
        plt.plot(list(range(len(mean_curve))), mean_curve, marker="o")
        plt.xlabel("context timestep")
        plt.ylabel("belief entropy")
        plt.title("Phenomenon 2: observation -> entropy decay")
        plt.tight_layout()
        plt.savefig(out_dir / "phenomenon2_entropy_decay.png", dpi=150)
        plt.close()

    result = {
        "phenomenon1": gap,
        "phenomenon2": {
            "init_entropy_mean": p2.get("init_entropy_mean"),
            "converged_entropy_mean": p2.get("converged_entropy_mean"),
            "per_episode_mean_drop": p2.get("per_episode_mean_drop"),
            "slope": p2.get("slope"),
            "supported": p2.get("supported"),
        },
    }

    if baseline_model is not None:
        p3 = phenomenon_single_vs_belief(model, baseline_model, test_ds, device)
        with (out_dir / "phenomenon3_belief_vs_single.json").open("w", encoding="utf-8") as f:
            json.dump(p3, f, ensure_ascii=False, indent=2)
        result["phenomenon3"] = p3

    print(f"Saved phenomenon analysis to {out_dir}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
