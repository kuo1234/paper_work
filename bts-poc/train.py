from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.transformer import TinyBTS, SinglePointBaseline, make_prev_action_onehot, sequence_attention_mask


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class EpisodeDataset(Dataset):
    def __init__(self, path: str):
        self.records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        steps = rec["steps"]
        T = len(steps)
        obs = [s["obs_vec"] for s in steps]
        actions = [s["action"] for s in steps]
        post = [s["oracle_posterior"] for s in steps]
        item = {
            "spec_vec": rec["spec_vec"],
            "obs": obs,
            "actions": actions,
            "oracle_posterior": post,
            "length": T,
            "task_idx": rec["task_idx"],
            "spec_text": rec["spec"]["text"],
            "compatible_tasks_t0": rec["compatible_tasks_t0"],
        }
        item.update(rec)
        return item


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    T = int(lengths.max().item())
    obs_dim = len(batch[0]["obs"][0])
    spec_dim = len(batch[0]["spec_vec"])
    n_tasks = len(batch[0]["oracle_posterior"][0])

    spec = torch.zeros(B, spec_dim)
    obs = torch.zeros(B, T, obs_dim)
    actions = torch.zeros(B, T, dtype=torch.long)
    post = torch.zeros(B, T, n_tasks)

    for i, b in enumerate(batch):
        spec[i] = torch.tensor(b["spec_vec"], dtype=torch.float)
        L = b["length"]
        obs[i, :L] = torch.tensor(b["obs"], dtype=torch.float)
        actions[i, :L] = torch.tensor(b["actions"], dtype=torch.long)
        post[i, :L] = torch.tensor(b["oracle_posterior"], dtype=torch.float)

    attn = sequence_attention_mask(lengths, T)
    prev_actions_oh = make_prev_action_onehot(actions, n_actions=5)

    return {
        "spec_vec": spec,
        "obs": obs,
        "actions": actions,
        "oracle_posterior": post,
        "lengths": lengths,
        "attn": attn,
        "prev_actions_oh": prev_actions_oh,
    }


# ------------------------------------------------------------
# Losses / metrics
# ------------------------------------------------------------
def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits [B,A], targets [B], mask [B]
    loss = F.cross_entropy(logits, targets, reduction="none")
    loss = loss * mask.float()
    return loss.sum() / mask.float().sum().clamp_min(1.0)


def masked_belief_kl(belief_logits: torch.Tensor, target_post: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # belief_logits [B,K], target_post [B,K], mask [B]
    log_b = F.log_softmax(belief_logits, dim=-1)
    # target_post already sums to 1 on valid rows
    kl = F.kl_div(log_b, target_post, reduction="none").sum(dim=-1)
    kl = kl * mask.float()
    return kl.sum() / mask.float().sum().clamp_min(1.0)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    lp = F.log_softmax(logits, dim=-1)
    return -(p * lp).sum(dim=-1)


# ------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_action_loss = 0.0
    total_belief_loss = 0.0
    total_entropy_exact = 0.0
    total_entropy_amb = 0.0
    n_exact = 0
    n_amb = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B, T, _ = batch["obs"].shape

        # 逐步 teacher-forcing：每個 t 用 history[:t+1] 的最後 token 預測 a_t / b_t
        for t in range(T):
            valid = (batch["attn"][:, t] > 0.5)
            if valid.sum() == 0:
                continue
            out = model(
                spec_vec=batch["spec_vec"],
                hist_obs=batch["obs"][:, : t + 1],
                hist_prev_actions_onehot=batch["prev_actions_oh"][:, : t + 1],
                attention_mask=batch["attn"][:, : t + 1],
            )
            total_action_loss += float(masked_cross_entropy(out["action_logits"], batch["actions"][:, t], valid).item())
            total_belief_loss += float(masked_belief_kl(out["belief_logits"], batch["oracle_posterior"][:, t], valid).item())

            ent = entropy_from_logits(out["belief_logits"])  # [B]
            # 用 t=0 的 oracle posterior 支援度判斷 exact/ambiguous
            if t == 0:
                support = (batch["oracle_posterior"][:, 0] > 0).sum(dim=-1)
                exact = valid & (support == 1)
                amb = valid & (support > 1)
                if exact.any():
                    total_entropy_exact += float(ent[exact].sum().item())
                    n_exact += int(exact.sum().item())
                if amb.any():
                    total_entropy_amb += float(ent[amb].sum().item())
                    n_amb += int(amb.sum().item())

    denom = max(len(loader), 1)
    return {
        "action_loss": total_action_loss / denom,
        "belief_kl": total_belief_loss / denom,
        "entropy_exact_t0": total_entropy_exact / max(n_exact, 1),
        "entropy_amb_t0": total_entropy_amb / max(n_amb, 1),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=str, default="data/gridworld_mixed/train.jsonl")
    parser.add_argument("--test-jsonl", type=str, default="data/gridworld_mixed/test.jsonl")
    parser.add_argument("--out-dir", type=str, default="runs/bts_poc")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-belief", type=float, default=0.5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--single-point-baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # v6: 讓訓練可重現（多 seed 實驗用）
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = EpisodeDataset(args.train_jsonl)
    test_ds = EpisodeDataset(args.test_jsonl)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, generator=loader_generator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # infer dims from one batch
    sample = collate_fn([train_ds[0]])
    obs_dim = sample["obs"].shape[-1]
    spec_dim = sample["spec_vec"].shape[-1]
    n_tasks = sample["oracle_posterior"].shape[-1]
    n_actions = 5

    device = torch.device(args.device)
    base = TinyBTS(
        obs_dim=obs_dim,
        spec_dim=spec_dim,
        n_tasks=n_tasks,
        n_actions=n_actions,
        d_model=args.d_model,
        nhead=args.heads,
        num_layers=args.layers,
        dim_feedforward=args.d_model * 2,
    )
    model: nn.Module = SinglePointBaseline(base) if args.single_point_baseline else base
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    best_metric = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_action = 0.0
        total_belief = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            B, T, _ = batch["obs"].shape
            loss_action_accum = 0.0
            loss_belief_accum = 0.0
            steps_used = 0

            # 逐步 teacher-forcing：每個 t 用 history[:t+1] 的 context 預測當前 action/belief
            for t in range(T):
                valid = (batch["attn"][:, t] > 0.5)
                if valid.sum() == 0:
                    continue
                out = model(
                    spec_vec=batch["spec_vec"],
                    hist_obs=batch["obs"][:, : t + 1],
                    hist_prev_actions_onehot=batch["prev_actions_oh"][:, : t + 1],
                    attention_mask=batch["attn"][:, : t + 1],
                )
                loss_action = masked_cross_entropy(out["action_logits"], batch["actions"][:, t], valid)
                loss_belief = masked_belief_kl(out["belief_logits"], batch["oracle_posterior"][:, t], valid)
                loss_action_accum = loss_action_accum + loss_action
                loss_belief_accum = loss_belief_accum + loss_belief
                steps_used += 1

            if steps_used == 0:
                continue
            loss_action_accum = loss_action_accum / steps_used
            loss_belief_accum = loss_belief_accum / steps_used
            loss = loss_action_accum + args.lambda_belief * loss_belief_accum

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_action += float(loss_action_accum.item())
            total_belief += float(loss_belief_accum.item())
            n_batches += 1

        train_log = {
            "epoch": epoch,
            "train_action_loss": total_action / max(n_batches, 1),
            "train_belief_kl": total_belief / max(n_batches, 1),
        }
        test_log = evaluate(model, test_loader, device)
        log = {**train_log, **test_log}
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))

        score = test_log["action_loss"] + args.lambda_belief * test_log["belief_kl"]
        if score < best_metric:
            best_metric = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": vars(args),
                    "obs_dim": obs_dim,
                    "spec_dim": spec_dim,
                    "n_tasks": n_tasks,
                    "n_actions": n_actions,
                },
                out_dir / "best.pt",
            )

    with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved best checkpoint to {out_dir / 'best.pt'}")
    print(f"Saved metrics to {out_dir / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
