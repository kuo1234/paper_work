from __future__ import annotations

import argparse
from random import Random

import numpy as np
import torch

from ..configs.gridworld import ENV_CONFIG, TRAINING_CONFIG
from ..envs.gridworld import GridWorldTask
from ..models.varibad import VariBAD
from ..trainers.a2c import a2c_losses
from ..trainers.varibad_vae import vae_losses
from ..utils.torch_rollout import collect_varibad_rollout


def make_env() -> GridWorldTask:
    return GridWorldTask(
        grid_size=ENV_CONFIG["grid_size"],
        allowed_goals=ENV_CONFIG["allowed_goals"],
        start_state=ENV_CONFIG["start_state"],
        episode_horizon=ENV_CONFIG["episode_horizon"],
        reward_goal=ENV_CONFIG["reward_goal"],
        reward_non_goal=ENV_CONFIG["reward_non_goal"],
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def evaluate(model: VariBAD, n_tasks: int, N: int, seed: int) -> dict:
    model.eval()
    returns, ep_sum, found, first_steps = [], [0.0] * N, 0, []
    rng = Random(seed + 999)
    with torch.no_grad():
        for k in range(n_tasks):
            batch = collect_varibad_rollout(
                make_env(), model, N=N, rng=rng,
                task_rng=Random(seed * 50021 + k), deterministic=True,
            )
            returns.append(batch.total_return)
            for i, er in enumerate(batch.episode_returns()):
                ep_sum[i] += er
            fgs = batch.first_goal_step()
            if fgs is not None:
                found += 1
                first_steps.append(fgs)
    model.train()
    return {
        "avg_return": sum(returns) / n_tasks,
        "episode_returns": [s / n_tasks for s in ep_sum],
        "success_rate": found / n_tasks,
        "avg_first_goal_step": (sum(first_steps) / len(first_steps)) if first_steps else float("nan"),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Simplified VariBAD (spec sections 8-9).")
    p.add_argument("--updates", type=int, default=1500)
    p.add_argument("--batch-tasks", type=int, default=8)
    p.add_argument("-N", "--episodes", type=int, default=ENV_CONFIG["num_episodes_per_task"], dest="N")
    p.add_argument("--lr", type=float, default=TRAINING_CONFIG["lr_policy"])
    p.add_argument("--gamma", type=float, default=TRAINING_CONFIG["gamma"])
    p.add_argument("--beta", type=float, default=0.1, help="KL weight in the ELBO")
    p.add_argument("--vae-coef", type=float, default=1.0, help="weight of the VAE loss vs A2C")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-tasks", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--save", type=str, default="varibad_policy.pt")
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    model = VariBAD().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_rng = Random(args.seed + 1)

    H_plus = args.N * ENV_CONFIG["episode_horizon"]
    print(f"### Training VariBAD  device={device}  updates={args.updates}  "
          f"batch_tasks={args.batch_tasks}  N={args.N}  H+={H_plus}  beta={args.beta}", flush=True)
    print(f"{'update':>7}{'a2c':>9}{'recon':>9}{'kl':>8}{'avg_return':>12}{'success%':>10}  ep_returns", flush=True)

    task_counter = 0
    for update in range(1, args.updates + 1):
        optimizer.zero_grad()
        a2c_acc = recon_acc = kl_acc = 0.0
        for _ in range(args.batch_tasks):
            batch = collect_varibad_rollout(
                make_env(), model, N=args.N, rng=train_rng,
                task_rng=Random(args.seed * 100003 + task_counter),
            )
            task_counter += 1
            al = a2c_losses(batch, gamma=args.gamma,
                            value_coef=TRAINING_CONFIG["value_coef"],
                            entropy_coef=TRAINING_CONFIG["entropy_coef"])
            vl = vae_losses(model, batch, beta=args.beta)
            loss = (al.total + args.vae_coef * vl.total) / args.batch_tasks
            loss.backward()
            a2c_acc += float(al.total.detach()) / args.batch_tasks
            recon_acc += float(vl.recon.detach()) / args.batch_tasks
            kl_acc += float(vl.kl.detach()) / args.batch_tasks
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["max_grad_norm"])
        optimizer.step()

        if update % args.eval_every == 0 or update == 1:
            stats = evaluate(model, args.eval_tasks, args.N, args.seed)
            ep_str = " ".join(f"{r:+.1f}" for r in stats["episode_returns"])
            print(
                f"{update:>7}{a2c_acc:>9.3f}{recon_acc:>9.3f}{kl_acc:>8.2f}"
                f"{stats['avg_return']:>12.2f}{stats['success_rate'] * 100:>9.0f}%   [{ep_str}]",
                flush=True,
            )

    torch.save(model.state_dict(), args.save)
    print(f"\nsaved model -> {args.save}")
    final = evaluate(model, args.eval_tasks, args.N, args.seed)
    print(f"final avg_return={final['avg_return']:.2f}  success={final['success_rate']*100:.0f}%  "
          f"first_goal_step={final['avg_first_goal_step']:.1f}")
    print("per-episode returns:", [f"{r:+.2f}" for r in final["episode_returns"]])


if __name__ == "__main__":
    main()
