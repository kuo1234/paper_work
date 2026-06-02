"""Faithful RL²-style recurrent baseline for the latent navigation task.

RL² treats adaptation as recurrent state update rather than gradient update or
probabilistic posterior inference.  This runner keeps the baseline simple and
aligned with the meta-RL comparison:

* recurrent policy receives `(state, previous_action, previous_reward, done)`;
* hidden state is reset only between tasks;
* hidden state persists across K trajectories during meta-test;
* policy is trained with on-policy REINFORCE across multiple episodes per task.

This is a lightweight faithful baseline, not an optimized PPO/TRPO RL²
implementation.  It is sufficient to start comparing whether PEARL's
probabilistic context inference shows an advantage over recurrent adaptation on
the same task split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean, pstdev
from typing import Sequence

import torch
from torch import nn

from goal_navigation import Transition, Vector, clip_vector, format_curve
from latent_navigation import LatentNavigationTask, LatentWindNavigationEnv, build_latent_task_splits


K_VALUES = (0, 1, 2, 3, 5)
HORIZON = 60
STATE_DIM = 2
ACTION_DIM = 2
INPUT_DIM = STATE_DIM + ACTION_DIM + 1 + 1
MAX_STEP = 0.08
LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


@dataclass(frozen=True)
class MethodResult:
    seed: int
    method: str
    curve: dict[int, float]


@dataclass(frozen=True)
class SummaryRow:
    method: str
    k: int
    mean_return: float
    std_return: float
    seed_count: int


class RL2Policy(nn.Module):
    """GRU policy for RL² recurrent adaptation."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(INPUT_DIM, hidden_size)
        self.mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ACTION_DIM),
        )
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -1.0))
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def step(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.gru(observation, hidden)
        mean_action = self.mean(hidden)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp().expand_as(mean_action)
        normal = torch.distributions.Normal(mean_action, std)
        raw_action = normal.rsample()
        squashed = torch.tanh(raw_action)
        action = squashed * MAX_STEP
        log_prob = normal.log_prob(raw_action) - torch.log(MAX_STEP * (1.0 - squashed.pow(2)) + 1e-6)
        value = self.value(hidden)
        return action, log_prob.sum(dim=-1, keepdim=True), value, hidden

    def deterministic_step(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.gru(observation, hidden)
        action = torch.tanh(self.mean(hidden)) * MAX_STEP
        return action, hidden


def make_observation(state: Vector, prev_action: Vector, prev_reward: float, done: float, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[state[0], state[1], prev_action[0], prev_action[1], prev_reward, done]],
        dtype=torch.float32,
        device=device,
    )


def collect_training_task(
    policy: RL2Policy,
    task: LatentNavigationTask,
    device: torch.device,
    episodes_per_task: int,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Collect one task sequence and return policy/value losses."""

    hidden = policy.initial_hidden(1, device)
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    rewards: list[float] = []
    total_return = 0.0

    prev_action: Vector = (0.0, 0.0)
    prev_reward = 0.0
    done_flag = 1.0

    for _episode in range(episodes_per_task):
        env = LatentWindNavigationEnv(task=task, horizon=HORIZON)
        state = env.reset()
        done_flag = 1.0
        for _step in range(HORIZON):
            observation = make_observation(state, prev_action, prev_reward, done_flag, device)
            action_tensor, log_prob, value, hidden = policy.step(observation, hidden)
            action = tuple(action_tensor.squeeze(0).detach().cpu().tolist())  # type: ignore[assignment]
            transition = env.step(clip_vector(action, MAX_STEP))

            log_probs.append(log_prob.squeeze(0))
            values.append(value.squeeze(0))
            rewards.append(transition.reward)
            total_return += transition.reward

            prev_action = transition.action
            prev_reward = transition.reward
            done_flag = 1.0 if transition.done else 0.0
            state = transition.next_state
            if transition.done:
                break

    returns = discounted_returns(rewards, gamma, device)
    value_tensor = torch.stack(values).view(-1)
    log_prob_tensor = torch.stack(log_probs).view(-1)
    advantage = returns - value_tensor.detach()
    policy_loss = -(log_prob_tensor * advantage).mean()
    value_loss = torch.nn.functional.mse_loss(value_tensor, returns)
    return policy_loss, value_loss, total_return


def discounted_returns(rewards: Sequence[float], gamma: float, device: torch.device) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    if tensor.numel() > 1:
        tensor = (tensor - tensor.mean()) / (tensor.std(unbiased=False) + 1e-6)
    return tensor


def train_rl2(
    seed: int,
    device: torch.device,
    iterations: int,
    meta_batch: int,
    episodes_per_task: int,
    gamma: float,
) -> tuple[RL2Policy, dict[str, float], list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train_tasks = build_latent_task_splits(seed)["train"]
    policy = RL2Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    curve: list[dict[str, float]] = []
    last_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "mean_task_return": 0.0}

    for iteration in range(iterations):
        losses = []
        value_losses = []
        task_returns = []
        for _ in range(meta_batch):
            task = train_tasks[rng.randrange(len(train_tasks))]
            policy_loss, value_loss, task_return = collect_training_task(policy, task, device, episodes_per_task, gamma)
            losses.append(policy_loss)
            value_losses.append(value_loss)
            task_returns.append(task_return)

        total_loss = torch.stack(losses).mean() + 0.5 * torch.stack(value_losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()

        last_metrics = {
            "policy_loss": float(torch.stack(losses).mean().detach().cpu()),
            "value_loss": float(torch.stack(value_losses).mean().detach().cpu()),
            "mean_task_return": mean(task_returns),
        }
        curve.append({"iteration": float(iteration + 1), **last_metrics})

    return policy, last_metrics, curve


def evaluate_rl2(policy: RL2Policy, seed: int, device: torch.device) -> MethodResult:
    test_tasks = build_latent_task_splits(seed)["test"]
    policy.eval()
    curve: dict[int, float] = {}

    for k in K_VALUES:
        returns = []
        for task in test_tasks:
            hidden = policy.initial_hidden(1, device)
            prev_action: Vector = (0.0, 0.0)
            prev_reward = 0.0
            done_flag = 1.0
            episode_returns = []

            for episode_index in range(max(1, k + 1)):
                env = LatentWindNavigationEnv(task=task, horizon=HORIZON)
                state = env.reset()
                total_return = 0.0
                done_flag = 1.0 if episode_index == 0 else 0.0
                for _ in range(HORIZON):
                    observation = make_observation(state, prev_action, prev_reward, done_flag, device)
                    with torch.no_grad():
                        action_tensor, hidden = policy.deterministic_step(observation, hidden)
                    action = tuple(action_tensor.squeeze(0).cpu().tolist())  # type: ignore[assignment]
                    transition = env.step(clip_vector(action, MAX_STEP))
                    total_return += transition.reward
                    prev_action = transition.action
                    prev_reward = transition.reward
                    done_flag = 1.0 if transition.done else 0.0
                    state = transition.next_state
                    if transition.done:
                        break
                episode_returns.append(total_return)
            returns.append(episode_returns[min(k, len(episode_returns) - 1)])
        curve[k] = sum(returns) / len(returns)

    policy.train()
    return MethodResult(seed=seed, method="RL2-recurrent", curve=curve)


def summarize(results: Sequence[MethodResult]) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    for method in sorted({result.method for result in results}):
        method_results = [result for result in results if result.method == method]
        for k in K_VALUES:
            values = [result.curve[k] for result in method_results]
            rows.append(SummaryRow(method, k, mean(values), pstdev(values) if len(values) > 1 else 0.0, len(values)))
    return rows


def write_outputs(results, summary, metrics, training_curves, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "RL2 recurrent baseline on latent navigation task",
        "k_values": list(K_VALUES),
        "metrics": {str(seed): value for seed, value in metrics.items()},
        "training_curves": {str(seed): value for seed, value in training_curves.items()},
        "results": [
            {"seed": result.seed, "method": result.method, "curve": {str(k): value for k, value in result.curve.items()}}
            for result in results
        ],
        "summary": [asdict(row) for row in summary],
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "k", "mean_return", "std_return", "seed_count"])
        for row in summary:
            writer.writerow([row.method, row.k, f"{row.mean_return:.6f}", f"{row.std_return:.6f}", row.seed_count])
    with (output_dir / "training_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "iteration", "policy_loss", "value_loss", "mean_task_return"])
        for seed, rows in training_curves.items():
            for row in rows:
                writer.writerow(
                    [
                        seed,
                        int(row["iteration"]),
                        f"{row['policy_loss']:.6f}",
                        f"{row['value_loss']:.6f}",
                        f"{row['mean_task_return']:.6f}",
                    ]
                )


def parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL2 recurrent baseline.")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--meta-batch", type=int, default=8)
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", default="experiments/rl2_baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    results = []
    metrics = {}
    training_curves = {}
    for seed in parse_seeds(args.seeds):
        policy, seed_metrics, seed_curve = train_rl2(
            seed=seed,
            device=device,
            iterations=args.iterations,
            meta_batch=args.meta_batch,
            episodes_per_task=args.episodes_per_task,
            gamma=args.gamma,
        )
        result = evaluate_rl2(policy, seed, device)
        results.append(result)
        metrics[seed] = seed_metrics
        training_curves[seed] = seed_curve
        print(f"seed={seed} metrics={seed_metrics} curve=({format_curve(result.curve)})")

    summary = summarize(results)
    for row in summary:
        print(f"{row.method} K={row.k}: {row.mean_return:.3f} ± {row.std_return:.3f}")

    output_dir = Path(args.output_dir)
    write_outputs(results, summary, metrics, training_curves, output_dir)
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
