"""PPO-style RL2 recurrent baseline for the latent navigation task.

RL2 adapts through recurrent state.  This runner trains a GRU Gaussian policy
with clipped PPO updates over whole task sequences, keeping hidden state across
episodes from the same task and resetting it only between tasks.

This is a stronger recurrent baseline than the lightweight REINFORCE runner,
but it still does not change the faithful PEARL implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import csv
import json
from pathlib import Path
import random
from statistics import mean, pstdev
from typing import Sequence

import torch
from torch import nn

from goal_navigation import Vector, clip_vector, format_curve
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


@dataclass
class TaskSequence:
    observations: torch.Tensor
    raw_actions: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: list[float]
    dones: list[float]
    total_return: float


class RL2Policy(nn.Module):
    """GRU policy/value model for recurrent task adaptation."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(INPUT_DIM, hidden_size)
        self.mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, ACTION_DIM),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -1.0))

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def distribution(self, hidden: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        mean_action = self.mean(hidden)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp().expand_as(mean_action)
        return torch.distributions.Normal(mean_action, std), mean_action

    def step(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.gru(observation, hidden)
        normal, _mean_action = self.distribution(hidden)
        raw_action = normal.sample()
        squashed = torch.tanh(raw_action)
        action = squashed * MAX_STEP
        log_prob = tanh_log_prob(normal, raw_action, squashed)
        value = self.value(hidden).squeeze(-1)
        return action, raw_action, log_prob, value, hidden

    def evaluate_sequence(
        self,
        observations: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.initial_hidden(1, observations.device)
        log_probs = []
        values = []
        entropies = []
        for observation, raw_action in zip(observations, raw_actions):
            hidden = self.gru(observation.unsqueeze(0), hidden)
            normal, _mean_action = self.distribution(hidden)
            squashed = torch.tanh(raw_action.unsqueeze(0))
            log_probs.append(tanh_log_prob(normal, raw_action.unsqueeze(0), squashed).squeeze(0))
            values.append(self.value(hidden).squeeze(0).squeeze(-1))
            entropies.append(normal.entropy().sum(dim=-1).squeeze(0))
        return torch.stack(log_probs), torch.stack(values), torch.stack(entropies)

    def deterministic_step(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.gru(observation, hidden)
        action = torch.tanh(self.mean(hidden)) * MAX_STEP
        return action, hidden


def tanh_log_prob(
    normal: torch.distributions.Normal,
    raw_action: torch.Tensor,
    squashed: torch.Tensor,
) -> torch.Tensor:
    log_prob = normal.log_prob(raw_action) - torch.log(MAX_STEP * (1.0 - squashed.pow(2)) + 1e-6)
    return log_prob.sum(dim=-1)


def make_observation(state: Vector, prev_action: Vector, prev_reward: float, done: float, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [state[0], state[1], prev_action[0], prev_action[1], prev_reward, done],
        dtype=torch.float32,
        device=device,
    )


def discounted_returns(
    rewards: Sequence[float],
    dones: Sequence[float],
    gamma: float,
    reward_scale: float,
    device: torch.device,
) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        running = reward_scale * reward + gamma * running * (1.0 - done)
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32, device=device)


def collect_task_sequence(
    policy: RL2Policy,
    task: LatentNavigationTask,
    device: torch.device,
    episodes_per_task: int,
) -> TaskSequence:
    hidden = policy.initial_hidden(1, device)
    observations = []
    raw_actions = []
    old_log_probs = []
    rewards = []
    dones = []
    total_return = 0.0

    prev_action: Vector = (0.0, 0.0)
    prev_reward = 0.0
    done_flag = 1.0

    for episode_index in range(episodes_per_task):
        env = LatentWindNavigationEnv(task=task, horizon=HORIZON)
        state = env.reset()
        done_flag = 1.0 if episode_index == 0 else 0.0
        for _step in range(HORIZON):
            observation = make_observation(state, prev_action, prev_reward, done_flag, device)
            with torch.no_grad():
                action_tensor, raw_action, log_prob, _value, hidden = policy.step(observation.unsqueeze(0), hidden)
            action = tuple(action_tensor.squeeze(0).cpu().tolist())  # type: ignore[assignment]
            transition = env.step(clip_vector(action, MAX_STEP))

            observations.append(observation.detach())
            raw_actions.append(raw_action.squeeze(0).detach())
            old_log_probs.append(log_prob.squeeze(0).detach())
            rewards.append(transition.reward)
            dones.append(1.0 if transition.done else 0.0)
            total_return += transition.reward

            prev_action = transition.action
            prev_reward = transition.reward
            done_flag = 1.0 if transition.done else 0.0
            state = transition.next_state
            if transition.done:
                break

    return TaskSequence(
        observations=torch.stack(observations),
        raw_actions=torch.stack(raw_actions),
        old_log_probs=torch.stack(old_log_probs),
        rewards=rewards,
        dones=dones,
        total_return=total_return,
    )


def ppo_update(
    policy: RL2Policy,
    optimizer: torch.optim.Optimizer,
    sequences: Sequence[TaskSequence],
    gamma: float,
    reward_scale: float,
    clip_epsilon: float,
    ppo_epochs: int,
    value_coef: float,
    entropy_coef: float,
) -> dict[str, float]:
    last_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

    prepared = []
    for sequence in sequences:
        returns = discounted_returns(sequence.rewards, sequence.dones, gamma, reward_scale, sequence.observations.device)
        with torch.no_grad():
            _log_probs, values, _entropy = policy.evaluate_sequence(sequence.observations, sequence.raw_actions)
            advantages = returns - values
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
        prepared.append((sequence, returns, advantages.detach()))

    for _ in range(ppo_epochs):
        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        for sequence, returns, advantages in prepared:
            new_log_probs, values, entropy = policy.evaluate_sequence(sequence.observations, sequence.raw_actions)
            ratio = torch.exp(new_log_probs - sequence.old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            policy_losses.append(-torch.min(unclipped, clipped).mean())
            value_losses.append(torch.nn.functional.mse_loss(values, returns))
            entropies.append(entropy.mean())
            approx_kls.append((sequence.old_log_probs - new_log_probs).mean().detach())

        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy_bonus = torch.stack(entropies).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()

        last_metrics = {
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy_bonus.detach().cpu()),
            "approx_kl": float(torch.stack(approx_kls).mean().cpu()),
        }

    return last_metrics


def train_rl2_ppo(
    seed: int,
    device: torch.device,
    iterations: int,
    meta_batch: int,
    episodes_per_task: int,
    gamma: float,
    reward_scale: float,
    clip_epsilon: float,
    ppo_epochs: int,
) -> tuple[RL2Policy, dict[str, float], list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train_tasks = build_latent_task_splits(seed)["train"]
    policy = RL2Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    curve: list[dict[str, float]] = []
    last_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "mean_task_return": 0.0}

    for iteration in range(iterations):
        sequences = []
        for _ in range(meta_batch):
            task = train_tasks[rng.randrange(len(train_tasks))]
            sequences.append(collect_task_sequence(policy, task, device, episodes_per_task))

        metrics = ppo_update(
            policy=policy,
            optimizer=optimizer,
            sequences=sequences,
            gamma=gamma,
            reward_scale=reward_scale,
            clip_epsilon=clip_epsilon,
            ppo_epochs=ppo_epochs,
            value_coef=0.5,
            entropy_coef=0.01,
        )
        last_metrics = {**metrics, "mean_task_return": mean(sequence.total_return for sequence in sequences)}
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
            episode_returns = []

            for episode_index in range(max(1, k + 1)):
                env = LatentWindNavigationEnv(task=task, horizon=HORIZON)
                state = env.reset()
                total_return = 0.0
                done_flag = 1.0 if episode_index == 0 else 0.0
                for _ in range(HORIZON):
                    observation = make_observation(state, prev_action, prev_reward, done_flag, device).unsqueeze(0)
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
    return MethodResult(seed=seed, method="RL2-PPO", curve=curve)


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
        "description": "PPO-style RL2 recurrent baseline on latent navigation task",
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
        writer.writerow(["seed", "iteration", "policy_loss", "value_loss", "entropy", "approx_kl", "mean_task_return"])
        for seed, rows in training_curves.items():
            for row in rows:
                writer.writerow(
                    [
                        seed,
                        int(row["iteration"]),
                        f"{row['policy_loss']:.6f}",
                        f"{row['value_loss']:.6f}",
                        f"{row['entropy']:.6f}",
                        f"{row['approx_kl']:.6f}",
                        f"{row['mean_task_return']:.6f}",
                    ]
                )


def parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO-style RL2 recurrent baseline.")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--meta-batch", type=int, default=8)
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--reward-scale", type=float, default=0.05)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", default="experiments/rl2_ppo_baseline")
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
        policy, seed_metrics, seed_curve = train_rl2_ppo(
            seed=seed,
            device=device,
            iterations=args.iterations,
            meta_batch=args.meta_batch,
            episodes_per_task=args.episodes_per_task,
            gamma=args.gamma,
            reward_scale=args.reward_scale,
            clip_epsilon=args.clip_epsilon,
            ppo_epochs=args.ppo_epochs,
        )
        result = evaluate_rl2(policy, seed, device)
        results.append(result)
        metrics[seed] = seed_metrics
        training_curves[seed] = seed_curve
        print(f"seed={seed} metrics={seed_metrics} curve=({format_curve(result.curve)})")

    summary = summarize(results)
    for row in summary:
        print(f"{row.method} K={row.k}: {row.mean_return:.3f} +/- {row.std_return:.3f}")

    output_dir = Path(args.output_dir)
    write_outputs(results, summary, metrics, training_curves, output_dir)
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
