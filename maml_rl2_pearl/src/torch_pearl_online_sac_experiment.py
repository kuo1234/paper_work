"""Run 005: online PEARL-SAC-style training loop.

Run 004 used a fixed offline replay dataset, which made actor updates vulnerable
to Q extrapolation.  This runner keeps the same neural building blocks but adds
online collection:

* each train task owns its own replay buffer;
* buffers are bootstrapped with probe episodes;
* after each update round, the current actor collects more data per task;
* context features are computed from each task's own replay buffer;
* SAC-style actor/critic updates train on the growing online replay.

This is still a compact research scaffold, not a full PEARL reproduction.  It
keeps a small latent-supervision auxiliary loss because the toy simulator exposes
the true latent and the current encoder is deterministic rather than a
probabilistic posterior with KL regularization.
"""

from __future__ import annotations

from dataclasses import asdict
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

from goal_navigation import Transition, Trajectory, Vector, clip_vector, format_curve
from latent_navigation import LatentNavigationTask, LatentWindNavigationEnv, build_latent_task_splits
from torch_pearl_experiment import (
    CONTEXT_DIM,
    HORIZON,
    K_VALUES,
    LATENT_DIM,
    MAX_STEP,
    MethodResult,
    NeuralPEARLPolicy,
    SummaryRow,
    collect_probe_context,
    context_features,
    latent_tensor,
    oracle_action,
    probe_action,
)
from torch_pearl_sac_experiment import ContextEncoder, GaussianActor, TwinQ, soft_update


class TaskReplayBuffer:
    """Small per-task replay buffer storing raw transitions.

    Keeping buffers per task mirrors PEARL's training structure and lets context
    be sampled from the same task as the RL batch.
    """

    def __init__(self, task: LatentNavigationTask, capacity: int = 6000):
        self.task = task
        self.capacity = capacity
        self.transitions: list[Transition] = []

    def add_many(self, transitions: Sequence[Transition]) -> None:
        self.transitions.extend(transitions)
        if len(self.transitions) > self.capacity:
            self.transitions = self.transitions[-self.capacity :]

    def sample(self, rng: random.Random, batch_size: int) -> list[Transition]:
        if not self.transitions:
            raise ValueError("Cannot sample an empty replay buffer.")
        return [self.transitions[rng.randrange(len(self.transitions))] for _ in range(batch_size)]

    def context(self, rng: random.Random, max_items: int = 180) -> list[Transition]:
        if len(self.transitions) <= max_items:
            return list(self.transitions)
        return [self.transitions[rng.randrange(len(self.transitions))] for _ in range(max_items)]


class OnlineSACPolicy(NeuralPEARLPolicy):
    """Evaluation wrapper for Gaussian actor using deterministic mean action."""

    name = "PEARL-online-SAC-style"

    def __init__(self, encoder: ContextEncoder, actor: GaussianActor, prior_latent: torch.Tensor, device: torch.device):
        super().__init__(encoder, actor, prior_latent, device)
        self.actor: GaussianActor = actor

    def act(self, state: Vector) -> Vector:
        state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
        latent_tensor = self.latent.view(1, LATENT_DIM)
        with torch.no_grad():
            action = self.actor.deterministic(state_tensor, latent_tensor).squeeze(0).cpu().tolist()
        return (float(action[0]), float(action[1]))


def bootstrap_buffer(buffer: TaskReplayBuffer, rng: random.Random, episodes: int) -> None:
    """Seed a task buffer with probes and noisy oracle transitions."""

    for episode in range(episodes):
        env = LatentWindNavigationEnv(task=buffer.task, horizon=HORIZON)
        state = env.reset()
        collected: list[Transition] = []
        for step in range(HORIZON):
            if episode == 0:
                action = probe_action(step)
            else:
                base = oracle_action(state, buffer.task)
                action = clip_vector((base[0] + rng.gauss(0.0, 0.03), base[1] + rng.gauss(0.0, 0.03)), MAX_STEP)
            transition = env.step(action)
            collected.append(transition)
            state = transition.next_state
            if transition.done:
                break
        buffer.add_many(collected)


def collect_actor_episode(
    buffer: TaskReplayBuffer,
    encoder: ContextEncoder,
    actor: GaussianActor,
    device: torch.device,
    rng: random.Random,
    exploration_std: float,
) -> float:
    """Collect one online episode using current context-conditioned actor."""

    context = context_features(buffer.context(rng)).to(device).view(1, CONTEXT_DIM)
    with torch.no_grad():
        latent = encoder(context)

    env = LatentWindNavigationEnv(task=buffer.task, horizon=HORIZON)
    state = env.reset()
    collected: list[Transition] = []
    total_return = 0.0
    for _ in range(HORIZON):
        state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
        with torch.no_grad():
            action_tensor = actor.deterministic(state_tensor, latent).squeeze(0).cpu()
        action = (
            float(action_tensor[0]) + rng.gauss(0.0, exploration_std),
            float(action_tensor[1]) + rng.gauss(0.0, exploration_std),
        )
        transition = env.step(clip_vector(action, MAX_STEP))
        collected.append(transition)
        total_return += transition.reward
        state = transition.next_state
        if transition.done:
            break
    buffer.add_many(collected)
    return total_return


def sample_training_batch(
    buffers: Sequence[TaskReplayBuffer],
    rng: random.Random,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Sample a PEARL-style batch with task-matched context and RL samples."""

    context_rows: list[torch.Tensor] = []
    state_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    reward_rows: list[torch.Tensor] = []
    next_state_rows: list[torch.Tensor] = []
    done_rows: list[torch.Tensor] = []
    latent_rows: list[torch.Tensor] = []
    expert_action_rows: list[torch.Tensor] = []

    per_task = max(1, batch_size // len(buffers))
    while len(state_rows) < batch_size:
        buffer = buffers[rng.randrange(len(buffers))]
        task_context = context_features(buffer.context(rng))
        for transition in buffer.sample(rng, per_task):
            context_rows.append(task_context)
            state_rows.append(torch.tensor(transition.state, dtype=torch.float32))
            action_rows.append(torch.tensor(transition.action, dtype=torch.float32))
            reward_rows.append(torch.tensor([transition.reward], dtype=torch.float32))
            next_state_rows.append(torch.tensor(transition.next_state, dtype=torch.float32))
            done_rows.append(torch.tensor([1.0 if transition.done else 0.0], dtype=torch.float32))
            latent_rows.append(latent_tensor(buffer.task))
            expert_action_rows.append(torch.tensor(oracle_action(transition.state, buffer.task), dtype=torch.float32))
            if len(state_rows) >= batch_size:
                break

    return tuple(
        torch.stack(rows).to(device)
        for rows in (
            context_rows,
            state_rows,
            action_rows,
            reward_rows,
            next_state_rows,
            done_rows,
            latent_rows,
            expert_action_rows,
        )
    )


def train_online_sac(
    seed: int,
    device: torch.device,
    iterations: int,
    updates_per_iteration: int,
    batch_size: int,
    gamma: float,
    alpha: float,
    bc_weight: float,
) -> tuple[ContextEncoder, GaussianActor, torch.Tensor, dict[str, float]]:
    """Train online PEARL-SAC-style networks on train tasks."""

    torch.manual_seed(seed)
    rng = random.Random(seed)
    splits = build_latent_task_splits(seed)
    buffers = [TaskReplayBuffer(task) for task in splits["train"]]
    for buffer in buffers:
        bootstrap_buffer(buffer, rng, episodes=2)

    encoder = ContextEncoder().to(device)
    actor = GaussianActor().to(device)
    critic = TwinQ().to(device)
    target_critic = TwinQ().to(device)
    target_critic.load_state_dict(critic.state_dict())

    critic_optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(critic.parameters()), lr=3e-4)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=3e-4)

    last_metrics = {"critic_loss": 0.0, "actor_loss": 0.0, "latent_loss": 0.0, "bc_loss": 0.0}
    collection_returns: list[float] = []

    for iteration in range(iterations):
        for _ in range(updates_per_iteration):
            context, state, action, reward, next_state, done, target_latent, expert_action = sample_training_batch(
                buffers, rng, batch_size, device
            )

            latent = encoder(context)
            with torch.no_grad():
                next_action, next_log_prob = actor.sample(next_state, latent)
                target_q1, target_q2 = target_critic(next_state, next_action, latent)
                target_q = torch.minimum(target_q1, target_q2) - alpha * next_log_prob
                td_target = reward + gamma * (1.0 - done) * target_q

            q1, q2 = critic(state, action, latent)
            critic_loss = torch.nn.functional.mse_loss(q1, td_target) + torch.nn.functional.mse_loss(q2, td_target)
            latent_loss = torch.nn.functional.mse_loss(latent, target_latent)
            critic_total = critic_loss + 0.1 * latent_loss

            critic_optimizer.zero_grad()
            critic_total.backward()
            critic_optimizer.step()

            actor_latent = encoder(context).detach()
            sampled_action, log_prob = actor.sample(state, actor_latent)
            sampled_q1, sampled_q2 = critic(state, sampled_action, actor_latent)
            actor_q = torch.minimum(sampled_q1, sampled_q2)
            bc_loss = torch.nn.functional.mse_loss(actor.deterministic(state, actor_latent), expert_action)
            actor_loss = (alpha * log_prob - actor_q).mean() + bc_weight * bc_loss

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            soft_update(critic, target_critic, tau=0.01)

            last_metrics = {
                "critic_loss": float(critic_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "latent_loss": float(latent_loss.detach().cpu()),
                "bc_loss": float(bc_loss.detach().cpu()),
            }

        # Collect online data after updates.  Exploration decays but never
        # reaches zero, so buffers continue to cover nearby actions.
        exploration_std = max(0.005, 0.04 * (1.0 - iteration / max(1, iterations)))
        for buffer in buffers:
            collection_returns.append(collect_actor_episode(buffer, encoder, actor, device, rng, exploration_std))

    prior = torch.stack([latent_tensor(task) for task in splits["train"]]).mean(dim=0)
    metrics = {
        **{f"last_{key}": value for key, value in last_metrics.items()},
        "mean_recent_collection_return": mean(collection_returns[-len(buffers) :]),
        "mean_all_collection_return": mean(collection_returns),
        "transitions_per_task_mean": mean([len(buffer.transitions) for buffer in buffers]),
    }
    return encoder, actor, prior, metrics


def evaluate_policy(
    encoder: ContextEncoder,
    actor: GaussianActor,
    prior: torch.Tensor,
    seed: int,
    device: torch.device,
) -> MethodResult:
    splits = build_latent_task_splits(seed)
    encoder.eval()
    actor.eval()

    curve: dict[int, float] = {}
    for k in K_VALUES:
        returns: list[float] = []
        for task in splits["test"]:
            policy = OnlineSACPolicy(encoder, actor, prior, device)
            policy.reset_for_latent_task(task)
            for _ in range(k):
                support = collect_probe_context(task, 1)
                policy.observe_trajectory(Trajectory(tuple(support), sum(item.reward for item in support)))
            env = LatentWindNavigationEnv(task=task, horizon=HORIZON)
            state = env.reset()
            total_return = 0.0
            for _ in range(HORIZON):
                transition = env.step(policy.act(state))
                total_return += transition.reward
                state = transition.next_state
                if transition.done:
                    break
            returns.append(total_return)
        curve[k] = sum(returns) / len(returns)
    return MethodResult(seed=seed, method="PEARL-online-SAC-style", curve=curve)


def summarize(results: Sequence[MethodResult]) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    for method in sorted({result.method for result in results}):
        method_results = [result for result in results if result.method == method]
        for k in K_VALUES:
            values = [result.curve[k] for result in method_results]
            rows.append(SummaryRow(method, k, mean(values), pstdev(values) if len(values) > 1 else 0.0, len(values)))
    return rows


def write_outputs(
    results: Sequence[MethodResult],
    summary: Sequence[SummaryRow],
    metrics: dict[int, dict[str, float]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Run 005 online PEARL-SAC-style actor-critic experiment",
        "k_values": list(K_VALUES),
        "horizon": HORIZON,
        "metrics": {str(seed): value for seed, value in metrics.items()},
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


def parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run online PEARL-SAC-style experiment.")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--updates-per-iteration", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--bc-weight", type=float, default=50.0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", default="experiments/torch_pearl_online_sac")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    results: list[MethodResult] = []
    metrics: dict[int, dict[str, float]] = {}
    for seed in parse_seeds(args.seeds):
        encoder, actor, prior, seed_metrics = train_online_sac(
            seed=seed,
            device=device,
            iterations=args.iterations,
            updates_per_iteration=args.updates_per_iteration,
            batch_size=args.batch_size,
            gamma=args.gamma,
            alpha=args.alpha,
            bc_weight=args.bc_weight,
        )
        result = evaluate_policy(encoder, actor, prior, seed, device)
        results.append(result)
        metrics[seed] = seed_metrics
        print(f"seed={seed} metrics={seed_metrics} curve=({format_curve(result.curve)})")

    summary = summarize(results)
    for row in summary:
        print(f"{row.method} K={row.k}: {row.mean_return:.3f} ± {row.std_return:.3f}")

    output_dir = Path(args.output_dir)
    write_outputs(results, summary, metrics, output_dir)
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
