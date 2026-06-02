"""Run 006: probabilistic PEARL-style encoder with KL regularization.

This upgrades Run 005's deterministic context encoder to a PEARL-like
posterior:

* encoder outputs `mu, log_var`;
* z is sampled with the reparameterization trick;
* critic/actor updates use sampled z;
* encoder training includes KL to a standard normal prior;
* online collection samples posterior z from each task's replay context.

The implementation is still compact and toy-task specific, but it now contains
the PEARL-specific posterior + KL mechanics that were missing in Run 005.
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

from goal_navigation import Trajectory, Vector, clip_vector, format_curve
from latent_navigation import LatentNavigationTask, LatentWindNavigationEnv, build_latent_task_splits
from torch_pearl_experiment import (
    CONTEXT_DIM,
    HORIZON,
    K_VALUES,
    LATENT_DIM,
    MAX_STEP,
    MethodResult,
    SummaryRow,
    collect_probe_context,
    context_features,
    latent_tensor,
    oracle_action,
)
from torch_pearl_online_sac_experiment import (
    TaskReplayBuffer,
    bootstrap_buffer,
    summarize,
)
from torch_pearl_sac_experiment import GaussianActor, TwinQ, soft_update


class ProbabilisticContextEncoder(nn.Module):
    """Encode context as a diagonal Gaussian posterior over latent z."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(CONTEXT_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_size, LATENT_DIM)
        self.log_var = nn.Linear(hidden_size, LATENT_DIM)

    def forward(self, context_features_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(context_features_batch)
        log_var = torch.clamp(self.log_var(hidden), -6.0, 2.0)
        return self.mu(hidden), log_var

    def sample(self, context_features_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self(context_features_batch)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log_var

    def mean(self, context_features_batch: torch.Tensor) -> torch.Tensor:
        mu, _ = self(context_features_batch)
        return mu


class ProbabilisticPEARLPolicy:
    """Evaluation wrapper that samples posterior z after observing support."""

    name = "PEARL-probabilistic"

    def __init__(
        self,
        encoder: ProbabilisticContextEncoder,
        actor: GaussianActor,
        prior_latent: torch.Tensor,
        device: torch.device,
        sample_posterior: bool,
        posterior_samples: int,
    ):
        self.encoder = encoder
        self.actor = actor
        self.prior_latent = prior_latent.to(device)
        self.device = device
        self.sample_posterior = sample_posterior
        self.posterior_samples = posterior_samples
        self.context = []
        self.latent = self.prior_latent.clone()

    def reset_for_latent_task(self, task: LatentNavigationTask) -> None:
        del task
        self.context = []
        self.latent = self.prior_latent.clone()

    def observe_trajectory(self, trajectory: Trajectory) -> None:
        self.context.extend(trajectory.transitions)
        features = context_features(self.context).to(self.device).view(1, CONTEXT_DIM)
        with torch.no_grad():
            if self.sample_posterior:
                samples = [self.encoder.sample(features)[0] for _ in range(self.posterior_samples)]
                z = torch.stack(samples).mean(dim=0)
            else:
                z = self.encoder.mean(features)
            self.latent = z.squeeze(0)

    def act(self, state: Vector) -> Vector:
        state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
        latent_tensor = self.latent.view(1, LATENT_DIM)
        with torch.no_grad():
            action = self.actor.deterministic(state_tensor, latent_tensor).squeeze(0).cpu().tolist()
        return (float(action[0]), float(action[1]))


def kl_standard_normal(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL(q(z|c) || N(0, I)) for diagonal Gaussian q."""

    return -0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()


def collect_actor_episode(
    buffer: TaskReplayBuffer,
    encoder: ProbabilisticContextEncoder,
    actor: GaussianActor,
    device: torch.device,
    rng: random.Random,
    exploration_std: float,
) -> float:
    features = context_features(buffer.context(rng)).to(device).view(1, CONTEXT_DIM)
    with torch.no_grad():
        z, _, _ = encoder.sample(features)

    env = LatentWindNavigationEnv(task=buffer.task, horizon=HORIZON)
    state = env.reset()
    total_return = 0.0
    collected = []
    for _ in range(HORIZON):
        state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
        with torch.no_grad():
            action_tensor = actor.deterministic(state_tensor, z).squeeze(0).cpu()
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
    context_items: int,
) -> tuple[torch.Tensor, ...]:
    context_rows = []
    state_rows = []
    action_rows = []
    reward_rows = []
    next_state_rows = []
    done_rows = []
    latent_rows = []
    expert_action_rows = []

    per_task = max(1, batch_size // len(buffers))
    while len(state_rows) < batch_size:
        buffer = buffers[rng.randrange(len(buffers))]
        # PEARL samples context and RL batches separately from the same task.
        # The context subset is intentionally independent from the transition
        # sample used for the TD update.
        context = context_features(buffer.context(rng, max_items=context_items))
        for transition in buffer.sample(rng, per_task):
            context_rows.append(context)
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


def train_probabilistic_pearl(
    seed: int,
    device: torch.device,
    iterations: int,
    updates_per_iteration: int,
    batch_size: int,
    gamma: float,
    alpha: float,
    bc_weight: float,
    kl_weight: float,
    latent_weight: float,
    kl_anneal_steps: int,
    z_action_weight: float,
    context_items: int,
) -> tuple[ProbabilisticContextEncoder, GaussianActor, torch.Tensor, dict[str, float]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    splits = build_latent_task_splits(seed)

    buffers = [TaskReplayBuffer(task) for task in splits["train"]]
    for buffer in buffers:
        bootstrap_buffer(buffer, rng, episodes=2)

    encoder = ProbabilisticContextEncoder().to(device)
    actor = GaussianActor().to(device)
    critic = TwinQ().to(device)
    target_critic = TwinQ().to(device)
    target_critic.load_state_dict(critic.state_dict())

    critic_optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(critic.parameters()), lr=3e-4)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=3e-4)

    last_metrics = {
        "critic_loss": 0.0,
        "actor_loss": 0.0,
        "kl_loss": 0.0,
        "latent_loss": 0.0,
        "bc_loss": 0.0,
    }
    collection_returns: list[float] = []

    global_update = 0
    total_updates = max(1, iterations * updates_per_iteration)
    anneal_denominator = kl_anneal_steps if kl_anneal_steps > 0 else total_updates

    for iteration in range(iterations):
        for _ in range(updates_per_iteration):
            global_update += 1
            context, state, action, reward, next_state, done, target_latent, expert_action = sample_training_batch(
                buffers, rng, batch_size, device, context_items
            )

            z, mu, log_var = encoder.sample(context)
            with torch.no_grad():
                next_action, next_log_prob = actor.sample(next_state, z)
                target_q1, target_q2 = target_critic(next_state, next_action, z)
                target_q = torch.minimum(target_q1, target_q2) - alpha * next_log_prob
                td_target = reward + gamma * (1.0 - done) * target_q

            q1, q2 = critic(state, action, z)
            critic_loss = torch.nn.functional.mse_loss(q1, td_target) + torch.nn.functional.mse_loss(q2, td_target)
            kl_loss = kl_standard_normal(mu, log_var)
            latent_loss = torch.nn.functional.mse_loss(mu, target_latent)
            kl_scale = min(1.0, global_update / anneal_denominator)
            critic_total = critic_loss + (kl_weight * kl_scale) * kl_loss + latent_weight * latent_loss

            critic_optimizer.zero_grad()
            critic_total.backward()
            critic_optimizer.step()

            with torch.no_grad():
                actor_z = encoder.mean(context)
            sampled_action, log_prob = actor.sample(state, actor_z)
            sampled_q1, sampled_q2 = critic(state, sampled_action, actor_z)
            actor_q = torch.minimum(sampled_q1, sampled_q2)
            deterministic_action = actor.deterministic(state, actor_z)
            bc_loss = torch.nn.functional.mse_loss(deterministic_action, expert_action)

            # Encourage the actor to actually condition on z.  The synthetic
            # task latent starts with goal coordinates, so shifting goal-z should
            # shift the oracle action by the same direction before clipping.
            shifted_z = actor_z.clone()
            shifted_z[:, :2] = shifted_z[:, :2] + 0.25
            shifted_action = actor.deterministic(state, shifted_z)
            z_delta = shifted_action - deterministic_action
            target_delta = torch.full_like(z_delta, 0.25 * MAX_STEP)
            z_action_loss = torch.nn.functional.mse_loss(z_delta, target_delta)

            actor_loss = (alpha * log_prob - actor_q).mean() + bc_weight * bc_loss + z_action_weight * z_action_loss

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            soft_update(critic, target_critic, tau=0.01)

            last_metrics = {
                "critic_loss": float(critic_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "kl_loss": float(kl_loss.detach().cpu()),
                "latent_loss": float(latent_loss.detach().cpu()),
                "bc_loss": float(bc_loss.detach().cpu()),
                "z_action_loss": float(z_action_loss.detach().cpu()),
            }

        exploration_std = max(0.005, 0.04 * (1.0 - iteration / max(1, iterations)))
        for buffer in buffers:
            collection_returns.append(collect_actor_episode(buffer, encoder, actor, device, rng, exploration_std))

    prior = torch.zeros(LATENT_DIM, dtype=torch.float32)
    metrics = {
        **{f"last_{key}": value for key, value in last_metrics.items()},
        **diagnose_encoder_actor(encoder, actor, splits["validation"], device),
        "mean_recent_collection_return": mean(collection_returns[-len(buffers) :]),
        "mean_all_collection_return": mean(collection_returns),
        "transitions_per_task_mean": mean([len(buffer.transitions) for buffer in buffers]),
    }
    return encoder, actor, prior, metrics


def evaluate_policy(
    encoder: ProbabilisticContextEncoder,
    actor: GaussianActor,
    prior: torch.Tensor,
    seed: int,
    device: torch.device,
    sample_posterior: bool,
    posterior_samples: int,
    rollout_ensemble: int,
) -> MethodResult:
    splits = build_latent_task_splits(seed)
    encoder.eval()
    actor.eval()
    curve: dict[int, float] = {}

    for k in K_VALUES:
        returns = []
        for task in splits["test"]:
            ensemble_returns = []
            for _ in range(rollout_ensemble):
                policy = ProbabilisticPEARLPolicy(encoder, actor, prior, device, sample_posterior, posterior_samples)
                policy.reset_for_latent_task(task)
                for _support_idx in range(k):
                    support = collect_probe_context(task, 1)
                    policy.observe_trajectory(Trajectory(tuple(support), sum(item.reward for item in support)))
                env = LatentWindNavigationEnv(task=task, horizon=HORIZON)
                state = env.reset()
                total_return = 0.0
                for _step in range(HORIZON):
                    transition = env.step(policy.act(state))
                    total_return += transition.reward
                    state = transition.next_state
                    if transition.done:
                        break
                ensemble_returns.append(total_return)
            returns.append(sum(ensemble_returns) / len(ensemble_returns))
        curve[k] = sum(returns) / len(returns)

    return MethodResult(seed=seed, method="PEARL-probabilistic", curve=curve)


def diagnose_encoder_actor(
    encoder: ProbabilisticContextEncoder,
    actor: GaussianActor,
    tasks: Sequence[LatentNavigationTask],
    device: torch.device,
) -> dict[str, float]:
    """Measure whether posterior z is informative and whether actor uses it."""

    encoder.eval()
    actor.eval()
    latent_errors = []
    posterior_vars = []
    sensitivities = []

    with torch.no_grad():
        for task in tasks:
            support = collect_probe_context(task, 1)
            features = context_features(support).to(device).view(1, CONTEXT_DIM)
            mu, log_var = encoder(features)
            target = latent_tensor(task).to(device).view(1, LATENT_DIM)
            latent_errors.append(float(torch.nn.functional.mse_loss(mu, target).cpu()))
            posterior_vars.append(float(log_var.exp().mean().cpu()))

            states = torch.tensor(
                [[0.0, 0.0], [0.5, 0.0], [0.0, -0.5], [-0.5, 0.5]],
                dtype=torch.float32,
                device=device,
            )
            base_z = mu.repeat(states.shape[0], 1)
            shifted_z = base_z.clone()
            shifted_z[:, :2] += 0.25
            base_action = actor.deterministic(states, base_z)
            shifted_action = actor.deterministic(states, shifted_z)
            sensitivities.append(float(torch.norm(shifted_action - base_action, dim=-1).mean().cpu()))

    encoder.train()
    actor.train()
    return {
        "diagnostic_latent_mse": mean(latent_errors),
        "diagnostic_posterior_var": mean(posterior_vars),
        "diagnostic_actor_z_sensitivity": mean(sensitivities),
    }


def write_outputs(results, summary, metrics, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Run 006 probabilistic PEARL-style online SAC experiment",
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
    parser = argparse.ArgumentParser(description="Run probabilistic PEARL-style experiment.")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--updates-per-iteration", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--bc-weight", type=float, default=50.0)
    parser.add_argument("--kl-weight", type=float, default=0.001)
    parser.add_argument("--latent-weight", type=float, default=0.3)
    parser.add_argument("--kl-anneal-steps", type=int, default=300)
    parser.add_argument("--posterior-samples", type=int, default=8)
    parser.add_argument("--z-action-weight", type=float, default=0.0)
    parser.add_argument("--context-items", type=int, default=64)
    parser.add_argument("--rollout-ensemble", type=int, default=3)
    parser.add_argument("--posterior", choices=["sample", "mean"], default="sample")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", default="experiments/torch_pearl_probabilistic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    results = []
    metrics = {}
    for seed in parse_seeds(args.seeds):
        encoder, actor, prior, seed_metrics = train_probabilistic_pearl(
            seed=seed,
            device=device,
            iterations=args.iterations,
            updates_per_iteration=args.updates_per_iteration,
            batch_size=args.batch_size,
            gamma=args.gamma,
            alpha=args.alpha,
            bc_weight=args.bc_weight,
            kl_weight=args.kl_weight,
            latent_weight=args.latent_weight,
            kl_anneal_steps=args.kl_anneal_steps,
            z_action_weight=args.z_action_weight,
            context_items=args.context_items,
        )
        result = evaluate_policy(
            encoder=encoder,
            actor=actor,
            prior=prior,
            seed=seed,
            device=device,
            sample_posterior=args.posterior == "sample",
            posterior_samples=args.posterior_samples,
            rollout_ensemble=args.rollout_ensemble,
        )
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
