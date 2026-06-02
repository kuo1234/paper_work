"""Run 004: PEARL-style encoder with offline SAC-style actor-critic.

This upgrades Run 003 from supervised actor imitation to a small actor-critic
training loop:

* replay samples contain `(context, state, action, reward, next_state, done)`;
* a context encoder produces latent task variable z;
* twin Q-functions learn TD targets with target networks;
* a tanh-squashed Gaussian actor is optimized against Q and entropy;
* a small behavior-cloning regularizer keeps offline actor updates conservative.

It is still not a full PEARL reproduction: the replay buffer is generated from a
scripted behavior policy instead of being collected online by SAC.  The update
structure is intentionally close enough that the next step can replace dataset
generation with online collection and add the PEARL KL term.
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

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required. Use .venv-py312 with requirements-cu126.txt.") from exc

from goal_navigation import Transition, Trajectory, Vector, clip_vector, format_curve
from latent_navigation import LatentNavigationTask, LatentWindNavigationEnv, build_latent_task_splits
from torch_pearl_experiment import (
    ACTION_DIM,
    CONTEXT_DIM,
    HORIZON,
    K_VALUES,
    LATENT_DIM,
    MAX_STEP,
    STATE_DIM,
    ContextEncoder,
    MethodResult,
    NeuralPEARLPolicy,
    SummaryRow,
    collect_probe_context,
    context_features,
    latent_tensor,
    oracle_action,
    probe_action,
)


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian actor used by SAC-style updates."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(STATE_DIM + LATENT_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, ACTION_DIM)
        self.log_std = nn.Linear(hidden_size, ACTION_DIM)

    def forward(self, state: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(torch.cat([state, latent], dim=-1))
        mean = self.mean(hidden)
        log_std = torch.clamp(self.log_std(hidden), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(state, latent)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw = normal.rsample()
        squashed = torch.tanh(raw)
        action = squashed * MAX_STEP

        # Tanh correction follows the standard SAC squashing transform.
        log_prob = normal.log_prob(raw) - torch.log(MAX_STEP * (1.0 - squashed.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def deterministic(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        mean, _ = self(state, latent)
        return torch.tanh(mean) * MAX_STEP


class TwinQ(nn.Module):
    """Twin critic to reduce over-estimation bias."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        input_dim = STATE_DIM + ACTION_DIM + LATENT_DIM
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action, latent], dim=-1)
        return self.q1(x), self.q2(x)


class SACPEARLPolicy(NeuralPEARLPolicy):
    """Evaluation policy that uses deterministic actor mean for query control."""

    name = "PEARL-SAC-style"

    def __init__(self, encoder: ContextEncoder, actor: GaussianActor, prior_latent: torch.Tensor, device: torch.device):
        super().__init__(encoder, actor, prior_latent, device)
        self.actor: GaussianActor = actor

    def act(self, state: Vector) -> Vector:
        state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
        latent_tensor = self.latent.view(1, LATENT_DIM)
        with torch.no_grad():
            action = self.actor.deterministic(state_tensor, latent_tensor).squeeze(0).detach().cpu().tolist()
        return (float(action[0]), float(action[1]))


def behavior_action(step: int, state: Vector, task: LatentNavigationTask, rng: random.Random) -> Vector:
    """Mixture behavior policy for offline replay.

    The replay buffer needs both exploration and useful near-goal control.  A
    pure random dataset makes Q learning possible but inefficient; a pure oracle
    dataset makes the actor brittle away from expert states.
    """

    mode = rng.random()
    if mode < 0.35:
        action = probe_action(step)
    elif mode < 0.75:
        base = oracle_action(state, task)
        action = (base[0] + rng.gauss(0.0, 0.03), base[1] + rng.gauss(0.0, 0.03))
    else:
        angle = rng.uniform(0.0, 2.0 * math.pi)
        action = (math.cos(angle), math.sin(angle))
    return clip_vector(action, MAX_STEP)


def build_replay_dataset(
    tasks: Sequence[LatentNavigationTask],
    rng: random.Random,
    episodes_per_task: int,
) -> TensorDataset:
    """Generate fixed offline replay samples from training tasks."""

    contexts: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    next_states: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    latents: list[torch.Tensor] = []
    oracle_actions: list[torch.Tensor] = []

    for task in tasks:
        latent = latent_tensor(task)
        for _ in range(episodes_per_task):
            context = collect_probe_context(task, rng.choice([1, 2]))
            features = context_features(context)
            env = LatentWindNavigationEnv(task=task, horizon=HORIZON)
            state = env.reset()
            for step in range(HORIZON):
                action = behavior_action(step, state, task, rng)
                transition = env.step(action)

                contexts.append(features)
                states.append(torch.tensor(transition.state, dtype=torch.float32))
                actions.append(torch.tensor(transition.action, dtype=torch.float32))
                rewards.append(torch.tensor([transition.reward], dtype=torch.float32))
                next_states.append(torch.tensor(transition.next_state, dtype=torch.float32))
                dones.append(torch.tensor([1.0 if transition.done else 0.0], dtype=torch.float32))
                latents.append(latent)
                oracle_actions.append(torch.tensor(oracle_action(transition.state, task), dtype=torch.float32))

                state = transition.next_state
                if transition.done:
                    break

    return TensorDataset(
        torch.stack(contexts),
        torch.stack(states),
        torch.stack(actions),
        torch.stack(rewards),
        torch.stack(next_states),
        torch.stack(dones),
        torch.stack(latents),
        torch.stack(oracle_actions),
    )


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def train_sac_pearl(
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    gamma: float,
    alpha: float,
    bc_weight: float,
) -> tuple[ContextEncoder, GaussianActor, torch.Tensor, dict[str, float]]:
    """Train encoder, stochastic actor, and twin Q on offline replay."""

    torch.manual_seed(seed)
    rng = random.Random(seed)
    splits = build_latent_task_splits(seed)
    train_data = build_replay_dataset(splits["train"], rng, episodes_per_task=4)
    val_data = build_replay_dataset(splits["validation"], rng, episodes_per_task=2)

    encoder = ContextEncoder().to(device)
    actor = GaussianActor().to(device)
    critic = TwinQ().to(device)
    target_critic = TwinQ().to(device)
    target_critic.load_state_dict(critic.state_dict())

    critic_optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(critic.parameters()), lr=3e-4)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=3e-4)

    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    last_metrics = {"critic_loss": 0.0, "actor_loss": 0.0, "latent_loss": 0.0, "bc_loss": 0.0}

    for _ in range(epochs):
        for batch in loader:
            context, state, action, reward, next_state, done, target_latent, expert_action = [
                item.to(device) for item in batch
            ]

            latent = encoder(context)
            with torch.no_grad():
                next_action, next_log_prob = actor.sample(next_state, latent)
                target_q1, target_q2 = target_critic(next_state, next_action, latent)
                target_q = torch.minimum(target_q1, target_q2) - alpha * next_log_prob
                td_target = reward + gamma * (1.0 - done) * target_q

            q1, q2 = critic(state, action, latent)
            critic_loss = nn.functional.mse_loss(q1, td_target) + nn.functional.mse_loss(q2, td_target)
            latent_loss = nn.functional.mse_loss(latent, target_latent)
            critic_total = critic_loss + 0.2 * latent_loss

            critic_optimizer.zero_grad()
            critic_total.backward()
            critic_optimizer.step()

            actor_latent = encoder(context).detach()
            sampled_action, log_prob = actor.sample(state, actor_latent)
            sampled_q1, sampled_q2 = critic(state, sampled_action, actor_latent)
            actor_q = torch.minimum(sampled_q1, sampled_q2)
            bc_loss = nn.functional.mse_loss(actor.deterministic(state, actor_latent), expert_action)
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

    val_metrics = evaluate_losses(encoder, actor, critic, val_data, device)
    val_metrics.update({f"last_{key}": value for key, value in last_metrics.items()})
    prior = torch.stack([latent_tensor(task) for task in splits["train"]]).mean(dim=0)
    return encoder, actor, prior, val_metrics


def evaluate_losses(
    encoder: ContextEncoder,
    actor: GaussianActor,
    critic: TwinQ,
    dataset: TensorDataset,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=512)
    latent_losses: list[float] = []
    bc_losses: list[float] = []
    q_means: list[float] = []
    encoder.eval()
    actor.eval()
    critic.eval()
    with torch.no_grad():
        for context, state, _action, _reward, _next_state, _done, target_latent, expert_action in loader:
            context = context.to(device)
            state = state.to(device)
            target_latent = target_latent.to(device)
            expert_action = expert_action.to(device)
            latent = encoder(context)
            action = actor.deterministic(state, latent)
            q1, q2 = critic(state, action, latent)
            latent_losses.append(float(nn.functional.mse_loss(latent, target_latent).cpu()))
            bc_losses.append(float(nn.functional.mse_loss(action, expert_action).cpu()))
            q_means.append(float(torch.minimum(q1, q2).mean().cpu()))
    encoder.train()
    actor.train()
    critic.train()
    return {
        "validation_latent_mse": mean(latent_losses),
        "validation_bc_mse": mean(bc_losses),
        "validation_q_mean": mean(q_means),
    }


def evaluate_sac_policy(
    encoder: ContextEncoder,
    actor: GaussianActor,
    prior: torch.Tensor,
    seed: int,
    device: torch.device,
) -> MethodResult:
    splits = build_latent_task_splits(seed)
    encoder.eval()
    actor.eval()

    def factory() -> SACPEARLPolicy:
        return SACPEARLPolicy(encoder, actor, prior, device)

    curve: dict[int, float] = {}
    for k in K_VALUES:
        returns: list[float] = []
        for task in splits["test"]:
            policy = factory()
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
    return MethodResult(seed=seed, method="PEARL-SAC-style", curve=curve)


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
        "description": "Run 004 PEARL-style offline SAC actor-critic experiment",
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
    parser = argparse.ArgumentParser(description="Run PEARL-style offline SAC experiment.")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--bc-weight", type=float, default=200.0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", default="experiments/torch_pearl_sac")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    results: list[MethodResult] = []
    metrics: dict[int, dict[str, float]] = {}
    for seed in parse_seeds(args.seeds):
        encoder, actor, prior, seed_metrics = train_sac_pearl(
            seed=seed,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            alpha=args.alpha,
            bc_weight=args.bc_weight,
        )
        result = evaluate_sac_policy(encoder, actor, prior, seed, device)
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
