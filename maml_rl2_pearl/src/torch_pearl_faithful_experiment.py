"""Faithful PEARL reproduction scaffold.

This runner intentionally removes the non-paper stabilizers used in earlier
research variants:

* no true latent supervision;
* no oracle behavior cloning;
* no scripted probe policy for context collection;
* no actor-z sensitivity regularizer;
* no analytic candidate posterior.

It keeps the core PEARL structure from Rakelly et al. (ICML 2019):

* per-task replay buffers;
* probabilistic context encoder q(z|c);
* product-of-Gaussians aggregation over transition factors;
* policy, Q-functions, and value function conditioned on z;
* context batch and RL batch sampled separately from the same task;
* SAC-style off-policy updates with KL regularization on q(z|c);
* meta-test adaptation by collecting context and re-inferring z.

Sources:
* Paper: https://proceedings.mlr.press/v97/rakelly19a.html
* Reference implementation: https://github.com/katerakelly/oyster
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

from goal_navigation import Transition, Trajectory, Vector, clip_vector, format_curve
from latent_navigation import LatentNavigationTask, LatentWindNavigationEnv, build_latent_task_splits


K_VALUES = (0, 1, 2, 3, 5)
HORIZON = 60
STATE_DIM = 2
ACTION_DIM = 2
TRANSITION_DIM = STATE_DIM + ACTION_DIM + 1 + STATE_DIM
LATENT_DIM = 4
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


@dataclass(frozen=True)
class PosteriorDiagnosticRow:
    seed: int
    k: int
    mean_return: float
    posterior_mean_norm: float
    posterior_var_mean: float
    posterior_kl: float


class TaskReplayBuffer:
    """Per-task replay buffer used for both context and RL batches."""

    def __init__(self, task: LatentNavigationTask, capacity: int = 10000):
        self.task = task
        self.capacity = capacity
        self.transitions: list[Transition] = []

    def add(self, transitions: Sequence[Transition]) -> None:
        self.transitions.extend(transitions)
        if len(self.transitions) > self.capacity:
            self.transitions = self.transitions[-self.capacity :]

    def sample(self, rng: random.Random, batch_size: int) -> list[Transition]:
        if not self.transitions:
            raise ValueError("Cannot sample from an empty replay buffer.")
        return [self.transitions[rng.randrange(len(self.transitions))] for _ in range(batch_size)]


class TransitionContextEncoder(nn.Module):
    """Encode each transition into one Gaussian factor over z."""

    def __init__(self, hidden_size: int = 200):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(TRANSITION_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, LATENT_DIM)
        self.log_var = nn.Linear(hidden_size, LATENT_DIM)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior parameters from a context set.

        Args:
            context: shape `[batch, context_size, transition_dim]`.
        """

        hidden = self.body(context)
        means = self.mean(hidden)
        variances = torch.nn.functional.softplus(self.log_var(hidden)) + 1e-6
        return product_of_gaussians(means, variances)


def product_of_gaussians(means: torch.Tensor, variances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine diagonal Gaussian factors with product-of-Gaussians."""

    precision = 1.0 / torch.clamp(variances, min=1e-6)
    posterior_variance = 1.0 / precision.sum(dim=1)
    posterior_mean = posterior_variance * (means * precision).sum(dim=1)
    return posterior_mean, posterior_variance


def sample_z(mean_tensor: torch.Tensor, variance_tensor: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(torch.clamp(variance_tensor, min=1e-6))
    return mean_tensor + torch.randn_like(std) * std


def kl_to_standard_normal(mean_tensor: torch.Tensor, variance_tensor: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(mean_tensor.pow(2) + variance_tensor - torch.log(variance_tensor + 1e-8) - 1.0, dim=-1).mean()


class TanhGaussianPolicy(nn.Module):
    """SAC policy conditioned on state and latent z."""

    def __init__(self, hidden_size: int = 200):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(STATE_DIM + LATENT_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, ACTION_DIM)
        self.log_std = nn.Linear(hidden_size, ACTION_DIM)

    def forward(self, state: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(torch.cat([state, z], dim=-1))
        mean_tensor = self.mean(hidden)
        log_std = torch.clamp(self.log_std(hidden), LOG_STD_MIN, LOG_STD_MAX)
        return mean_tensor, log_std

    def sample(self, state: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_tensor, log_std = self(state, z)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean_tensor, std)
        raw = normal.rsample()
        squashed = torch.tanh(raw)
        action = squashed * MAX_STEP
        log_prob = normal.log_prob(raw) - torch.log(MAX_STEP * (1.0 - squashed.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def deterministic(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        mean_tensor, _ = self(state, z)
        return torch.tanh(mean_tensor) * MAX_STEP


class QFunction(nn.Module):
    def __init__(self, hidden_size: int = 200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM + LATENT_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action, z], dim=-1))


class ValueFunction(nn.Module):
    def __init__(self, hidden_size: int = 200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + LATENT_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, z], dim=-1))


def transition_tensor(transitions: Sequence[Transition]) -> torch.Tensor:
    rows = [
        [
            item.state[0],
            item.state[1],
            item.action[0],
            item.action[1],
            item.reward,
            item.next_state[0],
            item.next_state[1],
        ]
        for item in transitions
    ]
    return torch.tensor(rows, dtype=torch.float32)


def rl_batch_tensor(transitions: Sequence[Transition]) -> tuple[torch.Tensor, ...]:
    states = torch.tensor([item.state for item in transitions], dtype=torch.float32)
    actions = torch.tensor([item.action for item in transitions], dtype=torch.float32)
    rewards = torch.tensor([[item.reward] for item in transitions], dtype=torch.float32)
    next_states = torch.tensor([item.next_state for item in transitions], dtype=torch.float32)
    dones = torch.tensor([[1.0 if item.done else 0.0] for item in transitions], dtype=torch.float32)
    return states, actions, rewards, next_states, dones


def random_action(rng: random.Random) -> Vector:
    angle = rng.uniform(0.0, 2.0 * math.pi)
    return (math.cos(angle), math.sin(angle))


def collect_episode(
    task: LatentNavigationTask,
    policy: TanhGaussianPolicy | None,
    encoder: TransitionContextEncoder | None,
    context: Sequence[Transition],
    device: torch.device,
    rng: random.Random,
    use_posterior: bool,
) -> list[Transition]:
    """Collect one trajectory with random warmup or PEARL policy."""

    env = LatentWindNavigationEnv(task=task, horizon=HORIZON)
    state = env.reset()
    collected: list[Transition] = []

    if policy is not None and encoder is not None and context:
        context_batch = transition_tensor(context).unsqueeze(0).to(device)
        with torch.no_grad():
            posterior_mean, posterior_var = encoder(context_batch)
            z = sample_z(posterior_mean, posterior_var) if use_posterior else torch.zeros(1, LATENT_DIM, device=device)
    else:
        z = torch.zeros(1, LATENT_DIM, device=device)

    for _ in range(HORIZON):
        if policy is None:
            action = random_action(rng)
        else:
            state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor, _ = policy.sample(state_tensor, z)
            action = tuple(action_tensor.squeeze(0).cpu().tolist())  # type: ignore[assignment]

        transition = env.step(clip_vector(action, MAX_STEP))
        collected.append(transition)
        state = transition.next_state
        if transition.done:
            break

    return collected


def initialize_buffers(
    tasks: Sequence[LatentNavigationTask],
    rng: random.Random,
    warmup_episodes: int,
) -> list[TaskReplayBuffer]:
    buffers = [TaskReplayBuffer(task) for task in tasks]
    for buffer in buffers:
        for _ in range(warmup_episodes):
            buffer.add(collect_episode(buffer.task, None, None, [], torch.device("cpu"), rng, use_posterior=False))
    return buffers


def sample_meta_batch(
    buffers: Sequence[TaskReplayBuffer],
    rng: random.Random,
    meta_batch: int,
    context_size: int,
    rl_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = [buffers[rng.randrange(len(buffers))] for _ in range(meta_batch)]
    context_rows = []
    state_rows = []
    action_rows = []
    reward_rows = []
    next_state_rows = []
    done_rows = []

    for buffer in selected:
        context_rows.append(transition_tensor(buffer.sample(rng, context_size)))
        states, actions, rewards, next_states, dones = rl_batch_tensor(buffer.sample(rng, rl_batch_size))
        state_rows.append(states)
        action_rows.append(actions)
        reward_rows.append(rewards)
        next_state_rows.append(next_states)
        done_rows.append(dones)

    return (
        torch.stack(context_rows).to(device),
        torch.stack(state_rows).to(device),
        torch.stack(action_rows).to(device),
        torch.stack(reward_rows).to(device),
        torch.stack(next_state_rows).to(device),
        torch.stack(done_rows).to(device),
    )


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def train_faithful_pearl(
    seed: int,
    device: torch.device,
    iterations: int,
    updates_per_iteration: int,
    meta_batch: int,
    context_size: int,
    rl_batch_size: int,
    warmup_episodes: int,
    collection_interval: int,
    gamma: float,
    alpha: float,
    kl_lambda: float,
) -> tuple[TransitionContextEncoder, TanhGaussianPolicy, dict[str, float], list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    splits = build_latent_task_splits(seed)
    buffers = initialize_buffers(splits["train"], rng, warmup_episodes)

    encoder = TransitionContextEncoder().to(device)
    policy = TanhGaussianPolicy().to(device)
    qf1 = QFunction().to(device)
    qf2 = QFunction().to(device)
    vf = ValueFunction().to(device)
    target_vf = ValueFunction().to(device)
    target_vf.load_state_dict(vf.state_dict())

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=3e-4)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=3e-4)
    v_optimizer = torch.optim.Adam(vf.parameters(), lr=3e-4)

    last_metrics = {"q_loss": 0.0, "v_loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0}
    training_curve: list[dict[str, float]] = []

    for iteration in range(iterations):
        for _ in range(updates_per_iteration):
            context, states, actions, rewards, next_states, dones = sample_meta_batch(
                buffers, rng, meta_batch, context_size, rl_batch_size, device
            )
            posterior_mean, posterior_var = encoder(context)
            z = sample_z(posterior_mean, posterior_var)
            z_expanded = z.unsqueeze(1).expand(-1, rl_batch_size, -1).reshape(meta_batch * rl_batch_size, LATENT_DIM)

            flat_states = states.reshape(meta_batch * rl_batch_size, STATE_DIM)
            flat_actions = actions.reshape(meta_batch * rl_batch_size, ACTION_DIM)
            flat_rewards = rewards.reshape(meta_batch * rl_batch_size, 1)
            flat_next_states = next_states.reshape(meta_batch * rl_batch_size, STATE_DIM)
            flat_dones = dones.reshape(meta_batch * rl_batch_size, 1)

            with torch.no_grad():
                target_v = target_vf(flat_next_states, z_expanded)
                q_target = flat_rewards + gamma * (1.0 - flat_dones) * target_v

            q1_pred = qf1(flat_states, flat_actions, z_expanded)
            q2_pred = qf2(flat_states, flat_actions, z_expanded)
            q_loss = torch.nn.functional.mse_loss(q1_pred, q_target) + torch.nn.functional.mse_loss(q2_pred, q_target)
            kl_loss = kl_to_standard_normal(posterior_mean, posterior_var)
            encoder_q_loss = q_loss + kl_lambda * kl_loss

            q_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            encoder_q_loss.backward(retain_graph=True)
            q_optimizer.step()
            encoder_optimizer.step()

            # Recompute z after encoder update for policy/value objectives.
            with torch.no_grad():
                posterior_mean, posterior_var = encoder(context)
                z = sample_z(posterior_mean, posterior_var)
                z_expanded = z.unsqueeze(1).expand(-1, rl_batch_size, -1).reshape(meta_batch * rl_batch_size, LATENT_DIM)

            new_actions, log_pi = policy.sample(flat_states, z_expanded)
            q_new_actions = torch.minimum(qf1(flat_states, new_actions, z_expanded), qf2(flat_states, new_actions, z_expanded))

            v_pred = vf(flat_states, z_expanded)
            v_target = (q_new_actions - alpha * log_pi).detach()
            v_loss = torch.nn.functional.mse_loss(v_pred, v_target)
            v_optimizer.zero_grad()
            v_loss.backward()
            v_optimizer.step()

            new_actions, log_pi = policy.sample(flat_states, z_expanded)
            policy_q = torch.minimum(qf1(flat_states, new_actions, z_expanded), qf2(flat_states, new_actions, z_expanded))
            policy_loss = (alpha * log_pi - policy_q).mean()
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            soft_update(vf, target_vf, tau=0.01)
            last_metrics = {
                "q_loss": float(q_loss.detach().cpu()),
                "v_loss": float(v_loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "kl_loss": float(kl_loss.detach().cpu()),
            }

        if iteration % collection_interval == 0:
            for buffer in buffers:
                # PEARL alternates data collection under the prior and under
                # the inferred posterior.  Prior sampling maintains exploration;
                # posterior sampling exploits the task belief inferred from
                # context in the task replay buffer.
                buffer.add(collect_episode(buffer.task, policy, encoder, [], device, rng, use_posterior=False))
                context = buffer.sample(rng, min(context_size, len(buffer.transitions)))
                buffer.add(collect_episode(buffer.task, policy, encoder, context, device, rng, use_posterior=True))

        training_curve.append(
            {
                "iteration": float(iteration + 1),
                "q_loss": last_metrics["q_loss"],
                "v_loss": last_metrics["v_loss"],
                "policy_loss": last_metrics["policy_loss"],
                "kl_loss": last_metrics["kl_loss"],
                "mean_transitions_per_task": mean([len(buffer.transitions) for buffer in buffers]),
            }
        )

    last_metrics["mean_transitions_per_task"] = mean([len(buffer.transitions) for buffer in buffers])
    return encoder, policy, last_metrics, training_curve


def evaluate_faithful(
    encoder: TransitionContextEncoder,
    policy: TanhGaussianPolicy,
    seed: int,
    device: torch.device,
) -> MethodResult:
    rng = random.Random(seed + 10000)
    test_tasks = build_latent_task_splits(seed)["test"]
    encoder.eval()
    policy.eval()
    curve: dict[int, float] = {}

    for k in K_VALUES:
        returns = []
        for task in test_tasks:
            context: list[Transition] = []
            for _ in range(k):
                support = collect_episode(task, policy, encoder, context, device, rng, use_posterior=bool(context))
                context.extend(support)

            query = collect_episode(task, policy, encoder, context, device, rng, use_posterior=bool(context))
            returns.append(sum(item.reward for item in query))
        curve[k] = sum(returns) / len(returns)

    encoder.train()
    policy.train()
    return MethodResult(seed=seed, method="PEARL-faithful", curve=curve)


def posterior_diagnostics(
    encoder: TransitionContextEncoder,
    policy: TanhGaussianPolicy,
    seed: int,
    device: torch.device,
) -> list[PosteriorDiagnosticRow]:
    rng = random.Random(seed + 20000)
    test_tasks = build_latent_task_splits(seed)["test"]
    encoder.eval()
    policy.eval()
    rows: list[PosteriorDiagnosticRow] = []

    for k in K_VALUES:
        returns = []
        mean_norms = []
        var_means = []
        kl_values = []

        for task in test_tasks:
            context: list[Transition] = []
            for _ in range(k):
                support = collect_episode(task, policy, encoder, context, device, rng, use_posterior=bool(context))
                context.extend(support)

            if context:
                with torch.no_grad():
                    context_batch = transition_tensor(context).unsqueeze(0).to(device)
                    posterior_mean, posterior_var = encoder(context_batch)
                    mean_norms.append(float(posterior_mean.norm(dim=-1).mean().cpu()))
                    var_means.append(float(posterior_var.mean().cpu()))
                    kl_values.append(float(kl_to_standard_normal(posterior_mean, posterior_var).cpu()))
            else:
                mean_norms.append(0.0)
                var_means.append(1.0)
                kl_values.append(0.0)

            query = collect_episode(task, policy, encoder, context, device, rng, use_posterior=bool(context))
            returns.append(sum(item.reward for item in query))

        rows.append(
            PosteriorDiagnosticRow(
                seed=seed,
                k=k,
                mean_return=sum(returns) / len(returns),
                posterior_mean_norm=mean(mean_norms),
                posterior_var_mean=mean(var_means),
                posterior_kl=mean(kl_values),
            )
        )

    encoder.train()
    policy.train()
    return rows


def summarize(results: Sequence[MethodResult]) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    for method in sorted({result.method for result in results}):
        method_results = [result for result in results if result.method == method]
        for k in K_VALUES:
            values = [result.curve[k] for result in method_results]
            rows.append(SummaryRow(method, k, mean(values), pstdev(values) if len(values) > 1 else 0.0, len(values)))
    return rows


def write_outputs(results, summary, metrics, training_curves, posterior_rows, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Faithful PEARL scaffold: no latent labels, no BC, no oracle probes",
        "k_values": list(K_VALUES),
        "metrics": {str(seed): value for seed, value in metrics.items()},
        "training_curves": {str(seed): value for seed, value in training_curves.items()},
        "posterior_diagnostics": [asdict(row) for row in posterior_rows],
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
        writer.writerow(["seed", "iteration", "q_loss", "v_loss", "policy_loss", "kl_loss", "mean_transitions_per_task"])
        for seed, rows in training_curves.items():
            for row in rows:
                writer.writerow(
                    [
                        seed,
                        int(row["iteration"]),
                        f"{row['q_loss']:.6f}",
                        f"{row['v_loss']:.6f}",
                        f"{row['policy_loss']:.6f}",
                        f"{row['kl_loss']:.6f}",
                        f"{row['mean_transitions_per_task']:.2f}",
                    ]
                )

    with (output_dir / "posterior_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "k", "mean_return", "posterior_mean_norm", "posterior_var_mean", "posterior_kl"])
        for row in posterior_rows:
            writer.writerow(
                [
                    row.seed,
                    row.k,
                    f"{row.mean_return:.6f}",
                    f"{row.posterior_mean_norm:.6f}",
                    f"{row.posterior_var_mean:.6f}",
                    f"{row.posterior_kl:.6f}",
                ]
            )


def parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run faithful PEARL scaffold.")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--updates-per-iteration", type=int, default=20)
    parser.add_argument("--meta-batch", type=int, default=8)
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--rl-batch-size", type=int, default=64)
    parser.add_argument("--warmup-episodes", type=int, default=2)
    parser.add_argument("--collection-interval", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--kl-lambda", type=float, default=0.1)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", default="experiments/pearl_faithful")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    results = []
    metrics = {}
    training_curves = {}
    posterior_rows = []
    for seed in parse_seeds(args.seeds):
        encoder, policy, seed_metrics, seed_training_curve = train_faithful_pearl(
            seed=seed,
            device=device,
            iterations=args.iterations,
            updates_per_iteration=args.updates_per_iteration,
            meta_batch=args.meta_batch,
            context_size=args.context_size,
            rl_batch_size=args.rl_batch_size,
            warmup_episodes=args.warmup_episodes,
            collection_interval=args.collection_interval,
            gamma=args.gamma,
            alpha=args.alpha,
            kl_lambda=args.kl_lambda,
        )
        result = evaluate_faithful(encoder, policy, seed, device)
        results.append(result)
        metrics[seed] = seed_metrics
        training_curves[seed] = seed_training_curve
        posterior_rows.extend(posterior_diagnostics(encoder, policy, seed, device))
        print(f"seed={seed} metrics={seed_metrics} curve=({format_curve(result.curve)})")

    summary = summarize(results)
    for row in summary:
        print(f"{row.method} K={row.k}: {row.mean_return:.3f} ± {row.std_return:.3f}")

    output_dir = Path(args.output_dir)
    write_outputs(results, summary, metrics, training_curves, posterior_rows, output_dir)
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
