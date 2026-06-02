"""PyTorch/CUDA PEARL-style neural experiment.

This is the first neural-network step toward a paper-level PEARL reproduction.
It deliberately focuses on the PEARL-specific contract before adding SAC:

* an encoder maps context transitions to a latent task estimate z;
* an actor conditions on `(state, z)` and outputs a continuous action;
* training runs on CUDA when available;
* evaluation uses the same K-shot support/query protocol as the earlier runs.

The training target is supervised from the known simulator latent and oracle
action.  That is not full PEARL-SAC yet, but it gives us a verified PyTorch CUDA
path and a clean place to attach critic/replay-buffer losses next.

Sources:
* PyTorch local install and CUDA verification:
  https://pytorch.org/get-started/locally/
* PyTorch saving/loading state_dict pattern:
  https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
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

from goal_navigation import Transition, Trajectory, Vector, clip_vector, format_curve
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - exercised by manual environment setup.
    raise SystemExit(
        "PyTorch is required. Use .venv-py312 and install requirements-cu126.txt first."
    ) from exc

from latent_navigation import (
    LatentNavigationTask,
    LatentWindNavigationEnv,
    build_latent_task_splits,
    collect_latent_trajectory,
)


K_VALUES = (0, 1, 2, 3, 5)
HORIZON = 60
CONTEXT_DIM = 13
LATENT_DIM = 4
STATE_DIM = 2
ACTION_DIM = 2
MAX_STEP = 0.08


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


class ContextEncoder(nn.Module):
    """Encode aggregated transition context into latent `(goal, wind)`."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CONTEXT_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, LATENT_DIM),
        )

    def forward(self, context_features: torch.Tensor) -> torch.Tensor:
        return self.net(context_features)


class ConditionedActor(nn.Module):
    """Continuous actor conditioned on current state and inferred latent z."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + LATENT_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ACTION_DIM),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        raw_action = self.net(torch.cat([state, latent], dim=-1))
        return raw_action * MAX_STEP


class NeuralPEARLPolicy:
    """Evaluation wrapper that follows the K-shot support/query protocol."""

    name = "PEARL-neural"

    def __init__(
        self,
        encoder: ContextEncoder,
        actor: ConditionedActor,
        prior_latent: torch.Tensor,
        device: torch.device,
    ):
        self.encoder = encoder
        self.actor = actor
        self.prior_latent = prior_latent.to(device)
        self.device = device
        self.context: list[Transition] = []
        self.latent = self.prior_latent.clone()

    def prepare_trajectory(self, is_support: bool, trajectory_index: int) -> None:
        del is_support, trajectory_index

    def reset_for_latent_task(self, task: LatentNavigationTask) -> None:
        del task
        self.context = []
        self.latent = self.prior_latent.clone()

    def act(self, state: Vector) -> Vector:
        state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
        latent_tensor = self.latent.view(1, LATENT_DIM)
        with torch.no_grad():
            action = self.actor(state_tensor, latent_tensor).squeeze(0).detach().cpu().tolist()
        return (float(action[0]), float(action[1]))

    def observe_trajectory(self, trajectory: Trajectory) -> None:
        self.context.extend(trajectory.transitions)
        features = context_features(self.context).to(self.device).view(1, CONTEXT_DIM)
        with torch.no_grad():
            self.latent = self.encoder(features).squeeze(0)


def latent_tensor(task: LatentNavigationTask) -> torch.Tensor:
    return torch.tensor([task.goal[0], task.goal[1], task.wind[0], task.wind[1]], dtype=torch.float32)


def oracle_action(state: Vector, task: LatentNavigationTask) -> Vector:
    action = (task.goal[0] - state[0] - task.wind[0], task.goal[1] - state[1] - task.wind[1])
    return clip_vector(action, MAX_STEP)


def probe_action(step: int) -> Vector:
    actions: tuple[Vector, ...] = (
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
        (1.0, -1.0),
    )
    segment = max(1, HORIZON // len(actions))
    return actions[(step // segment) % len(actions)]


def collect_probe_context(task: LatentNavigationTask, probe_count: int) -> list[Transition]:
    """Collect deterministic support context without using the hidden latent."""

    context: list[Transition] = []
    for _ in range(probe_count):
        env = LatentWindNavigationEnv(task=task, horizon=HORIZON)
        state = env.reset()
        for step in range(HORIZON):
            transition = env.step(probe_action(step))
            context.append(transition)
            state = transition.next_state
            if transition.done:
                break
    return context


def context_features(transitions: Sequence[Transition]) -> torch.Tensor:
    """Aggregate variable-length context into fixed features for the encoder."""

    if not transitions:
        return torch.zeros(CONTEXT_DIM, dtype=torch.float32)

    n = float(len(transitions))
    mean_state = [0.0, 0.0]
    mean_action = [0.0, 0.0]
    mean_next = [0.0, 0.0]
    mean_residual = [0.0, 0.0]
    reward_sum = 0.0
    reward_sq_sum = 0.0

    for transition in transitions:
        mean_state[0] += transition.state[0]
        mean_state[1] += transition.state[1]
        mean_action[0] += transition.action[0]
        mean_action[1] += transition.action[1]
        mean_next[0] += transition.next_state[0]
        mean_next[1] += transition.next_state[1]
        mean_residual[0] += transition.next_state[0] - transition.state[0] - transition.action[0]
        mean_residual[1] += transition.next_state[1] - transition.state[1] - transition.action[1]
        reward_sum += transition.reward
        reward_sq_sum += transition.reward * transition.reward

    reward_mean = reward_sum / n
    reward_var = max(0.0, reward_sq_sum / n - reward_mean * reward_mean)
    first_state = transitions[0].state
    last_next = transitions[-1].next_state

    values = [
        mean_state[0] / n,
        mean_state[1] / n,
        mean_action[0] / n,
        mean_action[1] / n,
        mean_next[0] / n,
        mean_next[1] / n,
        reward_mean,
        math.sqrt(reward_var),
        mean_residual[0] / n,
        mean_residual[1] / n,
        first_state[0],
        last_next[0],
        last_next[1],
    ]
    return torch.tensor(values, dtype=torch.float32)


def build_training_dataset(
    tasks: Sequence[LatentNavigationTask],
    rng: random.Random,
    contexts_per_task: int,
    states_per_context: int,
) -> TensorDataset:
    """Create supervised PEARL pretraining data from simulator tasks."""

    context_rows: list[torch.Tensor] = []
    state_rows: list[torch.Tensor] = []
    latent_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []

    for task in tasks:
        for _ in range(contexts_per_task):
            probe_count = rng.choice([1, 2, 3])
            context = collect_probe_context(task, probe_count)
            features = context_features(context)
            target_latent = latent_tensor(task)

            for _ in range(states_per_context):
                state = (rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2))
                action = oracle_action(state, task)
                context_rows.append(features)
                state_rows.append(torch.tensor(state, dtype=torch.float32))
                latent_rows.append(target_latent)
                action_rows.append(torch.tensor(action, dtype=torch.float32))

    return TensorDataset(
        torch.stack(context_rows),
        torch.stack(state_rows),
        torch.stack(latent_rows),
        torch.stack(action_rows),
    )


def train_neural_pearl(
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> tuple[ContextEncoder, ConditionedActor, torch.Tensor, dict[str, float]]:
    """Train encoder and actor on train split, report validation losses."""

    torch.manual_seed(seed)
    rng = random.Random(seed)
    splits = build_latent_task_splits(seed)

    train_data = build_training_dataset(splits["train"], rng, contexts_per_task=4, states_per_context=8)
    val_data = build_training_dataset(splits["validation"], rng, contexts_per_task=2, states_per_context=8)

    encoder = ContextEncoder().to(device)
    actor = ConditionedActor().to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(actor.parameters()), lr=3e-4)

    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        encoder.train()
        actor.train()
        for context_batch, state_batch, latent_batch, action_batch in loader:
            context_batch = context_batch.to(device)
            state_batch = state_batch.to(device)
            latent_batch = latent_batch.to(device)
            action_batch = action_batch.to(device)

            predicted_latent = encoder(context_batch)
            predicted_action = actor(state_batch, predicted_latent)

            latent_loss = nn.functional.mse_loss(predicted_latent, latent_batch)
            action_loss = nn.functional.mse_loss(predicted_action, action_batch)
            loss = latent_loss + 10.0 * action_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    metrics = evaluate_supervised_losses(encoder, actor, val_data, device)
    prior = torch.stack([latent_tensor(task) for task in splits["train"]]).mean(dim=0)
    return encoder, actor, prior, metrics


def evaluate_supervised_losses(
    encoder: ContextEncoder,
    actor: ConditionedActor,
    dataset: TensorDataset,
    device: torch.device,
) -> dict[str, float]:
    encoder.eval()
    actor.eval()
    loader = DataLoader(dataset, batch_size=256)
    latent_losses: list[float] = []
    action_losses: list[float] = []
    with torch.no_grad():
        for context_batch, state_batch, latent_batch, action_batch in loader:
            context_batch = context_batch.to(device)
            state_batch = state_batch.to(device)
            latent_batch = latent_batch.to(device)
            action_batch = action_batch.to(device)
            predicted_latent = encoder(context_batch)
            predicted_action = actor(state_batch, predicted_latent)
            latent_losses.append(float(nn.functional.mse_loss(predicted_latent, latent_batch).detach().cpu()))
            action_losses.append(float(nn.functional.mse_loss(predicted_action, action_batch).detach().cpu()))
    return {"validation_latent_mse": mean(latent_losses), "validation_action_mse": mean(action_losses)}


def evaluate_policy(
    encoder: ContextEncoder,
    actor: ConditionedActor,
    prior: torch.Tensor,
    seed: int,
    device: torch.device,
) -> MethodResult:
    splits = build_latent_task_splits(seed)
    encoder.eval()
    actor.eval()

    def factory() -> NeuralPEARLPolicy:
        return NeuralPEARLPolicy(encoder, actor, prior, device)

    curve = evaluate_neural_k_shot(factory, splits["test"])
    return MethodResult(seed=seed, method="PEARL-neural", curve=curve)


def evaluate_neural_k_shot(factory, tasks: Sequence[LatentNavigationTask]) -> dict[int, float]:
    averages: dict[int, float] = {}
    for k in K_VALUES:
        returns: list[float] = []
        for task in tasks:
            policy = factory()
            policy.reset_for_latent_task(task)
            for _ in range(k):
                support_context = collect_probe_context(task, 1)
                policy.observe_trajectory(Trajectory(tuple(support_context), sum(t.reward for t in support_context)))

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
        averages[k] = sum(returns) / len(returns)
    return averages


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
        "description": "PyTorch CUDA PEARL-style neural context encoder smoke experiment",
        "k_values": list(K_VALUES),
        "horizon": HORIZON,
        "metrics": {str(seed): value for seed, value in metrics.items()},
        "results": [
            {"seed": result.seed, "method": result.method, "curve": {str(k): v for k, v in result.curve.items()}}
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
    parser = argparse.ArgumentParser(description="Train/evaluate PyTorch CUDA PEARL-style neural model.")
    parser.add_argument("--seeds", default="7", help="Comma-separated seeds.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs per seed.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", default="experiments/torch_pearl_smoke")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False.")

    results: list[MethodResult] = []
    metrics_by_seed: dict[int, dict[str, float]] = {}

    for seed in parse_seeds(args.seeds):
        encoder, actor, prior, metrics = train_neural_pearl(
            seed=seed,
            device=requested_device,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        result = evaluate_policy(encoder, actor, prior, seed, requested_device)
        results.append(result)
        metrics_by_seed[seed] = metrics
        print(f"seed={seed} metrics={metrics} curve=({format_curve(result.curve)})")

    summary = summarize(results)
    for row in summary:
        print(f"{row.method} K={row.k}: {row.mean_return:.3f} ± {row.std_return:.3f}")

    output_dir = Path(args.output_dir)
    write_outputs(results, summary, metrics_by_seed, output_dir)
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
