"""Policy-gradient MAML baseline for the latent navigation task.

This runner meta-learns a Gaussian policy initialization.  For each sampled
task, it collects a support episode, applies one differentiable policy-gradient
inner update, then optimizes the query episode loss from the adapted policy.

The implementation intentionally avoids task labels, oracle probes, latent
supervision, recurrent memory, and PEARL-style posterior inference.  It is a
small MAML-PG baseline for the shared latent navigation benchmark, not a full
MAML-TRPO reproduction.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import argparse
import csv
import json
from pathlib import Path
import random
from statistics import mean, pstdev
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.func import functional_call

from goal_navigation import Vector, clip_vector, format_curve
from latent_navigation import LatentNavigationTask, LatentWindNavigationEnv, build_latent_task_splits


K_VALUES = (0, 1, 2, 3, 5)
HORIZON = 60
STATE_DIM = 2
ACTION_DIM = 2
MAX_STEP = 0.08
LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


ParamMap = OrderedDict[str, torch.Tensor]


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


class GaussianPolicy(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, ACTION_DIM),
        )
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -1.0))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_action = self.net(state)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_action, log_std.expand_as(mean_action)


def policy_outputs(
    policy: GaussianPolicy,
    params: Mapping[str, torch.Tensor],
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_action, log_std = functional_call(policy, params, (state,))
    return mean_action, log_std


def sample_action(
    policy: GaussianPolicy,
    params: Mapping[str, torch.Tensor],
    state: Vector,
    device: torch.device,
) -> tuple[Vector, torch.Tensor]:
    state_tensor = torch.tensor([[state[0], state[1]]], dtype=torch.float32, device=device)
    mean_action, log_std = policy_outputs(policy, params, state_tensor)
    std = log_std.exp()
    normal = torch.distributions.Normal(mean_action, std)
    raw_action = normal.rsample()
    squashed = torch.tanh(raw_action)
    action_tensor = squashed * MAX_STEP
    log_prob = normal.log_prob(raw_action) - torch.log(MAX_STEP * (1.0 - squashed.pow(2)) + 1e-6)
    action = tuple(action_tensor.squeeze(0).detach().cpu().tolist())  # type: ignore[assignment]
    return clip_vector(action, MAX_STEP), log_prob.sum(dim=-1).squeeze(0)


def deterministic_action(
    policy: GaussianPolicy,
    params: Mapping[str, torch.Tensor],
    state: Vector,
    device: torch.device,
) -> Vector:
    state_tensor = torch.tensor([[state[0], state[1]]], dtype=torch.float32, device=device)
    with torch.no_grad():
        mean_action, _ = policy_outputs(policy, params, state_tensor)
        action_tensor = torch.tanh(mean_action) * MAX_STEP
    action = tuple(action_tensor.squeeze(0).cpu().tolist())  # type: ignore[assignment]
    return clip_vector(action, MAX_STEP)


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


def collect_policy_gradient_loss(
    policy: GaussianPolicy,
    params: Mapping[str, torch.Tensor],
    task: LatentNavigationTask,
    device: torch.device,
    gamma: float,
) -> tuple[torch.Tensor, float]:
    env = LatentWindNavigationEnv(task=task, horizon=HORIZON)
    state = env.reset()
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    total_return = 0.0

    for _ in range(HORIZON):
        action, log_prob = sample_action(policy, params, state, device)
        transition = env.step(action)
        log_probs.append(log_prob)
        rewards.append(transition.reward)
        total_return += transition.reward
        state = transition.next_state
        if transition.done:
            break

    returns = discounted_returns(rewards, gamma, device)
    log_prob_tensor = torch.stack(log_probs)
    loss = -(log_prob_tensor * returns.detach()).mean()
    return loss, total_return


def collect_deterministic_return(
    policy: GaussianPolicy,
    params: Mapping[str, torch.Tensor],
    task: LatentNavigationTask,
    device: torch.device,
) -> float:
    env = LatentWindNavigationEnv(task=task, horizon=HORIZON)
    state = env.reset()
    total_return = 0.0
    for _ in range(HORIZON):
        action = deterministic_action(policy, params, state, device)
        transition = env.step(action)
        total_return += transition.reward
        state = transition.next_state
        if transition.done:
            break
    return total_return


def current_params(policy: GaussianPolicy) -> ParamMap:
    return OrderedDict((name, value) for name, value in policy.named_parameters())


def adapt_params(
    policy: GaussianPolicy,
    params: ParamMap,
    task: LatentNavigationTask,
    device: torch.device,
    gamma: float,
    inner_lr: float,
    create_graph: bool,
) -> tuple[ParamMap, float, torch.Tensor]:
    support_loss, support_return = collect_policy_gradient_loss(policy, params, task, device, gamma)
    grads = torch.autograd.grad(support_loss, tuple(params.values()), create_graph=create_graph)
    adapted = OrderedDict(
        (name, param - inner_lr * grad)
        for (name, param), grad in zip(params.items(), grads)
    )
    return adapted, support_return, support_loss


def train_maml(
    seed: int,
    device: torch.device,
    iterations: int,
    meta_batch: int,
    gamma: float,
    inner_lr: float,
) -> tuple[GaussianPolicy, dict[str, float], list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train_tasks = build_latent_task_splits(seed)["train"]
    policy = GaussianPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    curve: list[dict[str, float]] = []
    last_metrics = {"meta_loss": 0.0, "support_return": 0.0, "query_return": 0.0}

    for iteration in range(iterations):
        query_losses = []
        support_returns = []
        query_returns = []
        for _ in range(meta_batch):
            task = train_tasks[rng.randrange(len(train_tasks))]
            base_params = current_params(policy)
            adapted, support_return, _support_loss = adapt_params(
                policy=policy,
                params=base_params,
                task=task,
                device=device,
                gamma=gamma,
                inner_lr=inner_lr,
                create_graph=True,
            )
            query_loss, query_return = collect_policy_gradient_loss(policy, adapted, task, device, gamma)
            query_losses.append(query_loss)
            support_returns.append(support_return)
            query_returns.append(query_return)

        meta_loss = torch.stack(query_losses).mean()
        optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()

        last_metrics = {
            "meta_loss": float(meta_loss.detach().cpu()),
            "support_return": mean(support_returns),
            "query_return": mean(query_returns),
        }
        curve.append({"iteration": float(iteration + 1), **last_metrics})

    return policy, last_metrics, curve


def evaluate_maml(
    policy: GaussianPolicy,
    seed: int,
    device: torch.device,
    gamma: float,
    inner_lr: float,
) -> MethodResult:
    test_tasks = build_latent_task_splits(seed)["test"]
    policy.eval()
    curve: dict[int, float] = {}

    for k in K_VALUES:
        task_returns = []
        for task in test_tasks:
            params = current_params(policy)
            for _ in range(k):
                params, _support_return, _support_loss = adapt_params(
                    policy=policy,
                    params=params,
                    task=task,
                    device=device,
                    gamma=gamma,
                    inner_lr=inner_lr,
                    create_graph=False,
                )
            task_returns.append(collect_deterministic_return(policy, params, task, device))
        curve[k] = sum(task_returns) / len(task_returns)

    policy.train()
    return MethodResult(seed=seed, method="MAML-PG", curve=curve)


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
        "description": "Policy-gradient MAML baseline on latent navigation task",
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
        writer.writerow(["seed", "iteration", "meta_loss", "support_return", "query_return"])
        for seed, rows in training_curves.items():
            for row in rows:
                writer.writerow(
                    [
                        seed,
                        int(row["iteration"]),
                        f"{row['meta_loss']:.6f}",
                        f"{row['support_return']:.6f}",
                        f"{row['query_return']:.6f}",
                    ]
                )


def parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run policy-gradient MAML baseline.")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--meta-batch", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--inner-lr", type=float, default=0.05)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", default="experiments/maml_baseline")
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
        policy, seed_metrics, seed_curve = train_maml(
            seed=seed,
            device=device,
            iterations=args.iterations,
            meta_batch=args.meta_batch,
            gamma=args.gamma,
            inner_lr=args.inner_lr,
        )
        result = evaluate_maml(policy, seed, device, args.gamma, args.inner_lr)
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
