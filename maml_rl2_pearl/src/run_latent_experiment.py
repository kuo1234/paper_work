"""Run 002: harder latent goal + wind experiment.

This is a step toward the paper-level PEARL setting without introducing a deep
learning dependency that is unavailable in the current environment.  The task
latent is four-dimensional `(goal_x, goal_y, wind_x, wind_y)`, and each method
must infer it from transition context before query evaluation.

The methods are still "lite" compared with full MAML-PPO, RL²-PPO, and
PEARL-SAC, but PEARL-latent now uses the core PEARL idea more faithfully:
posterior inference over latent tasks from off-policy-style context.
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
from typing import Callable, Iterable, Sequence

from goal_navigation import EvaluationPolicy, Trajectory, Transition, Vector, format_curve
from latent_navigation import (
    LatentEvaluationPolicy,
    LatentNavigationTask,
    LatentOraclePolicy,
    LatentRandomPolicy,
    build_latent_task_splits,
    evaluate_latent_k_shot,
)


K_VALUES = (0, 1, 2, 3, 5)
HORIZON = 60
MAX_STEP = 0.08
WIND_RADIUS = 0.035


@dataclass(frozen=True)
class LatentEstimate:
    goal: Vector
    wind: Vector


@dataclass(frozen=True)
class MethodResult:
    seed: int
    method: str
    hyperparameters: dict[str, float | int | str]
    validation_score_k2: float | None
    curve: dict[int, float]


@dataclass(frozen=True)
class SummaryRow:
    method: str
    k: int
    mean_return: float
    std_return: float
    seed_count: int


def add(left: Vector, right: Vector) -> Vector:
    return (left[0] + right[0], left[1] + right[1])


def scale(vector: Vector, factor: float) -> Vector:
    return (vector[0] * factor, vector[1] * factor)


def mean_latent(tasks: Sequence[LatentNavigationTask]) -> LatentEstimate:
    return LatentEstimate(
        goal=(
            sum(task.goal[0] for task in tasks) / len(tasks),
            sum(task.goal[1] for task in tasks) / len(tasks),
        ),
        wind=(
            sum(task.wind[0] for task in tasks) / len(tasks),
            sum(task.wind[1] for task in tasks) / len(tasks),
        ),
    )


def infer_wind(transitions: Sequence[Transition]) -> Vector:
    """Estimate constant wind from transition residuals."""

    if not transitions:
        return (0.0, 0.0)
    residual_x = 0.0
    residual_y = 0.0
    for transition in transitions:
        residual_x += transition.next_state[0] - transition.state[0] - transition.action[0]
        residual_y += transition.next_state[1] - transition.state[1] - transition.action[1]
    return (residual_x / len(transitions), residual_y / len(transitions))


def fit_goal_from_rewards(
    transitions: Sequence[Transition],
    start: Vector,
    learning_rate: float,
    steps: int,
) -> Vector:
    """Fit hidden goal using dense reward distances from observed next states."""

    if not transitions:
        return start

    estimate = start
    for _ in range(steps):
        grad = (0.0, 0.0)
        for transition in transitions:
            observed_distance = -(transition.reward + 0.02 * (transition.action[0] ** 2 + transition.action[1] ** 2))
            predicted_distance = math.dist(estimate, transition.next_state)
            if predicted_distance == 0.0:
                continue
            error = predicted_distance - observed_distance
            direction = (
                (estimate[0] - transition.next_state[0]) / predicted_distance,
                (estimate[1] - transition.next_state[1]) / predicted_distance,
            )
            grad = add(grad, scale(direction, 2.0 * error / len(transitions)))
        estimate = add(estimate, scale(grad, -learning_rate))
    return estimate


class ProbeLatentPolicy(LatentEvaluationPolicy):
    """Base policy with deterministic support probes and query exploitation."""

    probe_actions: tuple[Vector, ...] = (
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
        (1.0, -1.0),
    )

    def __init__(self, initial: LatentEstimate):
        self.initial = initial
        self.current = initial
        self.is_support = False
        self.step_in_trajectory = 0

    def reset_for_latent_task(self, task: LatentNavigationTask) -> None:
        del task
        self.current = self.initial

    def prepare_trajectory(self, is_support: bool, trajectory_index: int) -> None:
        del trajectory_index
        self.is_support = is_support
        self.step_in_trajectory = 0

    def act(self, state: Vector) -> Vector:
        if self.is_support:
            segment = max(1, HORIZON // len(self.probe_actions))
            action = self.probe_actions[(self.step_in_trajectory // segment) % len(self.probe_actions)]
            self.step_in_trajectory += 1
            return action

        self.step_in_trajectory += 1
        return (
            self.current.goal[0] - state[0] - self.current.wind[0],
            self.current.goal[1] - state[1] - self.current.wind[1],
        )


class MAMLLatentPolicy(ProbeLatentPolicy):
    """Gradient-style adaptation over explicit latent parameters."""

    name = "MAML-latent"

    def __init__(self, initial: LatentEstimate, learning_rate: float, steps: int):
        super().__init__(initial)
        self.learning_rate = learning_rate
        self.steps = steps

    def observe_trajectory(self, trajectory: Trajectory) -> None:
        wind = infer_wind(trajectory.transitions)
        goal = fit_goal_from_rewards(trajectory.transitions, self.current.goal, self.learning_rate, self.steps)
        self.current = LatentEstimate(goal=goal, wind=wind)


class RL2LatentPolicy(ProbeLatentPolicy):
    """Memory-based adaptation over explicit latent estimates."""

    name = "RL2-latent"

    def __init__(self, initial: LatentEstimate, learning_rate: float, steps: int, memory_rate: float):
        super().__init__(initial)
        self.learning_rate = learning_rate
        self.steps = steps
        self.memory_rate = memory_rate

    def observe_trajectory(self, trajectory: Trajectory) -> None:
        inferred = LatentEstimate(
            goal=fit_goal_from_rewards(trajectory.transitions, self.current.goal, self.learning_rate, self.steps),
            wind=infer_wind(trajectory.transitions),
        )
        self.current = LatentEstimate(
            goal=add(scale(self.current.goal, 1.0 - self.memory_rate), scale(inferred.goal, self.memory_rate)),
            wind=add(scale(self.current.wind, 1.0 - self.memory_rate), scale(inferred.wind, self.memory_rate)),
        )


class PEARLLatentPolicy(ProbeLatentPolicy):
    """Posterior inference over latent goal + wind candidates."""

    name = "PEARL-latent"

    def __init__(
        self,
        initial: LatentEstimate,
        candidates: Sequence[LatentEstimate],
        temperature: float,
        transition_weight: float,
        prior_weight: float,
    ):
        super().__init__(initial)
        self.candidates = tuple(candidates)
        self.temperature = temperature
        self.transition_weight = transition_weight
        self.prior_weight = prior_weight
        self.context: list[Transition] = []

    def reset_for_latent_task(self, task: LatentNavigationTask) -> None:
        super().reset_for_latent_task(task)
        self.context = []

    def observe_trajectory(self, trajectory: Trajectory) -> None:
        self.context.extend(trajectory.transitions)
        self.current = posterior_latent_mean(
            transitions=self.context,
            candidates=self.candidates,
            prior=self.initial,
            temperature=self.temperature,
            transition_weight=self.transition_weight,
            prior_weight=self.prior_weight,
        )


def build_latent_candidates(train_tasks: Sequence[LatentNavigationTask], grid_size: int = 9) -> list[LatentEstimate]:
    """Build latent candidate grid plus exact train latents for posterior scoring."""

    goals: list[Vector] = [task.goal for task in train_tasks]
    # Keep wind candidates on a small grid.  Crossing every exact train wind
    # with every goal candidate makes posterior validation unnecessarily slow
    # for this dependency-free first pass.
    winds: list[Vector] = []

    goal_step = 2.0 / (grid_size - 1)
    wind_values = (-WIND_RADIUS, 0.0, WIND_RADIUS)
    for x_index in range(grid_size):
        for y_index in range(grid_size):
            goals.append((-1.0 + x_index * goal_step, -1.0 + y_index * goal_step))
    for wx in wind_values:
        for wy in wind_values:
            winds.append((wx, wy))

    unique_goals = list({(round(g[0], 5), round(g[1], 5)): g for g in goals}.values())
    unique_winds = list({(round(w[0], 5), round(w[1], 5)): w for w in winds}.values())

    return [LatentEstimate(goal=goal, wind=wind) for goal in unique_goals for wind in unique_winds]


def posterior_latent_mean(
    transitions: Sequence[Transition],
    candidates: Sequence[LatentEstimate],
    prior: LatentEstimate,
    temperature: float,
    transition_weight: float,
    prior_weight: float,
) -> LatentEstimate:
    """Posterior mean over latent candidates using reward and dynamics evidence."""

    if not transitions:
        return prior

    losses: list[float] = []
    for candidate in candidates:
        reward_loss = 0.0
        transition_loss = 0.0
        for transition in transitions:
            observed_distance = -(transition.reward + 0.02 * (transition.action[0] ** 2 + transition.action[1] ** 2))
            reward_loss += (math.dist(candidate.goal, transition.next_state) - observed_distance) ** 2

            predicted_next = (
                transition.state[0] + transition.action[0] + candidate.wind[0],
                transition.state[1] + transition.action[1] + candidate.wind[1],
            )
            transition_loss += math.dist(predicted_next, transition.next_state) ** 2

        reward_loss /= len(transitions)
        transition_loss /= len(transitions)
        prior_loss = math.dist(candidate.goal, prior.goal) ** 2 + math.dist(candidate.wind, prior.wind) ** 2
        losses.append(reward_loss + transition_weight * transition_loss + prior_weight * prior_loss)

    min_loss = min(losses)
    weights = [math.exp(-(loss - min_loss) / temperature) for loss in losses]
    total = sum(weights)
    if total == 0.0:
        return prior

    goal = (
        sum(candidate.goal[0] * weight for candidate, weight in zip(candidates, weights)) / total,
        sum(candidate.goal[1] * weight for candidate, weight in zip(candidates, weights)) / total,
    )
    wind = (
        sum(candidate.wind[0] * weight for candidate, weight in zip(candidates, weights)) / total,
        sum(candidate.wind[1] * weight for candidate, weight in zip(candidates, weights)) / total,
    )
    return LatentEstimate(goal=goal, wind=wind)


def validation_score(factory: Callable[[], EvaluationPolicy], tasks: Sequence[LatentNavigationTask]) -> float:
    return evaluate_latent_k_shot(factory, tasks, k_values=(2,), horizon=HORIZON)[2]


def choose_best(
    configs: Iterable[tuple[dict[str, float | int | str], Callable[[], EvaluationPolicy]]],
    validation_tasks: Sequence[LatentNavigationTask],
) -> tuple[dict[str, float | int | str], Callable[[], EvaluationPolicy], float]:
    best_params: dict[str, float | int | str] | None = None
    best_factory: Callable[[], EvaluationPolicy] | None = None
    best_score = -math.inf
    for params, factory in configs:
        score = validation_score(factory, validation_tasks)
        if score > best_score:
            best_params = params
            best_factory = factory
            best_score = score
    if best_params is None or best_factory is None:
        raise RuntimeError("No configs supplied.")
    return best_params, best_factory, best_score


def train_method_factories(seed: int) -> dict[str, tuple[dict[str, float | int | str], Callable[[], EvaluationPolicy], float | None]]:
    splits = build_latent_task_splits(seed)
    train_tasks = splits["train"]
    validation_tasks = splits["validation"]
    initial = mean_latent(train_tasks)
    candidates = build_latent_candidates(train_tasks)

    maml_configs = []
    for learning_rate in (0.05, 0.1, 0.2, 0.4):
        for steps in (5, 10, 20, 40):
            params = {"initial": "train_mean_latent", "learning_rate": learning_rate, "steps": steps}
            maml_configs.append((params, lambda lr=learning_rate, s=steps: MAMLLatentPolicy(initial, lr, s)))

    rl2_configs = []
    for learning_rate in (0.05, 0.1, 0.2):
        for steps in (5, 10, 20):
            for memory_rate in (0.25, 0.5, 0.75, 1.0):
                params = {
                    "initial": "train_mean_latent",
                    "learning_rate": learning_rate,
                    "steps": steps,
                    "memory_rate": memory_rate,
                }
                rl2_configs.append(
                    (params, lambda lr=learning_rate, s=steps, mr=memory_rate: RL2LatentPolicy(initial, lr, s, mr))
                )

    pearl_configs = []
    for temperature in (0.005, 0.02):
        for transition_weight in (5.0, 20.0):
            for prior_weight in (0.0, 0.01):
                params = {
                    "initial": "train_mean_latent",
                    "candidate_count": len(candidates),
                    "temperature": temperature,
                    "transition_weight": transition_weight,
                    "prior_weight": prior_weight,
                }
                pearl_configs.append(
                    (
                        params,
                        lambda temp=temperature, tw=transition_weight, pw=prior_weight: PEARLLatentPolicy(
                            initial, candidates, temp, tw, pw
                        ),
                    )
                )

    return {
        "Random": ({"seeded": seed}, lambda: LatentRandomPolicy(random.Random(seed)), None),
        "Latent oracle": ({"uses_hidden_latent": "debug_only"}, LatentOraclePolicy, None),
        "MAML-latent": choose_best(maml_configs, validation_tasks),
        "RL2-latent": choose_best(rl2_configs, validation_tasks),
        "PEARL-latent": choose_best(pearl_configs, validation_tasks),
    }


def run_for_seed(seed: int) -> list[MethodResult]:
    splits = build_latent_task_splits(seed)
    test_tasks = splits["test"]
    trained = train_method_factories(seed)
    results: list[MethodResult] = []
    for method, (params, factory, validation_k2) in trained.items():
        curve = evaluate_latent_k_shot(factory, test_tasks, k_values=K_VALUES, horizon=HORIZON)
        results.append(MethodResult(seed, method, params, validation_k2, curve))
    return results


def summarize(results: Sequence[MethodResult]) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    for method in sorted({result.method for result in results}):
        method_results = [result for result in results if result.method == method]
        for k in K_VALUES:
            values = [result.curve[k] for result in method_results]
            rows.append(SummaryRow(method, k, mean(values), pstdev(values) if len(values) > 1 else 0.0, len(values)))
    return rows


def write_outputs(results: Sequence[MethodResult], summary: Sequence[SummaryRow], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Run 002 latent goal + wind adaptation experiment",
        "k_values": list(K_VALUES),
        "horizon": HORIZON,
        "results": [
            {
                "seed": result.seed,
                "method": result.method,
                "hyperparameters": result.hyperparameters,
                "validation_score_k2": result.validation_score_k2,
                "curve": {str(k): value for k, value in result.curve.items()},
            }
            for result in results
        ],
        "summary": [asdict(row) for row in summary],
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (output_dir / "curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "method", "k", "return"])
        for result in results:
            for k, value in sorted(result.curve.items()):
                writer.writerow([result.seed, result.method, k, f"{value:.6f}"])

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "k", "mean_return", "std_return", "seed_count"])
        for row in summary:
            writer.writerow([row.method, row.k, f"{row.mean_return:.6f}", f"{row.std_return:.6f}", row.seed_count])


def print_summary(results: Sequence[MethodResult], summary: Sequence[SummaryRow]) -> None:
    for result in results:
        print(
            f"seed={result.seed} method={result.method} "
            f"validation_k2={result.validation_score_k2} curve=({format_curve(result.curve)})"
        )
    print("\nAggregate mean ± std:")
    for method in sorted({row.method for row in summary}):
        rows = [row for row in summary if row.method == method]
        print(f"{method}: " + ", ".join(f"K={row.k}: {row.mean_return:.3f} ± {row.std_return:.3f}" for row in rows))


def parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latent goal + wind few-shot experiment.")
    parser.add_argument("--seeds", default="7,13,23", help="Comma-separated seeds.")
    parser.add_argument("--output-dir", default="experiments/latent_run", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: list[MethodResult] = []
    for seed in parse_seeds(args.seeds):
        results.extend(run_for_seed(seed))
    summary = summarize(results)
    output_dir = Path(args.output_dir)
    write_outputs(results, summary, output_dir)
    print_summary(results, summary)
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()
