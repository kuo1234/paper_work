"""Run the first train/test experiment for 2D Goal Navigation.

The goal of this runner is to produce a real, repeatable first result without
pretending that we already implemented full MAML-PPO, RL²-PPO, or PEARL-SAC.
Each method below is a deliberately small "lite" proxy that preserves the
adaptation mechanism we want to study:

* MAML-lite learns a shared initial goal estimate and adapts it by gradient
  descent on support trajectory rewards.
* RL2-lite keeps a task memory estimate and updates that memory after each
  support trajectory.
* PEARL-lite performs posterior-style inference over candidate latent goals
  using context transitions.

All three methods are evaluated through the same K-shot protocol from
goal_navigation.py, so the first result is about adaptation curves rather than
about heavyweight neural-network implementation details.
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

from goal_navigation import (
    EvaluationPolicy,
    GoalDirectionOracle,
    GoalTask,
    RandomPolicy,
    Trajectory,
    Transition,
    Vector,
    build_task_splits,
    collect_trajectory,
    evaluate_k_shot,
    format_curve,
)


K_VALUES = (0, 1, 2, 3, 5)
HORIZON = 50


@dataclass(frozen=True)
class MethodResult:
    """One method's averaged curve for one seed."""

    seed: int
    method: str
    hyperparameters: dict[str, float | int | str]
    validation_score_k2: float | None
    curve: dict[int, float]


@dataclass(frozen=True)
class SummaryRow:
    """Mean/std summary for one method and K value across seeds."""

    method: str
    k: int
    mean_return: float
    std_return: float
    seed_count: int


def add(left: Vector, right: Vector) -> Vector:
    return (left[0] + right[0], left[1] + right[1])


def scale(vector: Vector, factor: float) -> Vector:
    return (vector[0] * factor, vector[1] * factor)


def mean_goal(tasks: Sequence[GoalTask]) -> Vector:
    """Learn the first-version meta-prior: the average training goal."""

    return (
        sum(task.goal[0] for task in tasks) / len(tasks),
        sum(task.goal[1] for task in tasks) / len(tasks),
    )


def flatten_transitions(trajectories: Iterable[Trajectory]) -> list[Transition]:
    transitions: list[Transition] = []
    for trajectory in trajectories:
        transitions.extend(trajectory.transitions)
    return transitions


def fit_goal_by_reward_distances(
    transitions: Sequence[Transition],
    start: Vector,
    learning_rate: float,
    steps: int,
) -> Vector:
    """Infer a goal estimate from dense rewards by gradient descent.

    Dense reward is `-distance(next_state, true_goal)`.  Therefore each observed
    transition tells us the distance from `next_state` to the hidden goal.  The
    lite methods use this signal to infer a task variable without directly
    reading the task goal.
    """

    if not transitions:
        return start

    estimate = start
    for _ in range(steps):
        grad = (0.0, 0.0)
        for transition in transitions:
            observed_distance = -transition.reward
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


class GoalEstimatePolicy(EvaluationPolicy):
    """Move toward the current hidden-goal estimate."""

    def __init__(self, initial_goal: Vector):
        self.initial_goal = initial_goal
        self.current_goal = initial_goal
        self.is_support = False
        self.step_in_trajectory = 0

    def reset_for_task(self, task: GoalTask) -> None:
        del task
        self.current_goal = self.initial_goal

    def prepare_trajectory(self, is_support: bool, trajectory_index: int) -> None:
        del trajectory_index
        self.is_support = is_support
        self.step_in_trajectory = 0

    def act(self, state: Vector) -> Vector:
        self.step_in_trajectory += 1
        return (self.current_goal[0] - state[0], self.current_goal[1] - state[1])


class ProbeThenExploitPolicy(GoalEstimatePolicy):
    """Use fixed exploration probes for support and goal estimate for query.

    The lite experiment needs support trajectories that actually reveal the
    hidden goal through dense rewards.  A purely exploitative support rollout can
    keep visiting the same region and make gradient/memory updates ambiguous.
    This probing pattern is intentionally deterministic so every method sees
    comparable support data for a given task and K.
    """

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

    def act(self, state: Vector) -> Vector:
        if self.is_support:
            segment = max(1, HORIZON // len(self.probe_actions))
            action = self.probe_actions[(self.step_in_trajectory // segment) % len(self.probe_actions)]
            self.step_in_trajectory += 1
            return action
        return super().act(state)


class MAMLLitePolicy(ProbeThenExploitPolicy):
    """Gradient-adaptation proxy for MAML.

    Full MAML would update neural policy parameters.  This first version updates
    a compact task-sensitive parameter: the policy's goal estimate.  The point is
    to test whether support trajectories can improve query return after one or
    more gradient-style adaptation steps.
    """

    name = "MAML-lite"

    def __init__(self, initial_goal: Vector, learning_rate: float, steps: int):
        super().__init__(initial_goal)
        self.learning_rate = learning_rate
        self.steps = steps

    def observe_trajectory(self, trajectory: Trajectory) -> None:
        self.current_goal = fit_goal_by_reward_distances(
            transitions=trajectory.transitions,
            start=self.current_goal,
            learning_rate=self.learning_rate,
            steps=self.steps,
        )


class RL2LitePolicy(ProbeThenExploitPolicy):
    """Recurrent-memory proxy for RL².

    A real RL² policy would update an RNN hidden state at every step.  This proxy
    stores an explicit hidden goal estimate and updates it after each support
    trajectory.  The exponential memory update makes the mechanism visibly
    different from MAML-lite's direct overwrite-by-gradient behavior.
    """

    name = "RL2-lite"

    def __init__(self, initial_goal: Vector, learning_rate: float, steps: int, memory_rate: float):
        super().__init__(initial_goal)
        self.learning_rate = learning_rate
        self.steps = steps
        self.memory_rate = memory_rate

    def observe_trajectory(self, trajectory: Trajectory) -> None:
        inferred_goal = fit_goal_by_reward_distances(
            transitions=trajectory.transitions,
            start=self.current_goal,
            learning_rate=self.learning_rate,
            steps=self.steps,
        )
        retained_memory = scale(self.current_goal, 1.0 - self.memory_rate)
        new_memory = scale(inferred_goal, self.memory_rate)
        self.current_goal = add(retained_memory, new_memory)


class PEARLLitePolicy(ProbeThenExploitPolicy):
    """Posterior-inference proxy for PEARL.

    The candidate goals play the role of a small latent z space.  Context
    transitions score each candidate by reward-distance consistency; the policy
    then acts toward the posterior-weighted candidate mean.
    """

    name = "PEARL-lite"

    def __init__(
        self,
        initial_goal: Vector,
        candidate_goals: Sequence[Vector],
        temperature: float,
        prior_weight: float,
    ):
        super().__init__(initial_goal)
        self.candidate_goals = tuple(candidate_goals)
        self.temperature = temperature
        self.prior_weight = prior_weight
        self.context: list[Transition] = []

    def reset_for_task(self, task: GoalTask) -> None:
        super().reset_for_task(task)
        self.context = []

    def observe_trajectory(self, trajectory: Trajectory) -> None:
        self.context.extend(trajectory.transitions)
        self.current_goal = posterior_mean_goal(
            transitions=self.context,
            candidates=self.candidate_goals,
            prior_mean=self.initial_goal,
            temperature=self.temperature,
            prior_weight=self.prior_weight,
        )


def posterior_mean_goal(
    transitions: Sequence[Transition],
    candidates: Sequence[Vector],
    prior_mean: Vector,
    temperature: float,
    prior_weight: float,
) -> Vector:
    """Compute a stable posterior-weighted mean over candidate latent goals."""

    if not transitions:
        return prior_mean

    losses: list[float] = []
    for candidate in candidates:
        reward_loss = 0.0
        for transition in transitions:
            observed_distance = -transition.reward
            predicted_distance = math.dist(candidate, transition.next_state)
            reward_loss += (predicted_distance - observed_distance) ** 2
        reward_loss /= len(transitions)

        prior_loss = math.dist(candidate, prior_mean) ** 2
        losses.append(reward_loss + prior_weight * prior_loss)

    min_loss = min(losses)
    weights = [math.exp(-(loss - min_loss) / temperature) for loss in losses]
    total_weight = sum(weights)
    if total_weight == 0.0:
        return prior_mean

    weighted_x = sum(candidate[0] * weight for candidate, weight in zip(candidates, weights))
    weighted_y = sum(candidate[1] * weight for candidate, weight in zip(candidates, weights))
    return (weighted_x / total_weight, weighted_y / total_weight)


def build_candidate_goals(train_tasks: Sequence[GoalTask], grid_size: int = 17) -> list[Vector]:
    """Create PEARL-lite latent candidates from train goals plus a coarse grid."""

    candidates = [task.goal for task in train_tasks]
    if grid_size < 2:
        return candidates

    step = 2.0 / (grid_size - 1)
    for x_index in range(grid_size):
        for y_index in range(grid_size):
            candidates.append((-1.0 + x_index * step, -1.0 + y_index * step))

    # Deduplicate rounded coordinates so posterior scoring stays deterministic.
    unique: dict[tuple[float, float], Vector] = {}
    for candidate in candidates:
        unique[(round(candidate[0], 6), round(candidate[1], 6))] = candidate
    return list(unique.values())


def validation_score(
    policy_factory: Callable[[], EvaluationPolicy],
    tasks: Sequence[GoalTask],
) -> float:
    """Tune on K=2, matching the planned sample-efficiency comparison point."""

    curve = evaluate_k_shot(policy_factory=policy_factory, tasks=tasks, k_values=(2,), horizon=HORIZON)
    return curve[2]


def choose_best(
    factories: Iterable[tuple[dict[str, float | int | str], Callable[[], EvaluationPolicy]]],
    validation_tasks: Sequence[GoalTask],
) -> tuple[dict[str, float | int | str], Callable[[], EvaluationPolicy], float]:
    """Select hyperparameters by validation K=2 return."""

    best_params: dict[str, float | int | str] | None = None
    best_factory: Callable[[], EvaluationPolicy] | None = None
    best_score = -math.inf

    for params, factory in factories:
        score = validation_score(factory, validation_tasks)
        if score > best_score:
            best_params = params
            best_factory = factory
            best_score = score

    if best_params is None or best_factory is None:
        raise RuntimeError("No method configurations were provided.")

    return best_params, best_factory, best_score


def train_method_factories(seed: int) -> dict[str, tuple[dict[str, float | int | str], Callable[[], EvaluationPolicy], float | None]]:
    """Train or tune all first-version methods for one seed."""

    splits = build_task_splits(seed=seed)
    train_tasks = splits["train"]
    validation_tasks = splits["validation"]
    initial_goal = mean_goal(train_tasks)
    candidates = build_candidate_goals(train_tasks)

    random_factory = lambda: RandomPolicy(random.Random(seed))
    oracle_factory = GoalDirectionOracle

    maml_configs = []
    for learning_rate in (0.05, 0.1, 0.2, 0.4):
        for steps in (5, 10, 20):
            params = {"initial_goal": "train_mean", "learning_rate": learning_rate, "steps": steps}
            maml_configs.append(
                (
                    params,
                    lambda lr=learning_rate, s=steps: MAMLLitePolicy(initial_goal, lr, s),
                )
            )

    rl2_configs = []
    for learning_rate in (0.05, 0.1, 0.2):
        for steps in (5, 10, 20):
            for memory_rate in (0.25, 0.5, 0.75, 1.0):
                params = {
                    "initial_goal": "train_mean",
                    "learning_rate": learning_rate,
                    "steps": steps,
                    "memory_rate": memory_rate,
                }
                rl2_configs.append(
                    (
                        params,
                        lambda lr=learning_rate, s=steps, mr=memory_rate: RL2LitePolicy(initial_goal, lr, s, mr),
                    )
                )

    pearl_configs = []
    for temperature in (0.001, 0.005, 0.01, 0.05, 0.1):
        for prior_weight in (0.0, 0.01, 0.05, 0.1):
            params = {
                "initial_goal": "train_mean",
                "candidate_count": len(candidates),
                "temperature": temperature,
                "prior_weight": prior_weight,
            }
            pearl_configs.append(
                (
                    params,
                    lambda temp=temperature, pw=prior_weight: PEARLLitePolicy(initial_goal, candidates, temp, pw),
                )
            )

    maml = choose_best(maml_configs, validation_tasks)
    rl2 = choose_best(rl2_configs, validation_tasks)
    pearl = choose_best(pearl_configs, validation_tasks)

    return {
        "Random": ({"seeded": seed}, random_factory, None),
        "Goal-direction oracle": ({"uses_hidden_goal": "debug_only"}, oracle_factory, None),
        "MAML-lite": maml,
        "RL2-lite": rl2,
        "PEARL-lite": pearl,
    }


def run_for_seed(seed: int) -> list[MethodResult]:
    """Train/tune on train+validation and evaluate on held-out test tasks."""

    splits = build_task_splits(seed=seed)
    test_tasks = splits["test"]
    trained = train_method_factories(seed)

    results: list[MethodResult] = []
    for method, (params, factory, validation_k2) in trained.items():
        curve = evaluate_k_shot(policy_factory=factory, tasks=test_tasks, k_values=K_VALUES, horizon=HORIZON)
        results.append(
            MethodResult(
                seed=seed,
                method=method,
                hyperparameters=params,
                validation_score_k2=validation_k2,
                curve=curve,
            )
        )
    return results


def summarize(results: Sequence[MethodResult]) -> list[SummaryRow]:
    """Aggregate method curves across seeds."""

    methods = sorted({result.method for result in results})
    rows: list[SummaryRow] = []
    for method in methods:
        method_results = [result for result in results if result.method == method]
        for k in K_VALUES:
            values = [result.curve[k] for result in method_results]
            rows.append(
                SummaryRow(
                    method=method,
                    k=k,
                    mean_return=mean(values),
                    std_return=pstdev(values) if len(values) > 1 else 0.0,
                    seed_count=len(values),
                )
            )
    return rows


def write_outputs(results: Sequence[MethodResult], summary: Sequence[SummaryRow], output_dir: Path) -> None:
    """Persist machine-readable and spreadsheet-friendly results."""

    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "description": "First-version 2D Goal Navigation lite experiment",
        "k_values": list(K_VALUES),
        "horizon": HORIZON,
        "results": [
            {
                "seed": result.seed,
                "method": result.method,
                "hyperparameters": result.hyperparameters,
                "validation_score_k2": result.validation_score_k2,
                "curve": {str(k): v for k, v in result.curve.items()},
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
    """Show enough information to inspect the experiment from the terminal."""

    for result in results:
        print(
            f"seed={result.seed} method={result.method} "
            f"validation_k2={result.validation_score_k2} curve=({format_curve(result.curve)})"
        )

    print("\nAggregate mean ± std:")
    for method in sorted({row.method for row in summary}):
        parts = []
        for row in [item for item in summary if item.method == method]:
            parts.append(f"K={row.k}: {row.mean_return:.3f} ± {row.std_return:.3f}")
        print(f"{method}: " + ", ".join(parts))


def parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run first-version 2D Goal Navigation training/testing.")
    parser.add_argument("--seeds", default="7,13,23", help="Comma-separated seeds for repeated runs.")
    parser.add_argument("--output-dir", default="experiments/first_run", help="Directory for JSON/CSV results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    all_results: list[MethodResult] = []

    for seed in seeds:
        all_results.extend(run_for_seed(seed))

    summary = summarize(all_results)
    output_dir = Path(args.output_dir)
    write_outputs(all_results, summary, output_dir)
    print_summary(all_results, summary)
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()
