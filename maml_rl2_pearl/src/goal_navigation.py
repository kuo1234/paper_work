"""First-version 2D Goal Navigation scaffold for few-shot meta-RL.

This module intentionally implements only the shared experiment surface:

* a tiny 2D point-mass environment,
* deterministic task splits,
* trajectory collection,
* and a K-shot evaluation loop.

It does not claim to be MAML, RL², or PEARL.  Those algorithms need their own
policy networks, optimizers, replay buffers, and training loops.  The purpose of
this file is to make the first learning version concrete and to mark the exact
places where those algorithms must plug into the same fair evaluation protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
import random
from typing import Callable, Iterable, Sequence


Vector = tuple[float, float]


@dataclass(frozen=True)
class GoalTask:
    """A meta-RL task represented by one hidden goal location.

    The policy observes only the current position in this first-version
    environment.  The goal is intentionally kept outside the observation so that
    a future meta-RL method must infer it from rewards, context, gradient data,
    or recurrent memory instead of reading it directly from the state.
    """

    goal: Vector


@dataclass(frozen=True)
class Transition:
    """One environment transition stored for adaptation or later analysis."""

    state: Vector
    action: Vector
    reward: float
    next_state: Vector
    done: bool


@dataclass(frozen=True)
class Trajectory:
    """A full episode and its return.

    Keeping transitions together is important for all three planned methods:
    MAML uses support trajectories for gradient updates, RL² feeds episode
    history into recurrent state, and PEARL turns transitions into context.
    """

    transitions: tuple[Transition, ...]
    total_return: float


class PointNavigationEnv:
    """Simple dense-reward 2D point navigation environment.

    The environment is deliberately small and dependency-free.  That makes it
    useful as a protocol smoke test before introducing MuJoCo, PyTorch, SAC,
    PPO, or any other moving parts that could hide mistakes in the comparison.
    """

    def __init__(self, task: GoalTask, horizon: int = 50, max_step: float = 0.1):
        self.task = task
        self.horizon = horizon
        self.max_step = max_step
        self.position: Vector = (0.0, 0.0)
        self.step_count = 0

    def reset(self) -> Vector:
        """Reset only the environment state, not any algorithm-side memory.

        RL² must keep its recurrent hidden state across trajectories from the
        same task.  For that reason, environment reset is intentionally separate
        from algorithm reset.  The evaluation loop decides when method state
        should be cleared.
        """

        self.position = (0.0, 0.0)
        self.step_count = 0
        return self.position

    def step(self, action: Vector) -> Transition:
        """Advance the point mass and return a dense negative-distance reward."""

        clipped_action = clip_vector(action, max_norm=self.max_step)
        state = self.position
        next_state = (
            state[0] + clipped_action[0],
            state[1] + clipped_action[1],
        )

        self.position = next_state
        self.step_count += 1

        distance = euclidean_distance(next_state, self.task.goal)
        reward = -distance
        done = self.step_count >= self.horizon

        return Transition(
            state=state,
            action=clipped_action,
            reward=reward,
            next_state=next_state,
            done=done,
        )


class EvaluationPolicy:
    """Small interface used by the shared K-shot evaluator.

    Future implementations should wrap their policy object behind this shape.
    Keeping the evaluator independent from a specific algorithm prevents MAML,
    RL², and PEARL from accidentally using different testing rules.
    """

    name = "base-policy"

    def reset_for_task(self, task: GoalTask) -> None:
        """Clear method state when moving to a new task.

        Examples:
        * MAML would restore the meta-initialized parameters.
        * RL² would reset the RNN hidden state here, and only here.
        * PEARL would clear context and return to the prior over z.
        """

    def prepare_trajectory(self, is_support: bool, trajectory_index: int) -> None:
        """Notify the policy before collecting one trajectory.

        The default is a no-op because simple policies do not need to know this.
        Trainable or adaptive methods may use the hook to separate exploration
        during support collection from exploitation during query evaluation.
        Keeping the mode switch here avoids hiding different support/query rules
        inside separate evaluators for each method.
        """

    def act(self, state: Vector) -> Vector:
        """Choose an action from the currently observable state."""

        raise NotImplementedError

    def observe_trajectory(self, trajectory: Trajectory) -> None:
        """Adapt after one trajectory if the method supports adaptation.

        MAML extension point:
            Use the trajectory as support data and run an inner-loop gradient
            update before the next query evaluation.

        RL² extension point:
            A real recurrent policy would usually update hidden state during
            every step, not only after the full trajectory.  This hook still
            marks the boundary where one episode's information becomes
            available to later episodes.

        PEARL extension point:
            Add transitions to context c and update or resample z from q(z|c).
        """


class RandomPolicy(EvaluationPolicy):
    """Exploration baseline used only to smoke-test the environment."""

    name = "random"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def act(self, state: Vector) -> Vector:
        del state
        angle = self.rng.uniform(0.0, 2.0 * math.pi)
        return (math.cos(angle), math.sin(angle))


class GoalDirectionOracle(EvaluationPolicy):
    """Upper-bound sanity baseline that directly knows the hidden goal.

    This is not a valid meta-RL method because it reads the task goal directly.
    It is useful only as a debugging reference: if this policy cannot get near
    the goal, the environment dynamics or reward calculation is probably wrong.
    """

    name = "goal-direction-oracle"

    def __init__(self) -> None:
        self.goal: Vector = (0.0, 0.0)

    def reset_for_task(self, task: GoalTask) -> None:
        self.goal = task.goal

    def act(self, state: Vector) -> Vector:
        return (self.goal[0] - state[0], self.goal[1] - state[1])


def euclidean_distance(left: Vector, right: Vector) -> float:
    """Return Euclidean distance between two 2D points."""

    return math.sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)


def clip_vector(vector: Vector, max_norm: float) -> Vector:
    """Limit action magnitude while preserving direction.

    A fixed action bound prevents policies from solving the task in a single
    huge step.  That keeps the horizon meaningful and makes adaptation curves
    easier to interpret.
    """

    norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    if norm <= max_norm or norm == 0.0:
        return vector
    scale = max_norm / norm
    return (vector[0] * scale, vector[1] * scale)


def sample_goal_tasks(count: int, rng: random.Random, radius: float = 1.0) -> list[GoalTask]:
    """Sample goal tasks uniformly from a square.

    The simple square distribution is enough for the first version.  Later work
    can replace it with circle, held-out quadrant, or OOD distributions, but all
    compared methods should still receive the same sampled task objects.
    """

    return [
        GoalTask(goal=(rng.uniform(-radius, radius), rng.uniform(-radius, radius)))
        for _ in range(count)
    ]


def build_task_splits(seed: int = 7) -> dict[str, list[GoalTask]]:
    """Create deterministic train/validation/test splits from one seed."""

    rng = random.Random(seed)
    return {
        "train": sample_goal_tasks(40, rng),
        "validation": sample_goal_tasks(10, rng),
        "test": sample_goal_tasks(20, rng),
    }


def collect_trajectory(
    policy: EvaluationPolicy,
    task: GoalTask,
    horizon: int,
    is_support: bool = False,
    trajectory_index: int = 0,
) -> Trajectory:
    """Run one episode and collect transitions.

    This function does not call policy.observe_trajectory.  Adaptation is owned
    by the evaluation loop so that the K-shot meaning remains explicit.
    """

    policy.prepare_trajectory(is_support=is_support, trajectory_index=trajectory_index)
    env = PointNavigationEnv(task=task, horizon=horizon)
    state = env.reset()
    transitions: list[Transition] = []
    total_return = 0.0

    for _ in range(horizon):
        action = policy.act(state)
        transition = env.step(action)
        transitions.append(transition)
        total_return += transition.reward
        state = transition.next_state
        if transition.done:
            break

    return Trajectory(transitions=tuple(transitions), total_return=total_return)


def evaluate_k_shot(
    policy_factory: Callable[[], EvaluationPolicy],
    tasks: Sequence[GoalTask],
    k_values: Iterable[int] = (0, 1, 2, 3, 5),
    horizon: int = 50,
) -> dict[int, float]:
    """Evaluate a policy under the shared K-shot protocol.

    For K=0, the policy is evaluated before any adaptation trajectory is passed
    to observe_trajectory.

    For K>0, the policy first receives K support trajectories from the same
    task, then one query trajectory is collected and scored.  This matches the
    MAML and PEARL interpretation directly.  RL² implementations can adapt this
    wrapper later so that hidden state is updated at every step, while still
    preserving the rule that hidden state is reset only between tasks.
    """

    average_returns: dict[int, float] = {}

    for k in k_values:
        returns_for_k: list[float] = []

        for task in tasks:
            policy = policy_factory()
            policy.reset_for_task(task)

            for support_index in range(k):
                support = collect_trajectory(
                    policy,
                    task,
                    horizon,
                    is_support=True,
                    trajectory_index=support_index,
                )
                policy.observe_trajectory(support)

            query = collect_trajectory(
                policy,
                task,
                horizon,
                is_support=False,
                trajectory_index=k,
            )
            returns_for_k.append(query.total_return)

        average_returns[k] = sum(returns_for_k) / len(returns_for_k)

    return average_returns


def format_curve(curve: dict[int, float]) -> str:
    """Render compact CLI output for smoke-test results."""

    return ", ".join(f"K={k}: {value:.3f}" for k, value in sorted(curve.items()))


def run_smoke_test(seed: int) -> None:
    """Run deterministic baselines to verify that the scaffold executes."""

    splits = build_task_splits(seed=seed)
    test_tasks = splits["test"]

    random_curve = evaluate_k_shot(
        policy_factory=lambda: RandomPolicy(random.Random(seed)),
        tasks=test_tasks,
    )
    oracle_curve = evaluate_k_shot(
        policy_factory=GoalDirectionOracle,
        tasks=test_tasks,
    )

    print(f"Task split sizes: train={len(splits['train'])}, validation={len(splits['validation'])}, test={len(test_tasks)}")
    print(f"Random smoke baseline: {format_curve(random_curve)}")
    print(f"Goal-direction oracle: {format_curve(oracle_curve)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D Goal Navigation first-version scaffold")
    parser.add_argument("--seed", type=int, default=7, help="Seed for deterministic task splits and smoke baselines.")
    return parser.parse_args()


if __name__ == "__main__":
    run_smoke_test(seed=parse_args().seed)
