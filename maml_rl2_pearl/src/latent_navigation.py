"""Harder latent-dynamics navigation environment for paper-facing experiments.

Run 001 used a simple hidden-goal task.  That was useful for validating the
protocol, but it was too easy: dense rewards plus one probing trajectory almost
identify the task.  This module adds a second latent variable, constant wind,
so adaptation must infer both "where should I go?" and "how does the world move?"

The structure intentionally mirrors PEARL's task-inference setup:

* context contains transitions `(s, a, r, s')`;
* the latent task is `(goal_x, goal_y, wind_x, wind_y)`;
* posterior-style methods score latent hypotheses by transition likelihood and
  reward likelihood;
* policies condition their action on the inferred latent task.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Iterable, Sequence

from goal_navigation import EvaluationPolicy, Transition, Trajectory, Vector, clip_vector, euclidean_distance


@dataclass(frozen=True)
class LatentNavigationTask:
    """A task with both hidden reward target and hidden dynamics."""

    goal: Vector
    wind: Vector


class LatentWindNavigationEnv:
    """2D point navigation with task-specific constant wind.

    Dynamics:
        `next_state = state + clipped(action) + wind`

    Reward:
        `-distance(next_state, goal) - action_penalty * ||action||^2`

    Wind makes naive "move directly toward the goal" policies systematically
    drift.  A method can only compensate after it infers wind from context.
    """

    def __init__(
        self,
        task: LatentNavigationTask,
        horizon: int = 60,
        max_step: float = 0.08,
        action_penalty: float = 0.02,
    ):
        self.task = task
        self.horizon = horizon
        self.max_step = max_step
        self.action_penalty = action_penalty
        self.position: Vector = (0.0, 0.0)
        self.step_count = 0

    def reset(self) -> Vector:
        self.position = (0.0, 0.0)
        self.step_count = 0
        return self.position

    def step(self, action: Vector) -> Transition:
        clipped_action = clip_vector(action, self.max_step)
        state = self.position
        next_state = (
            state[0] + clipped_action[0] + self.task.wind[0],
            state[1] + clipped_action[1] + self.task.wind[1],
        )
        self.position = next_state
        self.step_count += 1

        action_cost = self.action_penalty * (clipped_action[0] ** 2 + clipped_action[1] ** 2)
        reward = -euclidean_distance(next_state, self.task.goal) - action_cost
        done = self.step_count >= self.horizon

        return Transition(
            state=state,
            action=clipped_action,
            reward=reward,
            next_state=next_state,
            done=done,
        )


class LatentEvaluationPolicy(EvaluationPolicy):
    """Policy interface for latent goal + wind tasks."""

    def reset_for_latent_task(self, task: LatentNavigationTask) -> None:
        """Clear method-specific state before a new latent task."""

    def reset_for_task(self, task) -> None:  # type: ignore[override]
        """Compatibility shim for the simpler goal-only evaluator."""

        if isinstance(task, LatentNavigationTask):
            self.reset_for_latent_task(task)


class LatentOraclePolicy(LatentEvaluationPolicy):
    """Debug upper bound that directly knows goal and wind."""

    name = "latent-oracle"

    def __init__(self) -> None:
        self.goal: Vector = (0.0, 0.0)
        self.wind: Vector = (0.0, 0.0)

    def reset_for_latent_task(self, task: LatentNavigationTask) -> None:
        self.goal = task.goal
        self.wind = task.wind

    def act(self, state: Vector) -> Vector:
        return (self.goal[0] - state[0] - self.wind[0], self.goal[1] - state[1] - self.wind[1])


class LatentRandomPolicy(LatentEvaluationPolicy):
    """Random baseline for the harder latent-dynamics task."""

    name = "latent-random"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def reset_for_latent_task(self, task: LatentNavigationTask) -> None:
        del task

    def act(self, state: Vector) -> Vector:
        del state
        angle = self.rng.uniform(0.0, 2.0 * math.pi)
        return (math.cos(angle), math.sin(angle))


def sample_latent_tasks(
    count: int,
    rng: random.Random,
    goal_radius: float = 1.0,
    wind_radius: float = 0.035,
) -> list[LatentNavigationTask]:
    """Sample hidden goal and wind from independent square distributions."""

    return [
        LatentNavigationTask(
            goal=(rng.uniform(-goal_radius, goal_radius), rng.uniform(-goal_radius, goal_radius)),
            wind=(rng.uniform(-wind_radius, wind_radius), rng.uniform(-wind_radius, wind_radius)),
        )
        for _ in range(count)
    ]


def build_latent_task_splits(seed: int = 7) -> dict[str, list[LatentNavigationTask]]:
    """Create deterministic splits matching the Run 001 split sizes."""

    rng = random.Random(seed)
    return {
        "train": sample_latent_tasks(40, rng),
        "validation": sample_latent_tasks(10, rng),
        "test": sample_latent_tasks(20, rng),
    }


def collect_latent_trajectory(
    policy: EvaluationPolicy,
    task: LatentNavigationTask,
    horizon: int,
    is_support: bool = False,
    trajectory_index: int = 0,
) -> Trajectory:
    """Collect one latent-dynamics episode under the shared support/query rule."""

    policy.prepare_trajectory(is_support=is_support, trajectory_index=trajectory_index)
    env = LatentWindNavigationEnv(task=task, horizon=horizon)
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


def evaluate_latent_k_shot(
    policy_factory: Callable[[], EvaluationPolicy],
    tasks: Sequence[LatentNavigationTask],
    k_values: Iterable[int] = (0, 1, 2, 3, 5),
    horizon: int = 60,
) -> dict[int, float]:
    """Evaluate on held-out latent tasks with support adaptation then query."""

    averages: dict[int, float] = {}
    for k in k_values:
        returns: list[float] = []
        for task in tasks:
            policy = policy_factory()
            if isinstance(policy, LatentEvaluationPolicy):
                policy.reset_for_latent_task(task)
            else:
                policy.reset_for_task(task)

            for support_index in range(k):
                support = collect_latent_trajectory(
                    policy,
                    task,
                    horizon,
                    is_support=True,
                    trajectory_index=support_index,
                )
                policy.observe_trajectory(support)

            query = collect_latent_trajectory(
                policy,
                task,
                horizon,
                is_support=False,
                trajectory_index=k,
            )
            returns.append(query.total_return)

        averages[k] = sum(returns) / len(returns)
    return averages
