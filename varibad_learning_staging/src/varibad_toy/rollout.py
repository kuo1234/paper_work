from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Protocol

from .bandit import HiddenGoalBandit
from .belief import ExactBeliefEncoder


class Policy(Protocol):
    def act(self, belief: ExactBeliefEncoder, rng: Random) -> int:
        ...


@dataclass(frozen=True)
class EpisodeStep:
    t: int
    task_goal_arm: int
    action: int
    reward: int
    p_goal_0: float
    entropy_bits: float


def run_episode(
    env: HiddenGoalBandit,
    policy: Policy,
    horizon: int,
    rng: Random,
) -> list[EpisodeStep]:
    belief = ExactBeliefEncoder(
        reward_high=env.reward_high,
        reward_low=env.reward_low,
    )
    steps: list[EpisodeStep] = []

    for t in range(horizon):
        action = policy.act(belief, rng)
        reward = env.step(action, rng)
        belief.update(action, reward)
        steps.append(
            EpisodeStep(
                t=t,
                task_goal_arm=env.goal_arm,
                action=action,
                reward=reward,
                p_goal_0=belief.prob_goal_0(),
                entropy_bits=belief.entropy_bits(),
            )
        )

    return steps
