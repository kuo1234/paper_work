from __future__ import annotations

from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class HiddenGoalBandit:
    """Two-arm bandit where one hidden arm has higher reward probability."""

    goal_arm: int
    reward_high: float = 0.8
    reward_low: float = 0.2

    def __post_init__(self) -> None:
        if self.goal_arm not in (0, 1):
            raise ValueError("goal_arm must be 0 or 1")
        if not 0.0 <= self.reward_low <= self.reward_high <= 1.0:
            raise ValueError("require 0 <= reward_low <= reward_high <= 1")

    @classmethod
    def sample_task(
        cls,
        rng: Random,
        reward_high: float = 0.8,
        reward_low: float = 0.2,
    ) -> "HiddenGoalBandit":
        return cls(goal_arm=rng.randrange(2), reward_high=reward_high, reward_low=reward_low)

    def reward_probability(self, action: int) -> float:
        if action not in (0, 1):
            raise ValueError("action must be 0 or 1")
        return self.reward_high if action == self.goal_arm else self.reward_low

    def step(self, action: int, rng: Random) -> int:
        """Return a binary reward sampled from the hidden task."""

        return int(rng.random() < self.reward_probability(action))
