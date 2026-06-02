from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, log2


def _bernoulli_log_prob(value: int, prob: float) -> float:
    if value not in (0, 1):
        raise ValueError("reward must be 0 or 1")
    likelihood = prob if value == 1 else 1.0 - prob
    if likelihood == 0.0:
        return float("-inf")
    return log(likelihood)


@dataclass
class ExactBeliefEncoder:
    """Exact Bayesian task inference for the toy bandit.

    VARIBAD learns an approximate variational encoder q(m | trajectory). In this
    toy setting there are only two tasks, so we can compute the posterior exactly
    and use it as a readable stand-in for the learned belief.
    """

    reward_high: float = 0.8
    reward_low: float = 0.2
    log_prob_goal_0: float = log(0.5)
    log_prob_goal_1: float = log(0.5)

    def update(self, action: int, reward: int) -> None:
        self.log_prob_goal_0 += _bernoulli_log_prob(
            reward,
            self.reward_high if action == 0 else self.reward_low,
        )
        self.log_prob_goal_1 += _bernoulli_log_prob(
            reward,
            self.reward_high if action == 1 else self.reward_low,
        )
        self._normalize()

    def prob_goal_0(self) -> float:
        self._normalize()
        return exp(self.log_prob_goal_0)

    def prob_goal_1(self) -> float:
        self._normalize()
        return exp(self.log_prob_goal_1)

    def probs(self) -> tuple[float, float]:
        return (self.prob_goal_0(), self.prob_goal_1())

    def entropy_bits(self) -> float:
        p0, p1 = self.probs()
        return sum(-p * log2(p) for p in (p0, p1) if p > 0.0)

    def latent(self) -> tuple[float, float, float]:
        """Belief vector analogous to the latent given to a VARIBAD policy."""

        p0, p1 = self.probs()
        return (p0, p1, self.entropy_bits())

    def copy(self) -> "ExactBeliefEncoder":
        return ExactBeliefEncoder(
            reward_high=self.reward_high,
            reward_low=self.reward_low,
            log_prob_goal_0=self.log_prob_goal_0,
            log_prob_goal_1=self.log_prob_goal_1,
        )

    def _normalize(self) -> None:
        max_log = max(self.log_prob_goal_0, self.log_prob_goal_1)
        z = max_log + log(
            exp(self.log_prob_goal_0 - max_log) + exp(self.log_prob_goal_1 - max_log)
        )
        self.log_prob_goal_0 -= z
        self.log_prob_goal_1 -= z
