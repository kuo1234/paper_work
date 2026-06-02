from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .belief import ExactBeliefEncoder


class GreedyBeliefPolicy:
    """Exploit the arm that is currently most likely to be the goal arm."""

    def act(self, belief: ExactBeliefEncoder, rng: Random) -> int:
        p0, p1 = belief.probs()
        if p0 == p1:
            return rng.randrange(2)
        return int(p1 > p0)


class ThompsonPolicy:
    """Sample a task from belief, then act optimally for that sampled task."""

    def act(self, belief: ExactBeliefEncoder, rng: Random) -> int:
        return 0 if rng.random() < belief.prob_goal_0() else 1


@dataclass
class BayesAdaptivePolicy:
    """Reward plus information-gain policy for the toy Bayes-adaptive MDP."""

    exploration_bonus: float = 0.5

    def act(self, belief: ExactBeliefEncoder, rng: Random) -> int:
        values = [self._score_action(belief, action) for action in (0, 1)]
        if values[0] == values[1]:
            return rng.randrange(2)
        return int(values[1] > values[0])

    def _score_action(self, belief: ExactBeliefEncoder, action: int) -> float:
        return self._expected_reward(belief, action) + (
            self.exploration_bonus * self._expected_information_gain(belief, action)
        )

    def _expected_reward(self, belief: ExactBeliefEncoder, action: int) -> float:
        p_goal_0, p_goal_1 = belief.probs()
        if action == 0:
            return p_goal_0 * belief.reward_high + p_goal_1 * belief.reward_low
        return p_goal_1 * belief.reward_high + p_goal_0 * belief.reward_low

    def _expected_information_gain(self, belief: ExactBeliefEncoder, action: int) -> float:
        before = belief.entropy_bits()
        expected_after = 0.0
        reward_prob = self._expected_reward(belief, action)
        for reward, prob in ((1, reward_prob), (0, 1.0 - reward_prob)):
            updated = belief.copy()
            updated.update(action, reward)
            expected_after += prob * updated.entropy_bits()
        return before - expected_after
