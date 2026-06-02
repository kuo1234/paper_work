from random import Random
from unittest import TestCase

from varibad_toy.bandit import HiddenGoalBandit
from varibad_toy.policies import BayesAdaptivePolicy
from varibad_toy.rollout import run_episode


class RolloutTest(TestCase):
    def test_episode_records_belief_after_each_step(self) -> None:
        rng = Random(7)
        env = HiddenGoalBandit(goal_arm=0, reward_high=1.0, reward_low=0.0)
        policy = BayesAdaptivePolicy(exploration_bonus=0.0)

        steps = run_episode(env, policy, horizon=3, rng=rng)

        self.assertEqual(len(steps), 3)
        self.assertTrue(all(step.task_goal_arm == 0 for step in steps))
        self.assertTrue(all(0.0 <= step.p_goal_0 <= 1.0 for step in steps))
        self.assertGreaterEqual(sum(step.reward for step in steps), 1)
