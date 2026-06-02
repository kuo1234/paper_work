from unittest import TestCase

from varibad_toy.belief import ExactBeliefEncoder


class ExactBeliefEncoderTest(TestCase):
    def test_reward_on_arm_zero_increases_belief_that_zero_is_goal(self) -> None:
        belief = ExactBeliefEncoder(reward_high=0.8, reward_low=0.2)

        belief.update(action=0, reward=1)

        self.assertGreater(belief.prob_goal_0(), 0.5)
        self.assertAlmostEqual(belief.prob_goal_0(), 0.8)
        self.assertAlmostEqual(belief.prob_goal_1(), 0.2)

    def test_consistent_evidence_reduces_entropy(self) -> None:
        belief = ExactBeliefEncoder(reward_high=0.8, reward_low=0.2)
        before = belief.entropy_bits()

        for _ in range(4):
            belief.update(action=0, reward=1)

        self.assertLess(belief.entropy_bits(), before)
        self.assertGreater(belief.prob_goal_0(), 0.99)
