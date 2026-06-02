from random import Random
from unittest import TestCase

from varibad_toy.grid_policies import (
    BayesOptimalGridPolicy,
    RandomWalkPolicy,
)
from varibad_toy.grid_rollout import run_grid_episode
from varibad_toy.gridworld import HiddenGoalGrid


class GridRolloutTest(TestCase):
    def test_bayes_optimal_always_finds_goal(self) -> None:
        policy = BayesOptimalGridPolicy()
        rng = Random(0)
        for _ in range(30):
            env = HiddenGoalGrid.sample_task(rng, size=5)
            result = run_grid_episode(env, policy, max_steps=200, rng=rng)
            self.assertTrue(result.found)
            self.assertIsNotNone(result.steps_to_goal)

    def test_steps_to_goal_at_least_oracle(self) -> None:
        # You can never beat the agent that already knew the goal location.
        policy = BayesOptimalGridPolicy()
        rng = Random(1)
        for _ in range(30):
            env = HiddenGoalGrid.sample_task(rng, size=5)
            result = run_grid_episode(env, policy, max_steps=200, rng=rng)
            self.assertGreaterEqual(result.steps_to_goal, result.oracle_steps)

    def test_bayes_optimal_beats_random_on_average(self) -> None:
        tasks = [HiddenGoalGrid.sample_task(Random(s), size=5) for s in range(40)]

        def mean_steps(policy, seed):
            rng = Random(seed)
            totals = []
            for env in tasks:
                result = run_grid_episode(env, policy, max_steps=300, rng=rng)
                if result.found:
                    totals.append(result.steps_to_goal)
            return sum(totals) / len(totals)

        bayes = mean_steps(BayesOptimalGridPolicy(), 7)
        rand = mean_steps(RandomWalkPolicy(), 7)
        self.assertLess(bayes, rand)
