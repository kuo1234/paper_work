import math
import unittest

from goal_navigation import Transition
from run_latent_experiment import (
    LatentEstimate,
    infer_wind,
    posterior_latent_mean,
)


class LatentNavigationTests(unittest.TestCase):
    def test_infer_wind_from_transition_residuals(self) -> None:
        transitions = [
            Transition(state=(0.0, 0.0), action=(0.1, 0.0), reward=-1.0, next_state=(0.12, -0.03), done=False),
            Transition(state=(0.5, 0.5), action=(0.0, -0.1), reward=-1.0, next_state=(0.52, 0.37), done=False),
        ]

        wind = infer_wind(transitions)

        self.assertAlmostEqual(wind[0], 0.02)
        self.assertAlmostEqual(wind[1], -0.03)

    def test_posterior_prefers_candidate_matching_reward_and_dynamics(self) -> None:
        true_candidate = LatentEstimate(goal=(0.5, 0.0), wind=(0.02, 0.0))
        wrong_candidate = LatentEstimate(goal=(-0.5, 0.0), wind=(-0.02, 0.0))
        transition = Transition(
            state=(0.0, 0.0),
            action=(0.1, 0.0),
            reward=-math.dist((0.12, 0.0), true_candidate.goal) - 0.02 * 0.1**2,
            next_state=(0.12, 0.0),
            done=False,
        )

        posterior = posterior_latent_mean(
            transitions=[transition],
            candidates=[true_candidate, wrong_candidate],
            prior=LatentEstimate(goal=(0.0, 0.0), wind=(0.0, 0.0)),
            temperature=0.001,
            transition_weight=20.0,
            prior_weight=0.0,
        )

        self.assertGreater(posterior.goal[0], 0.45)
        self.assertGreater(posterior.wind[0], 0.015)


if __name__ == "__main__":
    unittest.main()
