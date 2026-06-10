from random import Random
from unittest import TestCase

from varibad_gridworld.configs.gridworld import ENV_CONFIG
from varibad_gridworld.envs.gridworld import GridWorldTask
from varibad_gridworld.models.rl2 import RL2Policy
from varibad_gridworld.utils.torch_rollout import collect_rollout


def make_env():
    return GridWorldTask(
        allowed_goals=ENV_CONFIG["allowed_goals"],
        start_state=ENV_CONFIG["start_state"],
        episode_horizon=ENV_CONFIG["episode_horizon"],
    )


class TorchRolloutTest(TestCase):
    def test_rollout_length_is_H_plus(self) -> None:
        model = RL2Policy()
        batch = collect_rollout(make_env(), model, N=4, rng=Random(0), task_rng=Random(0))
        self.assertEqual(len(batch.rewards), 60)        # H+ = 4*15
        self.assertEqual(len(batch.log_probs), 60)
        self.assertEqual(len(batch.episodes), 60)

    def test_episode_labels(self) -> None:
        model = RL2Policy()
        batch = collect_rollout(make_env(), model, N=4, rng=Random(0), task_rng=Random(0))
        # 15 steps in each of 4 episodes
        for ep in range(4):
            self.assertEqual(batch.episodes.count(ep), 15)

    def test_episode_returns_sum_to_total(self) -> None:
        model = RL2Policy()
        batch = collect_rollout(make_env(), model, N=4, rng=Random(0), task_rng=Random(1))
        self.assertAlmostEqual(sum(batch.episode_returns()), batch.total_return, places=4)

    def test_deterministic_is_reproducible(self) -> None:
        model = RL2Policy()
        b1 = collect_rollout(make_env(), model, N=4, rng=Random(0), task_rng=Random(5), deterministic=True)
        b2 = collect_rollout(make_env(), model, N=4, rng=Random(99), task_rng=Random(5), deterministic=True)
        # deterministic action selection + same task seed -> identical actions
        self.assertEqual(b1.actions, b2.actions)
