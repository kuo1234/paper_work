from unittest import TestCase

import torch

from varibad_gridworld.trainers.a2c import a2c_losses, discounted_returns
from varibad_gridworld.utils.torch_rollout import RolloutBatch


class A2CTest(TestCase):
    def test_discounted_returns_hand_computed(self) -> None:
        rewards = torch.tensor([1.0, 0.0, 2.0])
        g = discounted_returns(rewards, gamma=0.5)
        # G2 = 2; G1 = 0 + 0.5*2 = 1; G0 = 1 + 0.5*1 = 1.5
        self.assertAlmostEqual(g[2].item(), 2.0)
        self.assertAlmostEqual(g[1].item(), 1.0)
        self.assertAlmostEqual(g[0].item(), 1.5)

    def test_zero_gamma_is_immediate_reward(self) -> None:
        rewards = torch.tensor([1.0, -0.1, 0.3])
        g = discounted_returns(rewards, gamma=0.0)
        self.assertTrue(torch.allclose(g, rewards))

    def test_a2c_losses_finite_and_backprop(self) -> None:
        T = 5
        # Build a minimal batch with grad-bearing tensors.
        log_probs = torch.zeros(T, requires_grad=True)
        values = torch.zeros(T, requires_grad=True)
        entropies = torch.ones(T, requires_grad=True)
        batch = RolloutBatch(
            log_probs=log_probs,
            values=values,
            rewards=torch.tensor([1.0, -0.1, -0.1, 1.0, 1.0]),
            entropies=entropies,
            actions=[0] * T,
            on_goal_steps=[False, False, False, True, True],
            episodes=[0, 0, 0, 1, 1],
            H=3, N=2,
        )
        losses = a2c_losses(batch, gamma=0.95)
        self.assertTrue(torch.isfinite(losses.total))
        losses.total.backward()  # must not raise
        self.assertIsNotNone(log_probs.grad)

    def test_batch_episode_returns(self) -> None:
        batch = RolloutBatch(
            log_probs=torch.zeros(4),
            values=torch.zeros(4),
            rewards=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            entropies=torch.zeros(4),
            actions=[0, 0, 0, 0],
            on_goal_steps=[False] * 4,
            episodes=[0, 0, 1, 1],
            H=2, N=2,
        )
        self.assertEqual(batch.episode_returns(), [3.0, 7.0])
        self.assertEqual(batch.total_return, 10.0)
