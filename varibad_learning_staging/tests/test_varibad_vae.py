from random import Random
from unittest import TestCase

import torch

from varibad_gridworld.configs.gridworld import ENV_CONFIG
from varibad_gridworld.envs.gridworld import GridWorldTask
from varibad_gridworld.models.varibad import VariBAD
from varibad_gridworld.trainers.varibad_vae import default_context_times, vae_losses
from varibad_gridworld.utils.torch_rollout import collect_varibad_rollout


def make_env():
    return GridWorldTask(
        allowed_goals=ENV_CONFIG["allowed_goals"],
        start_state=ENV_CONFIG["start_state"],
        episode_horizon=ENV_CONFIG["episode_horizon"],
    )


class VariBADRolloutTest(TestCase):
    def test_rollout_length_and_transitions(self) -> None:
        model = VariBAD()
        batch = collect_varibad_rollout(make_env(), model, N=4, rng=Random(0), task_rng=Random(0))
        self.assertEqual(len(batch.rewards), 60)
        self.assertEqual(tuple(batch.states.shape), (60, 25))
        self.assertEqual(tuple(batch.actions_oh.shape), (60, 5))
        self.assertEqual(tuple(batch.next_states.shape), (60, 25))
        self.assertEqual(tuple(batch.inputs.shape), (60, 32))

    def test_reward_labels_match_on_goal(self) -> None:
        model = VariBAD()
        batch = collect_varibad_rollout(make_env(), model, N=4, rng=Random(0), task_rng=Random(2))
        for label, on_goal in zip(batch.reward_labels.tolist(), batch.on_goal_steps):
            self.assertEqual(label, 1.0 if on_goal else 0.0)

    def test_transitions_are_detached(self) -> None:
        model = VariBAD()
        batch = collect_varibad_rollout(make_env(), model, N=4, rng=Random(0), task_rng=Random(0))
        self.assertFalse(batch.states.requires_grad)
        self.assertFalse(batch.next_states.requires_grad)


class VariBADVAETest(TestCase):
    def test_context_times_clamped(self) -> None:
        self.assertTrue(all(t <= 60 for t in default_context_times(60)))
        self.assertTrue(all(t <= 30 for t in default_context_times(30)))

    def test_vae_loss_finite_and_backprop(self) -> None:
        model = VariBAD()
        batch = collect_varibad_rollout(make_env(), model, N=4, rng=Random(0), task_rng=Random(3))
        losses = vae_losses(model, batch, beta=0.1)
        self.assertTrue(torch.isfinite(losses.total))
        self.assertGreaterEqual(float(losses.recon.detach()), 0.0)
        self.assertGreaterEqual(float(losses.kl.detach()), 0.0)
        losses.total.backward()  # must not raise
        # encoder + decoder params should receive gradient
        self.assertIsNotNone(model.decoder.net[0].weight.grad)

    def test_reward_decoder_can_learn_separable_case(self) -> None:
        # Sanity: the decoder MLP can fit a trivial "this cell is the goal" signal.
        from varibad_gridworld.models.varibad import RewardDecoder
        dec = RewardDecoder()
        opt = torch.optim.Adam(dec.parameters(), lr=0.01)
        s = torch.zeros(2, 25)
        a = torch.zeros(2, 5)
        s_next = torch.zeros(2, 25); s_next[0, 7] = 1.0; s_next[1, 12] = 1.0
        m = torch.zeros(2, 5)
        labels = torch.tensor([1.0, 0.0])
        first = None
        for _ in range(200):
            opt.zero_grad()
            logit = dec(s, a, s_next, m).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, labels)
            if first is None:
                first = float(loss.detach())
            loss.backward(); opt.step()
        self.assertLess(float(loss.detach()), first)
