from unittest import TestCase

import torch

from varibad_gridworld.models.rl2 import ACTION_DIM, INPUT_DIM, STATE_DIM
from varibad_gridworld.models.varibad import (
    LATENT_DIM,
    RewardDecoder,
    VariBAD,
    VariBADEncoder,
    VariBADPolicy,
    kl_to_standard_normal,
    reparameterize,
)


class VariBADModelTest(TestCase):
    def test_encoder_step_shapes(self) -> None:
        enc = VariBADEncoder()
        h = enc.initial_hidden(1)
        mu, logvar, h2 = enc.step(torch.zeros(1, INPUT_DIM), h)
        self.assertEqual(tuple(mu.shape), (1, LATENT_DIM))
        self.assertEqual(tuple(logvar.shape), (1, LATENT_DIM))
        self.assertEqual(tuple(h2.shape), (1, 1, enc.hidden_dim))

    def test_encoder_forward_sequence(self) -> None:
        enc = VariBADEncoder()
        T, B = 6, 2
        mu, logvar, h = enc(torch.zeros(T, B, INPUT_DIM))
        self.assertEqual(tuple(mu.shape), (T, B, LATENT_DIM))

    def test_policy_shapes(self) -> None:
        pol = VariBADPolicy()
        logits, value = pol(torch.zeros(1, STATE_DIM), torch.zeros(1, LATENT_DIM), torch.zeros(1, LATENT_DIM))
        self.assertEqual(tuple(logits.shape), (1, ACTION_DIM))
        self.assertEqual(tuple(value.shape), (1, 1))

    def test_decoder_outputs_single_logit(self) -> None:
        dec = RewardDecoder()
        logit = dec(torch.zeros(1, STATE_DIM), torch.zeros(1, ACTION_DIM),
                    torch.zeros(1, STATE_DIM), torch.zeros(1, LATENT_DIM))
        self.assertEqual(tuple(logit.shape), (1, 1))

    def test_reparameterize_shape_and_mean(self) -> None:
        mu = torch.full((1000, LATENT_DIM), 2.0)
        logvar = torch.full((1000, LATENT_DIM), -10.0)  # tiny variance
        samples = reparameterize(mu, logvar)
        self.assertEqual(samples.shape, mu.shape)
        # near-zero variance -> samples ~ mu
        self.assertTrue(torch.allclose(samples.mean(), torch.tensor(2.0), atol=0.05))

    def test_kl_zero_at_standard_normal(self) -> None:
        mu = torch.zeros(1, LATENT_DIM)
        logvar = torch.zeros(1, LATENT_DIM)  # var = 1
        kl = kl_to_standard_normal(mu, logvar)
        self.assertAlmostEqual(float(kl.item()), 0.0, places=5)

    def test_kl_positive_when_shifted(self) -> None:
        kl = kl_to_standard_normal(torch.ones(1, LATENT_DIM), torch.zeros(1, LATENT_DIM))
        self.assertGreater(float(kl.item()), 0.0)

    def test_bundle_has_all_three(self) -> None:
        m = VariBAD()
        self.assertIsInstance(m.encoder, VariBADEncoder)
        self.assertIsInstance(m.policy, VariBADPolicy)
        self.assertIsInstance(m.decoder, RewardDecoder)
