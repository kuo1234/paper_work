from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..models.varibad import VariBAD, kl_to_standard_normal, reparameterize
from ..utils.torch_rollout import VariBADBatch


@dataclass
class VAELosses:
    total: torch.Tensor
    recon: torch.Tensor
    kl: torch.Tensor


def default_context_times(H_plus: int) -> list[int]:
    """Context cut points tau_{:t} (spec section 9.3), clamped to the horizon."""

    candidates = [0, 5, 10, 15, 30, 45, 60]
    return sorted({min(t, H_plus) for t in candidates})


def vae_losses(
    model: VariBAD,
    batch: VariBADBatch,
    context_times: list[int] | None = None,
    beta: float = 0.1,
) -> VAELosses:
    """VariBAD ELBO (spec sections 8.6-8.7, 9.3).

    For each context time t: encode tau_{:t} -> (mu_t, logvar_t), sample m_t, then
    have the reward decoder predict the reward of EVERY transition in the whole
    rollout from m_t. Reconstructing the full trajectory's rewards from a partial
    history is what forces the latent to infer the task (goal location).

    L_VAE = reconstruction_BCE + beta * KL(q || N(0, I)).
    """

    device = batch.rewards.device
    T = batch.states.shape[0]
    H_plus = batch.H * batch.N
    if context_times is None:
        context_times = default_context_times(H_plus)

    # Encode the whole input stream once: mu_all[t] is the posterior after x_0..x_t.
    x_seq = batch.inputs.unsqueeze(1)                      # (T, 1, INPUT_DIM)
    mu_all, logvar_all, _ = model.encoder(x_seq)           # (T, 1, L)
    mu_all = mu_all.squeeze(1)                             # (T, L)
    logvar_all = logvar_all.squeeze(1)

    recon_terms = []
    kl_terms = []
    for t in context_times:
        idx = min(max(t - 1, 0), T - 1)                   # posterior after t inputs
        mu_t = mu_all[idx]
        logvar_t = logvar_all[idx]
        m_t = reparameterize(mu_t, logvar_t)              # (L,)

        # Decode rewards for ALL T transitions from this single latent.
        m_rep = m_t.unsqueeze(0).expand(T, -1)            # (T, L)
        logits = model.decoder(batch.states, batch.actions_oh, batch.next_states, m_rep).squeeze(-1)
        recon_terms.append(F.binary_cross_entropy_with_logits(logits, batch.reward_labels))
        kl_terms.append(kl_to_standard_normal(mu_t, logvar_t))

    recon = torch.stack(recon_terms).mean()
    kl = torch.stack(kl_terms).mean()
    total = recon + beta * kl
    return VAELosses(total=total, recon=recon, kl=kl)
