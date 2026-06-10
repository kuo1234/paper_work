from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rl2 import ACTION_DIM, INPUT_DIM, STATE_DIM  # 5, 32, 25

LATENT_DIM = 5          # d_m (spec section 8.2)
ENCODER_HIDDEN = 64     # GRU hidden (spec section 8.2)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample m = mu + sigma * eps,  eps ~ N(0, I)  (spec section 8.3)."""

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, sigma^2) || N(0, I) ), summed over latent dim (spec section 8.6).

    Returns shape (...) with the latent dimension reduced.
    """

    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


class VariBADEncoder(nn.Module):
    """Variational task encoder q(m | tau_{:t})  (spec section 8.2).

    Consumes the same per-step stream x_t = [state, prev_action, reward, done]
    (32-d) as RL^2, but outputs Gaussian posterior parameters (mu, logvar) over a
    latent task embedding m. The GRU hidden state carries across episodes within a
    task (spec section 17.2); only a new task resets it.
    """

    def __init__(self, hidden_dim: int = ENCODER_HIDDEN, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.fc = nn.Linear(INPUT_DIM, 40)
        self.gru = nn.GRU(input_size=40, hidden_size=hidden_dim, batch_first=False)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def initial_hidden(self, batch_size: int = 1, device=None) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def step(self, x_t: torch.Tensor, hidden: torch.Tensor):
        """Single timestep. x_t: (batch, INPUT_DIM); hidden: (1, batch, hidden).

        Returns (mu (batch, L), logvar (batch, L), new_hidden).
        """

        z = F.relu(self.fc(x_t))
        out, new_hidden = self.gru(z.unsqueeze(0), hidden)
        h = out.squeeze(0)
        return self.mu_head(h), self.logvar_head(h), new_hidden

    def forward(self, x_seq: torch.Tensor, hidden: torch.Tensor | None = None):
        """Whole-sequence. x_seq: (T, batch, INPUT_DIM).

        Returns (mu (T, batch, L), logvar (T, batch, L), hidden). mu[t] is the
        posterior after consuming inputs x_0..x_t.
        """

        if hidden is None:
            hidden = self.initial_hidden(x_seq.shape[1], device=x_seq.device)
        z = F.relu(self.fc(x_seq))
        out, hidden = self.gru(z, hidden)
        return self.mu_head(out), self.logvar_head(out), hidden


class VariBADPolicy(nn.Module):
    """Belief-conditioned policy pi(a | s, mu, logvar)  (spec section 8.4, Design A).

    The policy reads the posterior PARAMETERS (mu, logvar), not a sampled latent,
    so it is explicitly uncertainty-aware: it can behave differently when the task
    posterior is broad (explore) vs sharp (exploit).
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        input_dim = STATE_DIM + latent_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
        )
        self.policy_head = nn.Linear(32, ACTION_DIM)
        self.value_head = nn.Linear(32, 1)

    def forward(self, state_onehot: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        x = torch.cat([state_onehot, mu, logvar], dim=-1)
        h = self.net(x)
        return self.policy_head(h), self.value_head(h)


class RewardDecoder(nn.Module):
    """Predict P(reward = +1) for a transition, conditioned on latent m.

    Input [s, a, s', m] (spec section 8.5). In GridWorld reward is binary
    (goal / non-goal), so this is a binary classifier trained with BCE. Training
    it forces the latent m to actually carry task (goal-location) information.
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        input_dim = STATE_DIM + ACTION_DIM + STATE_DIM + latent_dim  # 25+5+25+5 = 60
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor, m: torch.Tensor):
        x = torch.cat([s, a, s_next, m], dim=-1)
        return self.net(x)  # logit


class VariBAD(nn.Module):
    """Bundle encoder + policy + decoder so one optimizer/state_dict covers all."""

    def __init__(self, latent_dim: int = LATENT_DIM, encoder_hidden: int = ENCODER_HIDDEN) -> None:
        super().__init__()
        self.encoder = VariBADEncoder(hidden_dim=encoder_hidden, latent_dim=latent_dim)
        self.policy = VariBADPolicy(latent_dim=latent_dim)
        self.decoder = RewardDecoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim
