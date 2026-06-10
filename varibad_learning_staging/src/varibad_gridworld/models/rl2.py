from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Observation dimensions (spec section 7.2).
STATE_DIM = 25       # 5x5 one-hot
ACTION_DIM = 5       # up/right/down/left/stay one-hot
INPUT_DIM = STATE_DIM + ACTION_DIM + 1 + 1   # + prev_reward + done = 32
HIDDEN_DIM = 128


def build_input(
    state_onehot: torch.Tensor,
    prev_action_onehot: torch.Tensor,
    prev_reward: torch.Tensor,
    done: torch.Tensor,
) -> torch.Tensor:
    """Assemble x_t = [s_t, a_{t-1}, r_t, d_t]  (spec section 7.2).

    All inputs are shape (batch, *). prev_reward and done are (batch, 1).
    """

    return torch.cat([state_onehot, prev_action_onehot, prev_reward, done], dim=-1)


class RL2Policy(nn.Module):
    """Recurrent meta-RL baseline (RL^2, spec section 7).

    A GRU consumes the stream x_t = [state, prev_action, reward, done] and carries
    a hidden state that does online task inference *implicitly* -- there is no
    explicit posterior. The policy/value heads read the GRU hidden state together
    with the current state one-hot.

    Crucially (spec section 17.2) the hidden state is carried across episode
    boundaries within a task; only a new task resets it. The ``done`` flag in the
    input tells the GRU when an episode boundary occurs without wiping memory.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=INPUT_DIM, hidden_size=hidden_dim, batch_first=False)
        head_in = hidden_dim + STATE_DIM
        self.policy_head = nn.Linear(head_in, ACTION_DIM)
        self.value_head = nn.Linear(head_in, 1)

    def initial_hidden(self, batch_size: int = 1, device=None) -> torch.Tensor:
        """Zero hidden state, shape (1, batch, hidden) for nn.GRU."""

        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def step(
        self,
        x_t: torch.Tensor,
        state_onehot: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single timestep forward.

        x_t:          (batch, INPUT_DIM)
        state_onehot: (batch, STATE_DIM)   -- current state, for the heads
        hidden:       (1, batch, hidden)

        Returns (logits (batch, ACTION_DIM), value (batch, 1), new_hidden).
        """

        # GRU expects (seq=1, batch, input).
        out, new_hidden = self.gru(x_t.unsqueeze(0), hidden)
        h = out.squeeze(0)                       # (batch, hidden)
        head_in = torch.cat([h, state_onehot], dim=-1)
        logits = self.policy_head(head_in)
        value = self.value_head(head_in)
        return logits, value, new_hidden

    def forward(
        self,
        x_seq: torch.Tensor,
        state_seq: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Whole-sequence forward (for batched VAE-style processing / testing).

        x_seq:     (T, batch, INPUT_DIM)
        state_seq: (T, batch, STATE_DIM)
        Returns (logits (T, batch, ACTION_DIM), values (T, batch, 1), hidden).
        """

        if hidden is None:
            hidden = self.initial_hidden(x_seq.shape[1], device=x_seq.device)
        out, hidden = self.gru(x_seq, hidden)        # (T, batch, hidden)
        head_in = torch.cat([out, state_seq], dim=-1)
        logits = self.policy_head(head_in)
        values = self.value_head(head_in)
        return logits, values, hidden
