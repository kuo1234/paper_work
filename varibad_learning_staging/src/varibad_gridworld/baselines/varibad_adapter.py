from __future__ import annotations

from random import Random

import torch

from ..envs.gridworld import GridWorldTask
from ..models.rl2 import ACTION_DIM, build_input
from ..models.varibad import VariBAD
from ..utils.belief import GoalBelief

Cell = tuple[int, int]


class VariBADPolicyAdapter:
    """Wrap a trained VariBAD model in the hard-coded-baseline interface.

    Lets the learned VariBAD policy drop into ``run_bamdp_rollout`` and the
    comparison table. The adapter runs the encoder each step to get (mu, logvar),
    feeds the policy on (state, mu, logvar), and maintains the encoder hidden
    state across episodes (spec 17.2). The exact ``belief`` argument is ignored --
    VariBAD uses its own *learned* posterior.
    """

    name = "VariBAD"

    def __init__(self, model: VariBAD, grid_size: int = 5, deterministic: bool = True) -> None:
        self.model = model
        self.grid_size = grid_size
        self.deterministic = deterministic
        self._device = next(model.parameters()).device
        self._hidden = None
        self._prev_action_oh = torch.zeros(ACTION_DIM)
        self._prev_reward = torch.zeros(1)
        self._done = torch.zeros(1)

    def reset_task(self, env: GridWorldTask) -> None:
        self.grid_size = env.grid_size
        self._hidden = self.model.encoder.initial_hidden(batch_size=1, device=self._device)
        self._prev_action_oh = torch.zeros(ACTION_DIM, device=self._device)
        self._prev_reward = torch.zeros(1, device=self._device)
        self._done = torch.zeros(1, device=self._device)

    def _state_onehot(self, state: Cell) -> torch.Tensor:
        n = self.grid_size * self.grid_size
        vec = torch.zeros(n, device=self._device)
        vec[state[0] * self.grid_size + state[1]] = 1.0
        return vec

    def act(self, state: Cell, belief: GoalBelief, rng: Random) -> int:
        state_oh = self._state_onehot(state)
        x_t = build_input(
            state_oh.unsqueeze(0), self._prev_action_oh.unsqueeze(0),
            self._prev_reward.unsqueeze(0), self._done.unsqueeze(0),
        )
        with torch.no_grad():
            mu, logvar, self._hidden = self.model.encoder.step(x_t, self._hidden)
            logits, _value = self.model.policy(state_oh.unsqueeze(0), mu, logvar)
        logits = logits.squeeze(0)
        if self.deterministic:
            return int(torch.argmax(logits).item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())

    def observe(self, action: int, reward: float, episode_done: bool) -> None:
        oh = torch.zeros(ACTION_DIM, device=self._device)
        oh[action] = 1.0
        self._prev_action_oh = oh
        self._prev_reward = torch.tensor([reward], dtype=torch.float32, device=self._device)
        self._done = torch.tensor([1.0 if episode_done else 0.0], dtype=torch.float32, device=self._device)


def load_varibad_adapter(path: str, grid_size: int = 5, deterministic: bool = True) -> VariBADPolicyAdapter:
    model = VariBAD()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return VariBADPolicyAdapter(model, grid_size=grid_size, deterministic=deterministic)
