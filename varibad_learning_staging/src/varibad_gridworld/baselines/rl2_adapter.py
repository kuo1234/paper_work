from __future__ import annotations

from random import Random

import torch

from ..envs.gridworld import GridWorldTask
from ..models.rl2 import ACTION_DIM, RL2Policy, build_input
from ..utils.belief import GoalBelief

Cell = tuple[int, int]


class RL2PolicyAdapter:
    """Wrap a trained RL2Policy in the hard-coded-baseline interface.

    This lets a learned RL^2 policy drop into ``run_bamdp_rollout`` and the
    ``run_baselines`` comparison table alongside Random / Oracle / etc. The
    adapter ignores the exact ``belief`` (RL^2 has no explicit posterior); instead
    it carries the GRU hidden state and the previous (action, reward, done) to
    build the network input x_t = [state, prev_action, reward, done].

    The hidden state is created in ``reset_task`` and carried across episodes
    (spec section 17.2). ``observe`` is called by the rollout after each step to
    feed back the realized reward/done.
    """

    name = "RL2"

    def __init__(self, model: RL2Policy, grid_size: int = 5, deterministic: bool = True) -> None:
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
        self._hidden = self.model.initial_hidden(batch_size=1, device=self._device)
        self._prev_action_oh = torch.zeros(ACTION_DIM)
        self._prev_reward = torch.zeros(1)
        self._done = torch.zeros(1)

    def _state_onehot(self, state: Cell) -> torch.Tensor:
        n = self.grid_size * self.grid_size
        vec = torch.zeros(n)
        vec[state[0] * self.grid_size + state[1]] = 1.0
        return vec

    def act(self, state: Cell, belief: GoalBelief, rng: Random) -> int:
        state_oh = self._state_onehot(state)
        x_t = build_input(
            state_oh.unsqueeze(0),
            self._prev_action_oh.unsqueeze(0),
            self._prev_reward.unsqueeze(0),
            self._done.unsqueeze(0),
        ).to(self._device)
        with torch.no_grad():
            logits, _value, self._hidden = self.model.step(
                x_t, state_oh.unsqueeze(0).to(self._device), self._hidden
            )
        logits = logits.squeeze(0)
        if self.deterministic:
            return int(torch.argmax(logits).item())
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def observe(self, action: int, reward: float, episode_done: bool) -> None:
        oh = torch.zeros(ACTION_DIM)
        oh[action] = 1.0
        self._prev_action_oh = oh
        self._prev_reward = torch.tensor([reward], dtype=torch.float32)
        self._done = torch.tensor([1.0 if episode_done else 0.0], dtype=torch.float32)


def load_rl2_adapter(path: str, grid_size: int = 5, deterministic: bool = True) -> RL2PolicyAdapter:
    model = RL2Policy()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return RL2PolicyAdapter(model, grid_size=grid_size, deterministic=deterministic)
