from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

import torch

from ..envs.gridworld import GridWorldTask
from ..models.rl2 import ACTION_DIM, STATE_DIM, build_input

Cell = tuple[int, int]


def onehot(idx: int, dim: int) -> torch.Tensor:
    v = torch.zeros(dim)
    v[idx] = 1.0
    return v


@dataclass
class RolloutBatch:
    """Tensors for one BAMDP rollout (length T = H+), for A2C."""

    log_probs: torch.Tensor          # (T,)
    values: torch.Tensor             # (T,)
    rewards: torch.Tensor            # (T,)
    entropies: torch.Tensor          # (T,)
    actions: list[int] = field(default_factory=list)
    on_goal_steps: list[bool] = field(default_factory=list)
    episodes: list[int] = field(default_factory=list)
    H: int = 15
    N: int = 4

    @property
    def total_return(self) -> float:
        return float(self.rewards.sum())

    def episode_returns(self) -> list[float]:
        sums = [0.0] * self.N
        for ep, r in zip(self.episodes, self.rewards.tolist()):
            sums[ep] += r
        return sums

    def first_goal_step(self) -> int | None:
        for i, g in enumerate(self.on_goal_steps):
            if g:
                return i
        return None


def collect_rollout(
    env: GridWorldTask,
    model,
    N: int,
    rng: Random,
    task_rng: Random,
    deterministic: bool = False,
) -> RolloutBatch:
    """Run one BAMDP rollout (N episodes x H steps) with an RL2 model.

    The GRU hidden state is created once and carried through ALL episodes (spec
    section 17.2); only this fresh task resets it. The per-step input is
    x_t = [state, prev_action, reward, done] (spec section 7.2); at the very first
    step prev_action/prev_reward/done are zero.
    """

    env.reset_task(task_rng)
    H = env.H
    device = next(model.parameters()).device
    hidden = model.initial_hidden(batch_size=1, device=device)

    prev_action_oh = torch.zeros(ACTION_DIM)
    prev_reward = torch.zeros(1)
    done_flag = torch.zeros(1)

    log_probs, values, rewards, entropies = [], [], [], []
    actions, on_goal_steps, episodes = [], [], []

    state = env.state
    for episode in range(N):
        for t in range(H):
            state_oh = torch.tensor(env.get_state_onehot(state), dtype=torch.float32)
            x_t = build_input(
                state_oh.unsqueeze(0),
                prev_action_oh.unsqueeze(0),
                prev_reward.unsqueeze(0),
                done_flag.unsqueeze(0),
            ).to(device)

            logits, value, hidden = model.step(x_t, state_oh.unsqueeze(0).to(device), hidden)
            dist = torch.distributions.Categorical(logits=logits.squeeze(0))
            action = int(torch.argmax(logits.squeeze(0)).item()) if deterministic else int(dist.sample().item())

            next_state, reward, episode_done, info = env.step(action)

            log_probs.append(dist.log_prob(torch.tensor(action, device=device)))
            values.append(value.squeeze())
            rewards.append(reward)
            entropies.append(dist.entropy())
            actions.append(action)
            on_goal_steps.append(bool(info["on_goal"]))
            episodes.append(episode)

            # Prepare inputs for the NEXT step.
            prev_action_oh = onehot(action, ACTION_DIM)
            prev_reward = torch.tensor([reward], dtype=torch.float32)
            done_flag = torch.tensor([1.0 if episode_done else 0.0], dtype=torch.float32)
            state = env.state  # env auto-resets position on episode_done

    return RolloutBatch(
        log_probs=torch.stack(log_probs),
        values=torch.stack(values),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        entropies=torch.stack(entropies),
        actions=actions,
        on_goal_steps=on_goal_steps,
        episodes=episodes,
        H=H,
        N=N,
    )


@dataclass
class VariBADBatch:
    """One BAMDP rollout collected with a VariBAD model.

    Extends the A2C tensors with the full transition sequence and the per-step
    posterior parameters, so the same rollout feeds BOTH the A2C policy update
    and the VAE (reward-decoder) update.
    """

    # A2C tensors (grad-bearing)
    log_probs: torch.Tensor          # (T,)
    values: torch.Tensor             # (T,)
    rewards: torch.Tensor            # (T,)
    entropies: torch.Tensor          # (T,)
    # Transition sequence for the decoder (detached, one-hot / scalar)
    states: torch.Tensor             # (T, STATE_DIM)
    actions_oh: torch.Tensor         # (T, ACTION_DIM)
    next_states: torch.Tensor        # (T, STATE_DIM)
    reward_labels: torch.Tensor      # (T,) in {0,1}: 1 iff reward == +1 (goal)
    inputs: torch.Tensor             # (T, INPUT_DIM): the x_t stream for re-encoding
    # Bookkeeping
    actions: list[int] = field(default_factory=list)
    on_goal_steps: list[bool] = field(default_factory=list)
    episodes: list[int] = field(default_factory=list)
    H: int = 15
    N: int = 4

    @property
    def total_return(self) -> float:
        return float(self.rewards.sum())

    def episode_returns(self) -> list[float]:
        sums = [0.0] * self.N
        for ep, r in zip(self.episodes, self.rewards.tolist()):
            sums[ep] += r
        return sums

    def first_goal_step(self) -> int | None:
        for i, g in enumerate(self.on_goal_steps):
            if g:
                return i
        return None


def collect_varibad_rollout(
    env: GridWorldTask,
    model,
    N: int,
    rng: Random,
    task_rng: Random,
    deterministic: bool = False,
) -> VariBADBatch:
    """Run one BAMDP rollout with a VariBAD model (encoder + policy).

    Each step: (1) the encoder consumes x_t and updates its posterior (mu, logvar)
    -- its hidden state carries across episodes (spec 17.2); (2) the policy acts on
    (state, mu, logvar). We record the transition (s, a, s') and the reward label
    so the VAE update can reconstruct rewards from sampled latents later.
    """

    env.reset_task(task_rng)
    H = env.H
    device = next(model.parameters()).device
    enc_hidden = model.encoder.initial_hidden(batch_size=1, device=device)

    prev_action_oh = torch.zeros(ACTION_DIM, device=device)
    prev_reward = torch.zeros(1, device=device)
    done_flag = torch.zeros(1, device=device)

    log_probs, values, rewards, entropies = [], [], [], []
    states, actions_oh, next_states, reward_labels, inputs = [], [], [], [], []
    actions, on_goal_steps, episodes = [], [], []

    state = env.state
    for episode in range(N):
        for t in range(H):
            state_oh = torch.tensor(env.get_state_onehot(state), dtype=torch.float32, device=device)
            x_t = build_input(
                state_oh.unsqueeze(0),
                prev_action_oh.unsqueeze(0),
                prev_reward.unsqueeze(0),
                done_flag.unsqueeze(0),
            )

            mu, logvar, enc_hidden = model.encoder.step(x_t, enc_hidden)
            logits, value = model.policy(state_oh.unsqueeze(0), mu, logvar)
            dist = torch.distributions.Categorical(logits=logits.squeeze(0))
            action = int(torch.argmax(logits.squeeze(0)).item()) if deterministic else int(dist.sample().item())

            next_state, reward, episode_done, info = env.step(action)
            next_oh = torch.tensor(env.get_state_onehot(next_state), dtype=torch.float32, device=device)
            a_oh = torch.zeros(ACTION_DIM, device=device)
            a_oh[action] = 1.0

            log_probs.append(dist.log_prob(torch.tensor(action, device=device)))
            values.append(value.squeeze())
            rewards.append(reward)
            entropies.append(dist.entropy())
            states.append(state_oh.detach())
            actions_oh.append(a_oh.detach())
            next_states.append(next_oh.detach())
            reward_labels.append(1.0 if info["on_goal"] else 0.0)
            inputs.append(x_t.squeeze(0).detach())
            actions.append(action)
            on_goal_steps.append(bool(info["on_goal"]))
            episodes.append(episode)

            prev_action_oh = a_oh.detach()
            prev_reward = torch.tensor([reward], dtype=torch.float32, device=device)
            done_flag = torch.tensor([1.0 if episode_done else 0.0], dtype=torch.float32, device=device)
            state = env.state

    return VariBADBatch(
        log_probs=torch.stack(log_probs),
        values=torch.stack(values),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        entropies=torch.stack(entropies),
        states=torch.stack(states),
        actions_oh=torch.stack(actions_oh),
        next_states=torch.stack(next_states),
        reward_labels=torch.tensor(reward_labels, dtype=torch.float32, device=device),
        inputs=torch.stack(inputs),
        actions=actions,
        on_goal_steps=on_goal_steps,
        episodes=episodes,
        H=H,
        N=N,
    )
