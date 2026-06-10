from __future__ import annotations

from dataclasses import dataclass

import torch

from ..utils.torch_rollout import RolloutBatch


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """G_t = r_t + gamma r_{t+1} + ...  (computed backward over the full rollout).

    We treat the BAMDP rollout as a single horizon (H+ steps): the agent should
    value reward earned in later episodes, which is what makes it explore early.
    """

    G = torch.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


@dataclass
class A2CLosses:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


def a2c_losses(
    batch: RolloutBatch,
    gamma: float = 0.95,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> A2CLosses:
    """Standard A2C objective (spec sections 7.4 & 9.2).

    L = L_policy + value_coef * L_value - entropy_coef * H(pi)
    with advantage A_t = G_t - V_t (value detached in the policy term).
    """

    returns = discounted_returns(batch.rewards, gamma)
    values = batch.values
    advantages = returns - values.detach()

    policy_loss = -(batch.log_probs * advantages).mean()
    value_loss = (returns - values).pow(2).mean()
    entropy = batch.entropies.mean()

    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return A2CLosses(total=total, policy=policy_loss, value=value_loss, entropy=entropy)
