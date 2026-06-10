from __future__ import annotations

from dataclasses import dataclass

from .rollout import RolloutResult


@dataclass
class AggregateMetrics:
    """Mean metrics over many BAMDP rollouts (spec sections 6 & 18)."""

    n_tasks: int
    avg_return: float
    episode_returns: list[float]          # mean return per episode (ep1..epN)
    avg_first_goal_step: float            # over rollouts that found the goal
    success_rate_first_episode: float     # spec 6.5 (found in episode 0)
    success_rate_rollout: float           # spec 6.5 (found anywhere in H+)
    avg_redundant_visits: float           # spec 6.4


def aggregate(results: list[RolloutResult]) -> AggregateMetrics:
    n = len(results)
    if n == 0:
        raise ValueError("need at least one rollout")

    N = results[0].N
    ep_sums = [0.0] * N
    total_return = 0.0
    first_goal_steps: list[int] = []
    found_first_ep = 0
    found_rollout = 0
    redundant_total = 0

    for r in results:
        total_return += r.total_return
        for i, er in enumerate(r.episode_returns()):
            ep_sums[i] += er
        fgs = r.first_goal_step()
        if fgs is not None:
            first_goal_steps.append(fgs)
        if r.found_within_first_episode():
            found_first_ep += 1
        if r.found_within_rollout():
            found_rollout += 1
        redundant_total += r.redundant_visits()

    avg_fgs = (sum(first_goal_steps) / len(first_goal_steps)) if first_goal_steps else float("nan")

    return AggregateMetrics(
        n_tasks=n,
        avg_return=total_return / n,
        episode_returns=[s / n for s in ep_sums],
        avg_first_goal_step=avg_fgs,
        success_rate_first_episode=found_first_ep / n,
        success_rate_rollout=found_rollout / n,
        avg_redundant_visits=redundant_total / n,
    )
