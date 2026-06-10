from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from ..envs.gridworld import GridWorldTask
from ..utils.belief import GoalBelief

Cell = tuple[int, int]


@dataclass
class StepRecord:
    t_global: int
    episode: int
    t_in_episode: int
    state: Cell
    action: int
    reward: float
    on_goal: bool
    entropy_bits: float
    n_candidates: int


@dataclass
class RolloutResult:
    """One BAMDP rollout over a single fixed task.

    H  = steps per episode, N = episodes, H+ = N*H = BAMDP horizon.
    Belief persists across episodes; only position resets each episode.
    """

    goal: Cell
    H: int
    N: int
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def H_plus(self) -> int:
        return self.N * self.H

    @property
    def total_return(self) -> float:
        return sum(s.reward for s in self.steps)

    def episode_returns(self) -> list[float]:
        sums = [0.0] * self.N
        for s in self.steps:
            sums[s.episode] += s.reward
        return sums

    def first_goal_step(self) -> int | None:
        """Global step index at which the goal was first reached (spec 6.3)."""

        for s in self.steps:
            if s.on_goal:
                return s.t_global
        return None

    def found_within_first_episode(self) -> bool:
        return any(s.on_goal and s.episode == 0 for s in self.steps)

    def found_within_rollout(self) -> bool:
        return any(s.on_goal for s in self.steps)

    def redundant_visits(self) -> int:
        """Visits to known non-goal cells DURING EXPLORATION (spec 6.4 / 12.3).

        Counts re-visits to cells already stepped on (and found non-goal), but
        only up to the moment the goal is first found. After that the agent is
        exploiting -- re-walking the optimal path each episode is necessary, not
        wasteful, so those steps must not count. This isolates exploration
        inefficiency, which is exactly the Posterior-vs-Bayes-search gap the spec
        cares about.
        """

        known_non_goal: set[Cell] = set()
        redundant = 0
        for s in self.steps:
            if s.on_goal:
                break  # exploration ends once the goal is found
            if s.state in known_non_goal:
                redundant += 1
            known_non_goal.add(s.state)
        return redundant


def run_bamdp_rollout(
    env: GridWorldTask,
    policy,
    N: int,
    rng: Random,
    task_rng: Random | None = None,
) -> RolloutResult:
    """Run N episodes of H steps on ONE freshly sampled task.

    The belief is created once and persists across all N episodes. ``env.step``
    auto-resets the position at each episode boundary; we mirror that boundary to
    call the policy's ``reset_episode`` hook so stateful policies (e.g. posterior
    sampling) can draw a fresh sample.
    """

    sampler = task_rng if task_rng is not None else rng
    env.reset_task(sampler)
    belief = GoalBelief(env.allowed_goals)

    if hasattr(policy, "reset_task"):
        policy.reset_task(env)

    result = RolloutResult(goal=env.goal, H=env.H, N=N)
    state = env.state

    for episode in range(N):
        if hasattr(policy, "reset_episode"):
            policy.reset_episode()
        for t in range(env.H):
            action = policy.act(state, belief, rng)
            next_state, reward, episode_done, info = env.step(action)
            belief.update(next_state, was_goal=info["on_goal"])

            # Optional hook for stateful neural policies (e.g. RL2) that need the
            # reward/done of the action they just took to build their next input.
            # Hard-coded baselines don't define this, so they are unaffected.
            if hasattr(policy, "observe"):
                policy.observe(action, reward, episode_done)

            result.steps.append(
                StepRecord(
                    t_global=episode * env.H + t,
                    episode=episode,
                    t_in_episode=t,
                    state=next_state,
                    action=action,
                    reward=reward,
                    on_goal=info["on_goal"],
                    entropy_bits=belief.entropy_bits(),
                    n_candidates=len(belief.candidate_cells()),
                )
            )
            # env auto-resets position on episode_done; track the new state.
            state = env.state

    return result
