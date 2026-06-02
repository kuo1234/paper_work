from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .grid_belief import GridBeliefEncoder
from .grid_policies import GridPolicy
from .gridworld import Cell, HiddenGoalGrid


@dataclass(frozen=True)
class GridEpisodeStep:
    t: int
    state: Cell
    action: int
    reward: int
    entropy_bits: float
    n_candidates: int
    found_goal: bool


@dataclass(frozen=True)
class GridEpisodeResult:
    goal_cell: Cell
    start_cell: Cell
    steps: list[GridEpisodeStep]
    found: bool

    @property
    def steps_to_goal(self) -> int | None:
        """How many actions until the goal was stepped on (None if never)."""

        for step in self.steps:
            if step.found_goal:
                return step.t + 1
        return None

    @property
    def oracle_steps(self) -> int:
        """Steps an agent that *already knew* the goal would need.

        On a wall-free grid this is the Manhattan distance from start to goal --
        the absolute floor. ``steps_to_goal - oracle_steps`` is the price paid for
        not knowing where the goal was, i.e. the cost of exploration.
        """

        return abs(self.start_cell[0] - self.goal_cell[0]) + abs(
            self.start_cell[1] - self.goal_cell[1]
        )


def run_grid_episode(
    env: HiddenGoalGrid,
    policy: GridPolicy,
    max_steps: int,
    rng: Random,
) -> GridEpisodeResult:
    belief = GridBeliefEncoder(size=env.size, start_cell=env.start_cell)
    state = env.start_cell
    steps: list[GridEpisodeStep] = []
    found = False

    for t in range(max_steps):
        action = policy.act(state, belief, rng)
        next_state, reward, done = env.step(state, action)
        belief.update(next_state, was_goal=done)

        steps.append(
            GridEpisodeStep(
                t=t,
                state=next_state,
                action=action,
                reward=reward,
                entropy_bits=belief.entropy_bits(),
                n_candidates=len(belief.candidate_cells()),
                found_goal=done,
            )
        )
        state = next_state
        if done:
            found = True
            break

    return GridEpisodeResult(
        goal_cell=env.goal_cell,
        start_cell=env.start_cell,
        steps=steps,
        found=found,
    )
