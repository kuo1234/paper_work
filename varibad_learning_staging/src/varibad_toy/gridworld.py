from __future__ import annotations

from dataclasses import dataclass
from random import Random

Cell = tuple[int, int]

# Actions are (d_row, d_col). Order matters only for tie-breaking display.
ACTIONS: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES: tuple[str, ...] = ("up", "down", "left", "right")


@dataclass(frozen=True)
class HiddenGoalGrid:
    """A square GridWorld where one hidden cell is the goal.

    The agent only learns where the goal is by stepping onto it. This is the
    spatial analogue of ``HiddenGoalBandit``: the task (goal location) is hidden,
    and the only way to reduce uncertainty is to *go look*. That is exactly what
    makes Bayes-optimal exploration visible here -- the agent must physically
    sweep cells it has not ruled out.
    """

    size: int
    goal_cell: Cell
    start_cell: Cell = (0, 0)

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("size must be >= 1")
        if not self.in_bounds(self.goal_cell):
            raise ValueError("goal_cell out of bounds")
        if not self.in_bounds(self.start_cell):
            raise ValueError("start_cell out of bounds")

    @classmethod
    def sample_task(
        cls,
        rng: Random,
        size: int = 5,
        start_cell: Cell = (0, 0),
        exclude_start: bool = True,
    ) -> "HiddenGoalGrid":
        """Sample a task by placing the goal uniformly at random.

        With ``exclude_start`` the goal is never the start cell, so every episode
        requires at least one real move to find it.
        """

        while True:
            goal = (rng.randrange(size), rng.randrange(size))
            if exclude_start and goal == start_cell:
                continue
            return cls(size=size, goal_cell=goal, start_cell=start_cell)

    def in_bounds(self, cell: Cell) -> bool:
        row, col = cell
        return 0 <= row < self.size and 0 <= col < self.size

    def cells(self) -> list[Cell]:
        return [(r, c) for r in range(self.size) for c in range(self.size)]

    def step(self, state: Cell, action: int) -> tuple[Cell, int, bool]:
        """Apply an action.

        Returns ``(next_state, reward, done)``. Walking into a wall keeps the
        agent in place. Reward is +1 only on the step that lands on the goal,
        which also ends the episode.
        """

        if action not in range(len(ACTIONS)):
            raise ValueError(f"action must be in 0..{len(ACTIONS) - 1}")

        d_row, d_col = ACTIONS[action]
        candidate = (state[0] + d_row, state[1] + d_col)
        next_state = candidate if self.in_bounds(candidate) else state

        if next_state == self.goal_cell:
            return next_state, 1, True
        return next_state, 0, False
