from __future__ import annotations

from random import Random

from ..envs.gridworld import GridWorldTask
from ..utils.belief import GoalBelief
from ._common import action_toward

Cell = tuple[int, int]


class OraclePolicy:
    """Privileged policy that KNOWS the goal (spec section 5.2).

    Walks the shortest path to the goal, then stays. After each episode reset it
    walks straight back. This is the performance UPPER BOUND.

    Important: the Oracle is NOT Bayes-optimal -- it never has to explore, because
    it already knows the task. It only bounds performance from above:
        J(Bayes-optimal) <= J(Oracle).
    """

    name = "Oracle"

    def __init__(self) -> None:
        self._goal: Cell | None = None
        self._grid_size = 5

    def reset_task(self, env: GridWorldTask) -> None:
        self._goal = env.goal
        self._grid_size = env.grid_size

    def act(self, state: Cell, belief: GoalBelief, rng: Random) -> int:
        goal = self._goal if self._goal is not None else state
        return action_toward(state, goal, self._grid_size, rng)
