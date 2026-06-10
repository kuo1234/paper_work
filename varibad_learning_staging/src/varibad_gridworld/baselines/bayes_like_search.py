from __future__ import annotations

from random import Random

from ..envs.gridworld import GridWorldTask
from ..utils.belief import GoalBelief
from ._common import action_toward

Cell = tuple[int, int]

# Fixed boustrophedon (snake) sweep over the 15 allowed goal cells
# (spec section 5.4). It starts at row 2 -- the allowed row NEAREST the start
# (4,0) -- and snakes upward, so the agent checks close cells first.
DEFAULT_SEARCH_ORDER: list[Cell] = [
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
    (1, 4), (1, 3), (1, 2), (1, 1), (1, 0),
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
]


class BayesLikeSearchPolicy:
    """Systematic sweep along a fixed order, then exploit (spec section 5.4).

    Not strictly Bayes-optimal planning, but a deterministic systematic search:
    walk through the candidate cells along a pre-set snake path, skipping any the
    belief has already ruled out, until the goal is found -- then walk onto it and
    STAY. After an episode reset, go straight back to the known goal.

    Because it never wastes moves re-checking ruled-out cells, it is far more
    step-efficient than posterior sampling, and is the reference closest to
    Bayes-optimal exploration.
    """

    name = "Bayes-like Search"

    def __init__(self, search_order: list[Cell] | None = None) -> None:
        self._order = search_order if search_order is not None else DEFAULT_SEARCH_ORDER
        self._grid_size = 5

    def reset_task(self, env: GridWorldTask) -> None:
        self._grid_size = env.grid_size

    def act(self, state: Cell, belief: GoalBelief, rng: Random) -> int:
        if belief.known_goal is not None:
            return action_toward(state, belief.known_goal, self._grid_size, rng)

        candidates = set(belief.candidate_cells())
        # Next cell on the fixed sweep that hasn't been ruled out yet.
        for cell in self._order:
            if cell in candidates:
                return action_toward(state, cell, self._grid_size, rng)
        return action_toward(state, state, self._grid_size, rng)  # nothing left
