from __future__ import annotations

from random import Random

from ..envs.gridworld import N_ACTIONS
from ..utils.belief import GoalBelief

Cell = tuple[int, int]


class RandomPolicy:
    """Move uniformly at random over all 5 actions (spec section 5.1).

    The lowest baseline: it ignores the belief entirely, so it has no structure
    and frequently fails to reach the goal.
    """

    name = "Random"

    def act(self, state: Cell, belief: GoalBelief, rng: Random) -> int:
        return rng.randrange(N_ACTIONS)
