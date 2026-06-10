from __future__ import annotations

from random import Random

from ..envs.gridworld import ACTIONS, STAY

Cell = tuple[int, int]


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def action_toward(state: Cell, target: Cell, grid_size: int, rng: Random) -> int:
    """A shortest-path action toward ``target``; STAY if already there.

    On a wall-free grid any move that reduces Manhattan distance is on a shortest
    path, so we pick uniformly among the improving moves.
    """

    if state == target:
        return STAY
    improving = []
    for i, (d_row, d_col) in enumerate(ACTIONS):
        if i == STAY:
            continue
        nxt = (state[0] + d_row, state[1] + d_col)
        if not (0 <= nxt[0] < grid_size and 0 <= nxt[1] < grid_size):
            continue
        if manhattan(nxt, target) < manhattan(state, target):
            improving.append(i)
    if not improving:
        return STAY
    return improving[rng.randrange(len(improving))]
