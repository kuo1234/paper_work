from __future__ import annotations

from random import Random
from typing import Protocol

from .grid_belief import GridBeliefEncoder
from .gridworld import ACTIONS, Cell


class GridPolicy(Protocol):
    """A policy that maps (position, belief) to a grid action."""

    def act(self, state: Cell, belief: GridBeliefEncoder, rng: Random) -> int:
        ...


class BeliefConditionedPolicy(Protocol):
    """Interface a future *neural* VARIBAD policy can implement.

    The exact policies below read the structured ``belief`` object directly. A
    learned policy instead consumes ``belief.latent()`` -- a fixed-length vector.
    Implement this Protocol (e.g. with a small MLP) and you can drop it straight
    into ``run_grid_episode`` to measure how close the *learned* policy gets to
    the Bayes-optimal one.
    """

    def act_from_latent(self, state: Cell, latent: list[float], rng: Random) -> int:
        ...


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _action_toward(state: Cell, target: Cell, rng: Random) -> int:
    """Return an action index that strictly reduces Manhattan distance to target.

    On a wall-free grid every step toward the target is on a shortest path, so we
    pick uniformly among the moves that close the gap. If already on the target,
    any action is fine (caller normally avoids this case).
    """

    improving = [
        i
        for i, (d_row, d_col) in enumerate(ACTIONS)
        if _manhattan((state[0] + d_row, state[1] + d_col), target) < _manhattan(state, target)
    ]
    if not improving:
        return rng.randrange(len(ACTIONS))
    return improving[rng.randrange(len(improving))]


class BayesOptimalGridPolicy:
    """Bayes-optimal exploration for the hidden-goal GridWorld.

    The reasoning, made concrete:

    * The goal is uniform over the cells we have not yet ruled out (that *is* the
      posterior -- see ``GridBeliefEncoder``).
    * The only way to learn anything is to step on a candidate cell.
    * Expected steps-to-goal is minimized by visiting candidates along a short
      route. With a uniform posterior, repeatedly heading to the *nearest*
      remaining candidate is the canonical Bayes-optimal sweep: each move both
      makes progress and gathers information, and no candidate is wastefully
      skipped.

    Once the goal is found, belief collapses and the policy just walks to it --
    pure exploitation. So this single policy contains the whole explore->exploit
    arc, and serves as the *lower bound* (oracle) the other policies are scored
    against.
    """

    def act(self, state: Cell, belief: GridBeliefEncoder, rng: Random) -> int:
        candidates = belief.candidate_cells()
        if not candidates:
            return rng.randrange(len(ACTIONS))
        # Nearest candidate; deterministic tie-break by position keeps the sweep
        # systematic and reproducible across seeds.
        target = min(candidates, key=lambda cell: (_manhattan(state, cell), cell))
        return _action_toward(state, target, rng)


class GreedyGridPolicy:
    """Walk toward the single most-likely cell, with no drive to explore.

    Under a uniform posterior there is no unique most-likely cell, so this just
    fixes on one candidate (lowest-index tie-break) and heads there, ignoring
    that nearer cells are equally likely and cheaper to check. It still finds the
    goal eventually, but its route is not information-efficient -- the gap to the
    Bayes-optimal policy is the cost of *not* exploring deliberately.
    """

    def act(self, state: Cell, belief: GridBeliefEncoder, rng: Random) -> int:
        candidates = belief.candidate_cells()
        if not candidates:
            return rng.randrange(len(ACTIONS))
        target = min(candidates)  # fixed corner-ward target, not nearest
        return _action_toward(state, target, rng)


class RandomWalkPolicy:
    """Move uniformly at random. The "no belief at all" baseline (worst case)."""

    def act(self, state: Cell, belief: GridBeliefEncoder, rng: Random) -> int:
        return rng.randrange(len(ACTIONS))
