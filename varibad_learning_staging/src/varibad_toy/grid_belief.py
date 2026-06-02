from __future__ import annotations

from dataclasses import dataclass, field
from math import log2

Cell = tuple[int, int]


@dataclass
class GridBeliefEncoder:
    """Exact Bayesian belief over the hidden goal location.

    VARIBAD learns an approximate variational encoder ``q(m | trajectory)``. In
    this toy GridWorld the task ``m`` is just *which cell is the goal*, with a
    uniform prior, and the observation model is deterministic: stepping on a
    non-goal cell rules that cell out, stepping on the goal identifies it.

    So the exact posterior is dead simple -- it stays uniform over the cells we
    have not yet visited. We keep it as an explicit probability map so it reads
    like the belief a learned encoder would produce, and so ``latent()`` can hand
    a fixed-length vector to a future neural policy.
    """

    size: int
    start_cell: Cell = (0, 0)
    # Cells the agent has stepped on and found NOT to be the goal.
    ruled_out: set[Cell] = field(default_factory=set)
    # Set once the goal is actually found.
    known_goal: Cell | None = None

    def __post_init__(self) -> None:
        # The start cell is observed for free at t=0: if the goal were here the
        # episode would already be over, so (when goal != start) it is ruled out.
        self.ruled_out.add(self.start_cell)

    def _all_cells(self) -> list[Cell]:
        return [(r, c) for r in range(self.size) for c in range(self.size)]

    def candidate_cells(self) -> list[Cell]:
        """Cells that could still be the goal, given everything observed."""

        if self.known_goal is not None:
            return [self.known_goal]
        return [cell for cell in self._all_cells() if cell not in self.ruled_out]

    def update(self, visited_cell: Cell, was_goal: bool) -> None:
        """Bayesian update after stepping onto ``visited_cell``."""

        if was_goal:
            self.known_goal = visited_cell
        else:
            self.ruled_out.add(visited_cell)

    def probs(self) -> dict[Cell, float]:
        """Posterior probability that each cell is the goal."""

        if self.known_goal is not None:
            return {self.known_goal: 1.0}
        candidates = self.candidate_cells()
        if not candidates:
            return {}
        p = 1.0 / len(candidates)
        return {cell: p for cell in candidates}

    def entropy_bits(self) -> float:
        """Uncertainty over the goal location, in bits.

        Uniform over N candidates -> log2(N). Drops as cells are ruled out, and
        hits 0 the moment the goal is found. This is the curve to watch: a good
        explorer drives it down fast.
        """

        probs = self.probs().values()
        return sum(-p * log2(p) for p in probs if p > 0.0)

    def latent(self) -> list[float]:
        """Fixed-length belief vector, analogous to a VARIBAD latent.

        Layout: flattened size*size posterior map, followed by entropy in bits.
        This is exactly the kind of input a learned policy would condition on, so
        a future neural policy can consume ``latent()`` unchanged.
        """

        probs = self.probs()
        flat = [probs.get((r, c), 0.0) for r in range(self.size) for c in range(self.size)]
        flat.append(self.entropy_bits())
        return flat

    def copy(self) -> "GridBeliefEncoder":
        return GridBeliefEncoder(
            size=self.size,
            start_cell=self.start_cell,
            ruled_out=set(self.ruled_out),
            known_goal=self.known_goal,
        )
