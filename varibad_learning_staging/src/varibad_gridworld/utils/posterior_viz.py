from __future__ import annotations

from ..utils.belief import GoalBelief

Cell = tuple[int, int]

# Light-to-dark ramp for probability magnitude in the ASCII heatmap.
_RAMP = " .:-=+*#@"


def entropy_sparkline(values: list[float], max_bits: float | None = None) -> str:
    """One-line bar sparkline of entropy (bits) over time.

    Uses ASCII height characters (console-safe across code pages). A flat run of
    the lowest level after a drop means "goal found, uncertainty = 0".
    """

    levels = "_.,-:=*#"  # 8 ASCII levels, low -> high
    hi = max_bits if max_bits is not None else (max(values) if values else 1.0)
    if hi <= 0:
        return levels[0] * len(values)
    out = []
    for v in values:
        frac = max(0.0, min(1.0, v / hi))
        idx = int(round(frac * (len(levels) - 1)))
        out.append(levels[idx])
    return "".join(out)


def heatmap(belief: GoalBelief, grid_size: int, goal: Cell | None = None) -> str:
    """ASCII 5x5 posterior heatmap.

    Each cell shows a character whose darkness is proportional to the posterior
    probability that the cell is the goal. The true goal (if given) is marked 'G'
    when it still has mass, so you can watch the belief concentrate onto it.
    """

    probs = belief.probs()
    pmax = max(probs.values()) if probs else 1.0
    rows = []
    for r in range(grid_size):
        chars = []
        for c in range(grid_size):
            p = probs.get((r, c), 0.0)
            if goal is not None and (r, c) == goal and p > 0.0:
                ch = "G"
            elif p <= 0.0:
                ch = "."
            else:
                idx = int(round((p / pmax) * (len(_RAMP) - 1)))
                ch = _RAMP[max(1, idx)]
            chars.append(ch)
        rows.append(" ".join(chars))
    return "\n".join(rows)
