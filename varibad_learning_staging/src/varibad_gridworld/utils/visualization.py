from __future__ import annotations

from .rollout import RolloutResult

Cell = tuple[int, int]


def render_grid(
    grid_size: int,
    start: Cell,
    goal: Cell,
    visited: set[Cell],
    pos: Cell,
) -> str:
    """ASCII view of one moment (spec section 11.1).

    Legend:  S=start  G=goal  *=agent  .=visited  _=unseen
    """

    rows = []
    for r in range(grid_size):
        cells = []
        for c in range(grid_size):
            cell = (r, c)
            if cell == pos:
                ch = "*"
            elif cell == goal:
                ch = "G"
            elif cell == start:
                ch = "S"
            elif cell in visited:
                ch = "."
            else:
                ch = "_"
            cells.append(ch)
        rows.append(" ".join(cells))
    return "\n".join(rows)


def render_rollout(result: RolloutResult, grid_size: int, start: Cell, indent: str = "    ") -> str:
    """Per-episode ASCII maps + return/first-goal for one BAMDP rollout."""

    lines: list[str] = []
    lines.append(
        f"goal={result.goal}  start={start}  "
        f"H={result.H}  N={result.N}  H+={result.H_plus}"
    )
    for episode in range(result.N):
        ep_steps = [s for s in result.steps if s.episode == episode]
        visited = {start} | {s.state for s in ep_steps}
        final_pos = ep_steps[-1].state if ep_steps else start
        ep_return = sum(s.reward for s in ep_steps)
        first = next((s.t_in_episode for s in ep_steps if s.on_goal), None)
        first_str = f"reached goal at t={first}" if first is not None else "goal NOT reached"
        lines.append(f"\n  episode {episode}: return={ep_return:+.2f}  ({first_str})")
        grid = render_grid(grid_size, start, result.goal, visited, final_pos)
        for gl in grid.splitlines():
            lines.append(indent + gl)
    lines.append(
        f"\n  rollout total return = {result.total_return:+.2f}  "
        f"redundant_visits = {result.redundant_visits()}"
    )
    return "\n".join(lines)
