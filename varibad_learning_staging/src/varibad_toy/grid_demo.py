from __future__ import annotations

import argparse
from random import Random

from .grid_belief import GridBeliefEncoder
from .grid_policies import (
    BayesOptimalGridPolicy,
    GreedyGridPolicy,
    GridPolicy,
    RandomWalkPolicy,
)
from .grid_rollout import GridEpisodeResult, run_grid_episode
from .gridworld import Cell, HiddenGoalGrid

POLICIES = ("bayes_optimal", "greedy", "random")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GridWorld experiment: is belief-conditioned behaviour close "
        "to Bayes-optimal exploration?"
    )
    parser.add_argument("--policy", choices=POLICIES, default="bayes_optimal",
                        help="policy to visualise step-by-step")
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--compare-episodes", type=int, default=200,
                        help="episodes per policy for the final comparison table")
    return parser


def make_policy(name: str) -> GridPolicy:
    if name == "bayes_optimal":
        return BayesOptimalGridPolicy()
    if name == "greedy":
        return GreedyGridPolicy()
    if name == "random":
        return RandomWalkPolicy()
    raise ValueError(f"unknown policy: {name}")


def sample_tasks(rng: Random, size: int, n: int) -> list[HiddenGoalGrid]:
    return [HiddenGoalGrid.sample_task(rng, size=size) for _ in range(n)]


def render_grid(size: int, start: Cell, goal: Cell, visited: set[Cell], pos: Cell) -> str:
    """Text view of one moment.  S=start  G=goal  *=agent  .=visited  _=unseen."""

    rows = []
    for r in range(size):
        cells = []
        for c in range(size):
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


def print_episode(episode: int, env: HiddenGoalGrid, result: GridEpisodeResult) -> None:
    print(f"\nepisode {episode}  goal={env.goal_cell}  start={env.start_cell}")
    visited: set[Cell] = {env.start_cell}
    # Show the final trajectory map plus the entropy curve.
    for step in result.steps:
        visited.add(step.state)
    final_pos = result.steps[-1].state if result.steps else env.start_cell
    print(render_grid(env.size, env.start_cell, env.goal_cell, visited, final_pos))

    print("t  pos     entropy  candidates")
    for step in result.steps:
        marker = "  <- found" if step.found_goal else ""
        print(
            f"{step.t:>2} {str(step.state):>7}  {step.entropy_bits:>6.2f}  "
            f"{step.n_candidates:>9}{marker}"
        )
    sg = result.steps_to_goal
    print(f"steps_to_goal={sg}  oracle(shortest)={result.oracle_steps}")


def run_comparison(size: int, episodes: int, seed: int, max_steps: int) -> dict[str, dict]:
    """Run every policy on the SAME sampled tasks, so the comparison is fair."""

    task_rng = Random(seed)
    tasks = sample_tasks(task_rng, size=size, n=episodes)

    summary: dict[str, dict] = {}
    for offset, name in enumerate(POLICIES):
        policy = make_policy(name)
        # Fresh action RNG per policy, seeded deterministically (no hash() -- its
        # ordering varies across runs unless PYTHONHASHSEED is fixed).
        act_rng = Random(seed * 1000 + offset)
        steps_list: list[int] = []
        oracle_list: list[int] = []
        found_count = 0
        for env in tasks:
            result = run_grid_episode(env, policy, max_steps=max_steps, rng=act_rng)
            if result.found and result.steps_to_goal is not None:
                steps_list.append(result.steps_to_goal)
                oracle_list.append(result.oracle_steps)
                found_count += 1
        mean_steps = sum(steps_list) / len(steps_list) if steps_list else float("nan")
        mean_oracle = sum(oracle_list) / len(oracle_list) if oracle_list else float("nan")
        summary[name] = {
            "mean_steps": mean_steps,
            "mean_oracle": mean_oracle,
            "found_rate": found_count / episodes,
        }
    return summary


def print_comparison(summary: dict[str, dict], episodes: int) -> None:
    bayes = summary["bayes_optimal"]["mean_steps"]
    print(f"\n=== comparison over {episodes} shared tasks ===")
    print(f"{'policy':<14}{'mean_steps_to_goal':>20}{'regret_vs_bayes':>18}{'found':>8}")
    for name in POLICIES:
        s = summary[name]
        regret = s["mean_steps"] - bayes
        print(
            f"{name:<14}{s['mean_steps']:>20.2f}{regret:>+18.2f}"
            f"{s['found_rate'] * 100:>7.0f}%"
        )
    print(
        "\nregret_vs_bayes = mean_steps(policy) - mean_steps(bayes_optimal).\n"
        "0 means it explores as efficiently as the Bayes-optimal policy.\n"
        "(bayes_optimal is the oracle lower bound here; a learned VARIBAD policy\n"
        " would slot into this same table.)"
    )


def main() -> None:
    args = build_parser().parse_args()

    # 1) Step-by-step visualisation of the chosen policy.
    task_rng = Random(args.seed)
    act_rng = Random(args.seed + 1)
    policy = make_policy(args.policy)
    print(f"### step-by-step view: policy={args.policy}, size={args.size}")
    for episode in range(args.episodes):
        env = HiddenGoalGrid.sample_task(task_rng, size=args.size)
        result = run_grid_episode(env, policy, max_steps=args.max_steps, rng=act_rng)
        print_episode(episode, env, result)

    # 2) Quantitative comparison across all policies on shared tasks.
    summary = run_comparison(
        size=args.size,
        episodes=args.compare_episodes,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    print_comparison(summary, args.compare_episodes)


if __name__ == "__main__":
    main()
