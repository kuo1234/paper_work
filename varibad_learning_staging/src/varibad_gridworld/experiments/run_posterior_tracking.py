from __future__ import annotations

import argparse
from random import Random

from ..baselines.bayes_like_search import BayesLikeSearchPolicy
from ..configs.gridworld import ENV_CONFIG
from ..envs.gridworld import GridWorldTask
from ..utils.belief import GoalBelief
from ..utils.posterior_viz import entropy_sparkline, heatmap


def make_env() -> GridWorldTask:
    return GridWorldTask(
        grid_size=ENV_CONFIG["grid_size"],
        allowed_goals=ENV_CONFIG["allowed_goals"],
        start_state=ENV_CONFIG["start_state"],
        episode_horizon=ENV_CONFIG["episode_horizon"],
        reward_goal=ENV_CONFIG["reward_goal"],
        reward_non_goal=ENV_CONFIG["reward_non_goal"],
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 2: track the exact posterior over the hidden goal "
        "along one BAMDP rollout."
    )
    p.add_argument("-N", "--episodes", type=int, default=ENV_CONFIG["num_episodes_per_task"], dest="N")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snapshots", type=int, default=5,
                   help="number of heatmap snapshots across the rollout")
    return p


def main() -> None:
    args = build_parser().parse_args()
    grid_size = ENV_CONFIG["grid_size"]
    H = ENV_CONFIG["episode_horizon"]

    env = make_env()
    env.reset_task(Random(args.seed))
    belief = GoalBelief(env.allowed_goals)
    policy = BayesLikeSearchPolicy()
    policy.reset_task(env)
    act_rng = Random(args.seed + 1)

    max_bits = belief.entropy_bits()  # initial entropy = log2(#allowed goals)
    entropies = [belief.entropy_bits()]
    # Record a heatmap after every step; we pick which to display afterwards so
    # snapshots concentrate on the exploration phase (where the belief changes).
    heatmaps: list[str] = [heatmap(belief, grid_size, env.goal)]
    H_plus = args.N * H

    state = env.state
    first_goal_step = None
    for episode in range(args.N):
        policy.reset_episode() if hasattr(policy, "reset_episode") else None
        for t in range(H):
            t_global = episode * H + t
            action = policy.act(state, belief, act_rng)
            next_state, reward, episode_done, info = env.step(action)
            belief.update(next_state, was_goal=info["on_goal"])
            state = env.state

            entropies.append(belief.entropy_bits())
            heatmaps.append(heatmap(belief, grid_size, env.goal))
            if info["on_goal"] and first_goal_step is None:
                first_goal_step = t_global

    # Choose snapshot steps: spread across the exploration phase (up to a couple
    # steps past first_goal_step) so we actually see the posterior change.
    last_interesting = (first_goal_step + 2) if first_goal_step is not None else (H_plus - 1)
    last_interesting = min(last_interesting, len(heatmaps) - 1)
    n = max(2, args.snapshots)
    snap_steps = sorted({int(round(i * last_interesting / (n - 1))) for i in range(n)})
    snapshots = [(s, heatmaps[s]) for s in snap_steps]

    print(f"### Posterior tracking  goal={env.goal}  H={H}  N={args.N}  H+={H_plus}")
    print(f"initial entropy = {max_bits:.2f} bits  (uniform over {len(env.allowed_goals)} allowed cells)\n")

    print("entropy over time (bits, high -> low as cells are ruled out):")
    print("  " + entropy_sparkline(entropies, max_bits=max_bits))
    print(f"  start={entropies[0]:.2f}   end={entropies[-1]:.2f}   "
          f"first_goal_step={first_goal_step}\n")

    print("posterior heatmaps (darker = higher P(goal here); G = true goal):")
    for step, hm in snapshots:
        ent = entropies[step]
        print(f"\n  t={step}  entropy={ent:.2f} bits")
        for line in hm.splitlines():
            print("    " + line)

    print(
        "\nWatch: entropy steps DOWN as the systematic search rules out cells,\n"
        "then collapses to 0 the moment the goal is found. The heatmap mass\n"
        "concentrates from a uniform sheet onto the single goal cell.\n"
        "This exact posterior is the ground truth a learned VariBAD encoder\n"
        "should reproduce."
    )


if __name__ == "__main__":
    main()
