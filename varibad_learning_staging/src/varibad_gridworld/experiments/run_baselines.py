from __future__ import annotations

import argparse
from random import Random

from ..baselines.bayes_like_search import BayesLikeSearchPolicy
from ..baselines.oracle_policy import OraclePolicy
from ..baselines.posterior_sampling import PosteriorSamplingPolicy
from ..baselines.random_policy import RandomPolicy
from ..configs.gridworld import ENV_CONFIG
from ..envs.gridworld import GridWorldTask
from ..utils.metrics import aggregate
from ..utils.rollout import run_bamdp_rollout
from ..utils.visualization import render_rollout

POLICY_BUILDERS = {
    "random": RandomPolicy,
    "oracle": OraclePolicy,
    "posterior_sampling": PosteriorSamplingPolicy,
    "bayes_like_search": BayesLikeSearchPolicy,
}
# Display order: upper bound first, worst last (spec section 18).
ORDER = ["oracle", "bayes_like_search", "posterior_sampling", "random"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 1: run hard-coded baselines on the GridWorld BAMDP.")
    p.add_argument("--tasks", type=int, default=300, help="number of BAMDP rollouts (tasks)")
    p.add_argument("-N", "--episodes", type=int, default=ENV_CONFIG["num_episodes_per_task"],
                   dest="N", help="episodes per rollout (N); H+ = N*H")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--visualize", choices=list(POLICY_BUILDERS), default=None,
                   help="print one ASCII rollout for this policy before the table")
    p.add_argument("--rl2", type=str, default=None,
                   help="path to a trained RL2 model (.pt); adds an RL2 row to the table")
    p.add_argument("--varibad", type=str, default=None,
                   help="path to a trained VariBAD model (.pt); adds a VariBAD row")
    return p


def make_env() -> GridWorldTask:
    return GridWorldTask(
        grid_size=ENV_CONFIG["grid_size"],
        allowed_goals=ENV_CONFIG["allowed_goals"],
        start_state=ENV_CONFIG["start_state"],
        episode_horizon=ENV_CONFIG["episode_horizon"],
        reward_goal=ENV_CONFIG["reward_goal"],
        reward_non_goal=ENV_CONFIG["reward_non_goal"],
    )


def evaluate(policy_factory, label: str, n_tasks: int, N: int, seed: int):
    """Run one policy over n_tasks rollouts on a FIXED set of task seeds.

    ``policy_factory()`` returns a fresh policy instance per task. Task goals come
    from ``task_rng`` seeded identically for every policy, so all policies face
    the same goals (fair comparison). Action randomness uses a separate rng.
    """

    results = []
    act_rng = Random(seed + 17)
    for k in range(n_tasks):
        task_rng = Random(seed * 100003 + k)  # same goal sequence for every policy
        env = make_env()
        policy = policy_factory()
        results.append(run_bamdp_rollout(env, policy, N=N, rng=act_rng, task_rng=task_rng))
    return aggregate(results)


def print_table(rows: list[tuple[str, object]], N: int, H: int) -> None:
    print(f"\n=== baseline comparison over rollouts  (H={H}, N={N}, H+={N*H}) ===\n")
    cols = (
        f"{'Method':<20}{'AvgReturn':>11}{'Ep1':>8}{'Ep2':>8}"
        f"{'FirstGoal':>11}{'Success%':>10}{'Redundant':>11}"
    )
    print(cols)
    print("-" * len(cols))
    for label, m in rows:
        ep1 = m.episode_returns[0]
        ep2 = m.episode_returns[1] if len(m.episode_returns) > 1 else float("nan")
        print(
            f"{label:<20}{m.avg_return:>11.2f}{ep1:>8.2f}{ep2:>8.2f}"
            f"{m.avg_first_goal_step:>11.2f}{m.success_rate_rollout * 100:>9.0f}%"
            f"{m.avg_redundant_visits:>11.2f}"
        )
    print(
        "\nExpected: Oracle > Bayes-like Search > Posterior Sampling > Random.\n"
        "Bayes-like Search should have far FEWER redundant visits than Posterior\n"
        "Sampling -- that gap is the cost of unsystematic exploration.\n"
        "(Oracle is the upper bound, NOT Bayes-optimal: it never has to explore.)\n"
        "A trained RL2 (if --rl2 given) should land between Posterior Sampling and\n"
        "Bayes-like Search once it has learned to explore."
    )


def main() -> None:
    args = build_parser().parse_args()
    H = ENV_CONFIG["episode_horizon"]

    if args.visualize:
        env = make_env()
        policy = POLICY_BUILDERS[args.visualize]()
        result = run_bamdp_rollout(
            env, policy, N=args.N,
            rng=Random(args.seed + 1),
            task_rng=Random(args.seed),
        )
        print(f"### ASCII rollout: {policy.name}\n")
        print(render_rollout(result, ENV_CONFIG["grid_size"], ENV_CONFIG["start_state"]))

    rows: list[tuple[str, object]] = []
    for name in ORDER:
        label = POLICY_BUILDERS[name]().name
        m = evaluate(POLICY_BUILDERS[name], label, args.tasks, args.N, args.seed)
        rows.append((label, m))

    if args.rl2:
        # Lazy import so the baseline table works without torch installed.
        from ..baselines.rl2_adapter import load_rl2_adapter
        shared = load_rl2_adapter(args.rl2, grid_size=ENV_CONFIG["grid_size"], deterministic=True)
        # Reuse the single loaded model; reset_task gives each rollout a fresh
        # hidden state, so one instance is correct and cheap.
        m = evaluate(lambda: shared, "RL2", args.tasks, args.N, args.seed)
        # Insert RL2 just below Bayes-like Search for readability.
        rows.insert(2, ("RL2", m))

    if args.varibad:
        from ..baselines.varibad_adapter import load_varibad_adapter
        shared = load_varibad_adapter(args.varibad, grid_size=ENV_CONFIG["grid_size"], deterministic=True)
        m = evaluate(lambda: shared, "VariBAD", args.tasks, args.N, args.seed)
        rows.insert(2, ("VariBAD", m))

    print_table(rows, N=args.N, H=H)


if __name__ == "__main__":
    main()
