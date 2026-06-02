from __future__ import annotations

import argparse
from random import Random

from .bandit import HiddenGoalBandit
from .policies import BayesAdaptivePolicy, GreedyBeliefPolicy, ThompsonPolicy
from .rollout import EpisodeStep, run_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VARIBAD toy learning lab.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-high", type=float, default=0.8)
    parser.add_argument("--reward-low", type=float, default=0.2)
    parser.add_argument(
        "--policy",
        choices=("bayes", "greedy", "thompson"),
        default="bayes",
    )
    parser.add_argument("--exploration-bonus", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = Random(args.seed)
    policy = make_policy(args.policy, args.exploration_bonus)

    total_reward = 0
    for episode in range(args.episodes):
        env = HiddenGoalBandit.sample_task(
            rng,
            reward_high=args.reward_high,
            reward_low=args.reward_low,
        )
        steps = run_episode(env, policy, horizon=args.horizon, rng=rng)
        total_reward += sum(step.reward for step in steps)
        print_episode(episode, steps)

    steps_count = args.episodes * args.horizon
    print(f"\nmean reward: {total_reward / steps_count:.3f}")


def make_policy(name: str, exploration_bonus: float):
    if name == "bayes":
        return BayesAdaptivePolicy(exploration_bonus=exploration_bonus)
    if name == "greedy":
        return GreedyBeliefPolicy()
    if name == "thompson":
        return ThompsonPolicy()
    raise ValueError(f"unknown policy: {name}")


def print_episode(episode: int, steps: list[EpisodeStep]) -> None:
    goal = steps[0].task_goal_arm if steps else "?"
    print(f"\nepisode {episode} hidden_goal_arm={goal}")
    print("t  action  reward  p_goal_0  entropy")
    for step in steps:
        print(
            f"{step.t:>1}  {step.action:^6}  {step.reward:^6}  "
            f"{step.p_goal_0:>8.3f}  {step.entropy_bits:>7.3f}"
        )


if __name__ == "__main__":
    main()
