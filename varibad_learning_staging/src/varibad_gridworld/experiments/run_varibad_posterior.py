from __future__ import annotations

import argparse
from random import Random

import torch

from ..baselines.bayes_like_search import BayesLikeSearchPolicy
from ..baselines.varibad_adapter import load_varibad_adapter
from ..configs.gridworld import ENV_CONFIG
from ..envs.gridworld import GridWorldTask
from ..models.rl2 import ACTION_DIM, build_input
from ..models.varibad import reparameterize
from ..utils.belief import GoalBelief
from ..utils.posterior_viz import entropy_sparkline, heatmap

Cell = tuple[int, int]


def make_env() -> GridWorldTask:
    return GridWorldTask(
        grid_size=ENV_CONFIG["grid_size"],
        allowed_goals=ENV_CONFIG["allowed_goals"],
        start_state=ENV_CONFIG["start_state"],
        episode_horizon=ENV_CONFIG["episode_horizon"],
        reward_goal=ENV_CONFIG["reward_goal"],
        reward_non_goal=ENV_CONFIG["reward_non_goal"],
    )


def decode_goal_map(model, mu, logvar, grid_size: int, allowed_goals) -> dict[Cell, float]:
    """Turn the VariBAD latent into a per-cell P(goal here) map.

    For each allowed cell c we build a dummy transition that ENDS on c and ask the
    reward decoder how likely that step earns the goal reward. sigmoid(logit) is
    then the model's belief that c is the goal. This is the learned analogue of
    the exact belief's posterior.
    """

    m = reparameterize(mu, logvar).detach()
    n = grid_size * grid_size
    probs: dict[Cell, float] = {}
    dummy_s = torch.zeros(1, n)        # state before the step (content-free here)
    dummy_a = torch.zeros(1, ACTION_DIM)
    with torch.no_grad():
        for (r, c) in allowed_goals:
            s_next = torch.zeros(1, n)
            s_next[0, r * grid_size + c] = 1.0
            logit = model.decoder(dummy_s, dummy_a, s_next, m.unsqueeze(0))
            probs[(r, c)] = float(torch.sigmoid(logit).item())
    return probs


def render_prob_map(probs: dict[Cell, float], grid_size: int, goal: Cell) -> str:
    ramp = " .:-=+*#@"
    pmax = max(probs.values()) if probs else 1.0
    rows = []
    for r in range(grid_size):
        chars = []
        for c in range(grid_size):
            p = probs.get((r, c), 0.0)
            if (r, c) == goal:
                chars.append("G")
            elif p <= 0.01:
                chars.append(".")
            else:
                idx = int(round((p / pmax) * (len(ramp) - 1)))
                chars.append(ramp[max(1, idx)])
        rows.append(" ".join(chars))
    return "\n".join(rows)


def side_by_side(left: str, right: str, gap: str = "     ") -> str:
    ll, rl = left.splitlines(), right.splitlines()
    return "\n".join(f"{a}{gap}{b}" for a, b in zip(ll, rl))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare VariBAD's LEARNED posterior to the EXACT belief along a rollout."
    )
    p.add_argument("--varibad", type=str, required=True, help="trained VariBAD model (.pt)")
    p.add_argument("-N", "--episodes", type=int, default=ENV_CONFIG["num_episodes_per_task"], dest="N")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--snapshots", type=int, default=4)
    return p


def main() -> None:
    args = build_parser().parse_args()
    grid_size = ENV_CONFIG["grid_size"]
    H = ENV_CONFIG["episode_horizon"]
    H_plus = args.N * H

    adapter = load_varibad_adapter(args.varibad, grid_size=grid_size, deterministic=True)
    model = adapter.model

    # Run the VariBAD policy; at each step record BOTH the exact belief and the
    # VariBAD-decoded goal map, plus the posterior variance mean.
    env = make_env()
    env.reset_task(Random(args.seed))
    adapter.reset_task(env)
    belief = GoalBelief(env.allowed_goals)
    bayes_entropy0 = belief.entropy_bits()

    enc_hidden = model.encoder.initial_hidden(batch_size=1)
    prev_a = torch.zeros(ACTION_DIM); prev_r = torch.zeros(1); done = torch.zeros(1)

    exact_maps, learned_maps, variances, entropies = [], [], [], []
    state = env.state
    first_goal = None
    rng = Random(args.seed + 1)
    for episode in range(args.N):
        for t in range(H):
            state_oh = torch.tensor(env.get_state_onehot(state), dtype=torch.float32)
            x_t = build_input(state_oh.unsqueeze(0), prev_a.unsqueeze(0), prev_r.unsqueeze(0), done.unsqueeze(0))
            with torch.no_grad():
                mu, logvar, enc_hidden = model.encoder.step(x_t, enc_hidden)
                logits, _ = model.policy(state_oh.unsqueeze(0), mu, logvar)
            action = int(torch.argmax(logits.squeeze(0)).item())
            next_state, reward, episode_done, info = env.step(action)
            belief.update(next_state, was_goal=info["on_goal"])

            exact_maps.append(heatmap(belief, grid_size, env.goal))
            learned = decode_goal_map(model, mu.squeeze(0), logvar.squeeze(0), grid_size, env.allowed_goals)
            learned_maps.append(render_prob_map(learned, grid_size, env.goal))
            variances.append(float(torch.exp(logvar).mean().item()))
            entropies.append(belief.entropy_bits())
            if info["on_goal"] and first_goal is None:
                first_goal = episode * H + t

            prev_a = torch.zeros(ACTION_DIM); prev_a[action] = 1.0
            prev_r = torch.tensor([reward], dtype=torch.float32)
            done = torch.tensor([1.0 if episode_done else 0.0], dtype=torch.float32)
            state = env.state

    print(f"### VariBAD learned posterior  vs  exact belief")
    print(f"goal={env.goal}  H={H}  N={args.N}  H+={H_plus}  first_goal_step={first_goal}\n")

    print("exact-belief entropy (bits):")
    print("  " + entropy_sparkline(entropies, max_bits=bayes_entropy0))
    print("VariBAD posterior variance  mean(exp(logvar)) over time:")
    print("  " + entropy_sparkline(variances, max_bits=max(variances) if variances else 1.0))
    print("  (both should trend DOWN as the task becomes certain)\n")

    last = (first_goal + 2) if first_goal is not None else (H_plus - 1)
    last = min(last, len(exact_maps) - 1)
    n = max(2, args.snapshots)
    steps = sorted({int(round(i * last / (n - 1))) for i in range(n)})

    print(f"{'EXACT belief':<14}{'':5}{'VariBAD learned':<14}")
    for s in steps:
        print(f"\n  t={s + 1}  (exact entropy={entropies[s]:.2f} bits,  varibad var={variances[s]:.3f})")
        print(side_by_side(exact_maps[s], learned_maps[s], gap="      "))

    print(
        "\nRead: if VariBAD truly learned Bayes-optimal task inference, its decoded\n"
        "goal map (right) should track the exact posterior (left) -- ruling out\n"
        "visited cells and concentrating onto the goal. That is the explicit,\n"
        "inspectable belief RL^2 never gives you."
    )


if __name__ == "__main__":
    main()
