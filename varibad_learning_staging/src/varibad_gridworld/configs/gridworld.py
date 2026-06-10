"""GridWorld experiment config (spec section 15), as a zero-dependency dict.

The spec writes this as YAML; we keep it as a Python dict so the project stays
dependency-free. The fields mirror the YAML in spec section 15 exactly.
"""

from __future__ import annotations

# Allowed goal cells: the top three rows (rows 0,1,2), 15 cells total.
# (spec section 3.2). The start (4,0) is in the bottom row, so it is never a goal.
ALLOWED_GOALS: list[tuple[int, int]] = [
    (r, c) for r in range(3) for c in range(5)
]

ENV_CONFIG: dict = {
    "grid_size": 5,
    "start_state": (4, 0),
    "episode_horizon": 15,        # H
    "num_episodes_per_task": 4,   # N  ->  H+ = N*H = 60
    "reward_goal": 1.0,
    "reward_non_goal": -0.1,
    "allowed_goals": ALLOWED_GOALS,
}

# Reserved for Phase 3+ (RL2 / VariBAD). Kept here so the config layout matches
# the spec; unused by the Phase 0/1 baselines.
TRAINING_CONFIG: dict = {
    "algorithm": "a2c",
    "gamma": 0.95,
    "lr_policy": 1e-3,
    "lr_vae": 1e-3,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "num_processes": 16,
    "updates": 5000,
}
