from __future__ import annotations

import numpy as np


def zero_policy(obs, context):
    """External policy adapter smoke: always return no-op 7D action."""
    return [0.0] * 7


def target_reach_policy(obs, context):
    """External adapter equivalent of target_reach_fast for smoke tests.

    This demonstrates the future OpenVLA/BTS adapter shape:
        action = fn(obs, context)
    """
    target_key = context.get("target_key")
    action = [0.0] * 7
    if target_key and target_key in obs and "robot0_eef_pos" in obs:
        delta = np.asarray(obs[target_key]) - np.asarray(obs["robot0_eef_pos"])
        action[:3] = np.clip(5.0 * delta, -0.20, 0.20).tolist()
    action[-1] = 0.0
    return action
