"""Tiny educational components for VARIBAD-style meta-RL."""

from .bandit import HiddenGoalBandit
from .belief import ExactBeliefEncoder
from .policies import BayesAdaptivePolicy, GreedyBeliefPolicy, ThompsonPolicy
from .rollout import EpisodeStep, run_episode

__all__ = [
    "BayesAdaptivePolicy",
    "EpisodeStep",
    "ExactBeliefEncoder",
    "GreedyBeliefPolicy",
    "HiddenGoalBandit",
    "ThompsonPolicy",
    "run_episode",
]
