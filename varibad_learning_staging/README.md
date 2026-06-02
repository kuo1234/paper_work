# VARIBAD Learning Lab

This is a small, runnable learning project for the paper:

**VARIBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning**

It is not a full reproduction of the paper. Instead, it turns the core idea into
a tiny toy problem you can inspect and modify:

1. A task is sampled but hidden from the agent.
2. The agent sees transitions and rewards.
3. A belief encoder updates `p(task | history)`.
4. The policy chooses actions conditioned on that belief.

In the full paper, the encoder is a learned variational model and the policy is
trained with deep RL. Here, the belief update is exact so the moving parts are
easy to see.

## Quick Start

```bash
cd ~/paper_code_learning/varibad
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m varibad_toy.demo --episodes 3 --horizon 8 --seed 0
python -m unittest discover -s tests
```

If you do not want a virtual environment:

```bash
cd ~/paper_code_learning/varibad
PYTHONPATH=src python3 -m varibad_toy.demo --episodes 3 --horizon 8 --seed 0
PYTHONPATH=src python3 -m unittest discover -s tests
```

## What To Look For

Run the demo and watch these columns:

- `p_goal_0`: the agent's belief that arm 0 is the high-reward arm.
- `entropy`: uncertainty over the hidden task.
- `action`: chosen from both reward value and information value.

When entropy is high, the policy may choose actions that are informative. As the
history grows, belief sharpens and the policy exploits the arm it thinks is best.

That is the small version of the VARIBAD story: action selection is conditioned
on task uncertainty, not only on the current environment state.

## Files

- `src/varibad_toy/bandit.py`: hidden-task environment.
- `src/varibad_toy/belief.py`: exact Bayesian belief encoder.
- `src/varibad_toy/policies.py`: policies that consume belief.
- `src/varibad_toy/rollout.py`: episode simulation.
- `src/varibad_toy/demo.py`: command-line demo.
- `tests/`: small tests for the learning mechanics.

## Suggested Experiments

1. Change `reward_high` and `reward_low` in `demo.py`.
2. Set `exploration_bonus=0.0` and compare behavior.
3. Increase the number of arms/tasks.
4. Replace `ExactBeliefEncoder` with a small neural network that predicts the
   hidden task from `(action, reward)` history.
