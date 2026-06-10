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

There are two tracks in this repo:

- **`varibad_toy/`** — the original two-arm *bandit* toy (this README).
- **`varibad_gridworld/`** — the GridWorld BAMDP experiment following
  [`spec.md`](spec.md). Phase 0 (environment) + Phase 1 (4 hard-coded baselines)
  + Phase 2 (posterior tracking) are zero-dependency; Phase 3 (RL²) and
  Phase 4 (Simplified VariBAD) need PyTorch (`pip install -e .[neural]`). See
  [`GRIDWORLD_EXPERIMENT.md`](GRIDWORLD_EXPERIMENT.md). Quick start:

  ```bash
  # Phase 1: baseline comparison table
  PYTHONPATH=src python -m varibad_gridworld.experiments.run_baselines --tasks 300 -N 4
  # Phase 2: watch the exact posterior collapse (zero-dependency)
  PYTHONPATH=src python -m varibad_gridworld.experiments.run_posterior_tracking --seed 7
  # Phase 3: train the RL² baseline, then add it to the table
  PYTHONPATH=src python -m varibad_gridworld.trainers.train_rl2 --updates 1500
  PYTHONPATH=src python -m varibad_gridworld.experiments.run_baselines --rl2 rl2_policy.pt
  # Phase 4: train Simplified VariBAD and inspect learned posterior vs exact belief
  PYTHONPATH=src python -m varibad_gridworld.trainers.train_varibad --updates 3000 --beta 0.01
  PYTHONPATH=src python -m varibad_gridworld.experiments.run_varibad_posterior --varibad varibad_policy.pt --seed 7
  PYTHONPATH=src python -m varibad_gridworld.experiments.run_baselines --rl2 rl2_policy.pt --varibad varibad_policy.pt
  ```

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
