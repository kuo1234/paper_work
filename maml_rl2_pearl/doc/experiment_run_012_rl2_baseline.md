# Run 012 RL2 Recurrent Baseline

## Purpose

This run adds the first neural RL2-style recurrent baseline on the same latent navigation task used by the faithful PEARL runner.

The implementation is intentionally conservative:

- recurrent policy input is `(state, previous_action, previous_reward, done)`;
- GRU hidden state is reset only between tasks;
- hidden state persists across K trajectories during meta-test;
- training uses on-policy REINFORCE with a learned value baseline;
- no task labels, oracle probes, PEARL posterior, or hand-crafted latent summaries are used.

This is not yet a full RL2-PPO/TRPO reproduction, but it gives a real neural recurrent baseline before implementing the stronger paper-level version.

## Command

```powershell
$env:PYTHONIOENCODING='utf-8'
rtk .\.venv-py312\Scripts\python.exe -B src\torch_rl2_baseline_experiment.py `
  --seeds 7,13,23 `
  --iterations 50 `
  --meta-batch 8 `
  --episodes-per-task 3 `
  --output-dir experiments\rl2_baseline_run012 `
  --device cuda
```

## Environment

- Python: `.venv-py312`
- PyTorch: CUDA build
- Device: CUDA
- Output: `experiments/rl2_baseline_run012/`

## Results

Mean +/- population std over 3 seeds:

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RL2-recurrent | -64.535 +/- 6.769 | -65.597 +/- 6.966 | -65.693 +/- 6.957 | -65.699 +/- 6.955 | -65.700 +/- 6.955 |

Seed-level curves:

| Seed | K=0 | K=1 | K=2 | K=3 | K=5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | -73.493 | -75.116 | -75.221 | -75.226 | -75.226 |
| 13 | -57.134 | -58.638 | -58.804 | -58.817 | -58.818 |
| 23 | -62.978 | -63.038 | -63.055 | -63.055 | -63.055 |

## Comparison With Faithful PEARL Run 011

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful Run 011 | -47.377 +/- 0.331 | -47.611 +/- 3.286 | -49.453 +/- 2.308 | -49.590 +/- 3.354 | -49.360 +/- 2.968 |
| RL2-recurrent Run 012 | -64.535 +/- 6.769 | -65.597 +/- 6.966 | -65.693 +/- 6.957 | -65.699 +/- 6.955 | -65.700 +/- 6.955 |

On this small benchmark, faithful PEARL currently outperforms the lightweight neural RL2 baseline at every K.

This is useful evidence, but not the final answer to the research question. The RL2 runner still needs a stronger PPO/TRPO-style training loop before it can be treated as a paper-level RL2 baseline.

## Current Interpretation

- PEARL's probabilistic context inference is already competitive on this task after removing non-paper aids.
- The current RL2 recurrent baseline does not show useful within-task improvement from K=0 to K=5.
- The comparison is still incomplete because only a lightweight MAML-PG baseline has been added so far, and RL2 is not yet a full PPO/TRPO-grade implementation.

## Next Required Step

Implement MAML with a policy-gradient inner loop on the same task split, then upgrade RL2 training from REINFORCE to PPO-style optimization if the goal is a paper-level baseline comparison.
