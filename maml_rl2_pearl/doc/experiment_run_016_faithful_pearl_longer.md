# Run 016 Faithful PEARL Longer Training

## Purpose

Run 016 increases the faithful PEARL training budget without changing the PEARL architecture.

Compared with Run 011:

- iterations increased from `30` to `45`;
- updates per iteration increased from `20` to `30`;
- warmup episodes increased from `4` to `6`;
- prior and posterior data collection are still both used;
- no true latent supervision, oracle behavior cloning, scripted probe, hand-crafted posterior, or actor-z regularizer is added.

This run tests whether the existing faithful PEARL implementation improves with more data and updates before making further architectural decisions.

## Command

```powershell
$env:PYTHONIOENCODING='utf-8'
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_faithful_experiment.py `
  --seeds 7,13,23 `
  --iterations 45 `
  --updates-per-iteration 30 `
  --meta-batch 8 `
  --context-size 64 `
  --rl-batch-size 64 `
  --warmup-episodes 6 `
  --collection-interval 1 `
  --output-dir experiments\pearl_faithful_run016 `
  --device cuda
```

## Results

Mean +/- population std over 3 seeds:

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful Run 016 | -46.296 +/- 1.697 | -46.508 +/- 3.807 | -47.489 +/- 2.390 | -47.284 +/- 2.233 | -46.677 +/- 1.937 |

Seed-level curves:

| Seed | K=0 | K=1 | K=2 | K=3 | K=5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | -45.838 | -44.023 | -45.608 | -46.016 | -45.302 |
| 13 | -48.566 | -51.888 | -50.862 | -50.423 | -49.416 |
| 23 | -44.486 | -43.615 | -45.998 | -45.414 | -45.314 |

Final training metrics:

| Seed | q_loss | v_loss | policy_loss | kl_loss | mean transitions/task |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 4.617 | 0.392 | 10.758 | 2.921 | 5760 |
| 13 | 4.640 | 0.230 | 12.074 | 3.077 | 5760 |
| 23 | 2.642 | 0.062 | 7.590 | 2.084 | 5760 |

## Comparison With Run 011

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful Run 011 | -47.377 +/- 0.331 | -47.611 +/- 3.286 | -49.453 +/- 2.308 | -49.590 +/- 3.354 | -49.360 +/- 2.968 |
| PEARL-faithful Run 016 | -46.296 +/- 1.697 | -46.508 +/- 3.807 | -47.489 +/- 2.390 | -47.284 +/- 2.233 | -46.677 +/- 1.937 |

Run 016 improves the overall PEARL return across all K values, but does not produce stable positive adaptation on the 3-seed mean.

## Current Full Comparison

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful Run 016 | -46.296 +/- 1.697 | -46.508 +/- 3.807 | -47.489 +/- 2.390 | -47.284 +/- 2.233 | -46.677 +/- 1.937 |
| RL2-recurrent Run 012 | -64.535 +/- 6.769 | -65.597 +/- 6.966 | -65.693 +/- 6.957 | -65.699 +/- 6.955 | -65.700 +/- 6.955 |
| RL2-PPO Run 015 | -69.927 +/- 11.566 | -71.509 +/- 13.394 | -71.571 +/- 13.447 | -71.572 +/- 13.449 | -71.572 +/- 13.449 |
| MAML-PG Run 013 | -65.438 +/- 10.321 | -65.243 +/- 12.414 | -65.564 +/- 17.272 | -65.413 +/- 15.859 | -67.144 +/- 16.314 |

## Interpretation

- PEARL remains the strongest implemented method on the local latent navigation benchmark.
- Longer faithful PEARL training improves absolute return, so PEARL is still a reasonable direction to continue optimizing.
- The key weakness is not zero-shot control anymore; it is unstable posterior adaptation. Seeds 7 and 23 improve at K=1, while seed 13 degrades.
- This still is not a complete paper reproduction because the benchmark is not MuJoCo and the MAML/RL2 baselines are not fully tuned TRPO/PPO-grade reproductions.

## Next Required Step

The next faithful PEARL step should focus on diagnostics that are already part of the PEARL training story, not new architecture:

- inspect posterior variance and KL over training;
- compare prior-policy return vs posterior-policy return;
- check whether context encoder posterior changes meaningfully after support episodes;
- tune PEARL hyperparameters such as `kl_lambda`, `alpha`, batch sizes, and update/data ratio.
