# Run 013 MAML-PG Baseline

## Purpose

This run adds the first neural MAML-style baseline on the same latent navigation task used by faithful PEARL and RL2.

The implementation:

- meta-learns a Gaussian policy initialization;
- collects a support episode for each sampled task;
- applies one policy-gradient inner update to the policy parameters;
- optimizes the query episode loss from the adapted parameters;
- uses no task latent labels, oracle probes, recurrent memory, or PEARL posterior.

This is a small MAML-PG baseline. It is not yet a full MAML-TRPO/PPO reproduction.

## Command

```powershell
$env:PYTHONIOENCODING='utf-8'
rtk .\.venv-py312\Scripts\python.exe -B src\torch_maml_baseline_experiment.py `
  --seeds 7,13,23 `
  --iterations 10 `
  --meta-batch 4 `
  --output-dir experiments\maml_baseline_run013 `
  --device cuda
```

## Results

Mean +/- population std over 3 seeds:

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAML-PG | -65.438 +/- 10.321 | -65.243 +/- 12.414 | -65.564 +/- 17.272 | -65.413 +/- 15.859 | -67.144 +/- 16.314 |

Seed-level curves:

| Seed | K=0 | K=1 | K=2 | K=3 | K=5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | -78.926 | -82.307 | -89.971 | -87.838 | -87.927 |
| 13 | -53.864 | -53.135 | -52.522 | -54.435 | -65.428 |
| 23 | -63.525 | -60.287 | -54.199 | -53.964 | -48.076 |

## Comparison

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful Run 011 | -47.377 +/- 0.331 | -47.611 +/- 3.286 | -49.453 +/- 2.308 | -49.590 +/- 3.354 | -49.360 +/- 2.968 |
| RL2-recurrent Run 012 | -64.535 +/- 6.769 | -65.597 +/- 6.966 | -65.693 +/- 6.957 | -65.699 +/- 6.955 | -65.700 +/- 6.955 |
| MAML-PG Run 013 | -65.438 +/- 10.321 | -65.243 +/- 12.414 | -65.564 +/- 17.272 | -65.413 +/- 15.859 | -67.144 +/- 16.314 |

## Interpretation

- Faithful PEARL currently has the best return among the three implemented neural baselines on this small latent navigation task.
- MAML-PG shows high seed variance. Seed 23 improves with K, but seed 7 degrades strongly.
- RL2-recurrent is more stable than MAML-PG but does not improve with K.
- This is still not the final paper-level conclusion because MAML and RL2 are not yet PPO/TRPO-grade baselines and the benchmark is not MuJoCo.

## Next Required Step

For a paper-level comparison, upgrade the baselines:

- MAML: move from this compact MAML-PG runner toward MAML-TRPO/PPO-style training.
- RL2: move from REINFORCE + value baseline toward PPO/TRPO-style recurrent training.
- PEARL: continue longer faithful PEARL-SAC training and optionally add MuJoCo-style benchmark tasks.
