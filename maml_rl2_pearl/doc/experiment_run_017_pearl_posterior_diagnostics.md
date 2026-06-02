# Run 017 PEARL Posterior Diagnostics

## Purpose

Run 017 adds diagnostics for the faithful PEARL posterior without changing the training architecture.

The new output inspects `q(z|c)` during meta-test:

- posterior mean norm;
- posterior variance mean;
- KL to standard normal prior;
- diagnostic query return under the same K-shot context protocol.

This is evaluation-only instrumentation. It does not add latent supervision, oracle probes, behavior cloning, handcrafted posterior logic, or new losses.

## Code Change

`src/torch_pearl_faithful_experiment.py` now writes:

- `posterior_diagnostics.csv`
- `posterior_diagnostics` inside `results.json`

The existing `summary.csv` and `training_curve.csv` outputs remain unchanged.

## Command

```powershell
$env:PYTHONIOENCODING='utf-8'
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_faithful_experiment.py `
  --seeds 7,13,23 `
  --iterations 15 `
  --updates-per-iteration 20 `
  --meta-batch 8 `
  --context-size 64 `
  --rl-batch-size 64 `
  --warmup-episodes 4 `
  --collection-interval 1 `
  --output-dir experiments\pearl_faithful_diag_run017 `
  --device cuda
```

## Return Results

Mean +/- population std over 3 seeds:

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful Run 017 | -54.122 +/- 2.638 | -59.330 +/- 3.277 | -55.967 +/- 1.858 | -55.470 +/- 0.953 | -56.388 +/- 1.442 |

## Posterior Diagnostics

Mean across seeds from `posterior_diagnostics.csv`:

| K | Diagnostic return | posterior mean norm | posterior var mean | posterior KL |
| ---: | ---: | ---: | ---: | ---: |
| 0 | -54.872 | 0.000 | 1.00000 | 0.000 |
| 1 | -57.644 | 0.986 | 0.18854 | 2.464 |
| 2 | -56.467 | 0.981 | 0.09629 | 3.630 |
| 3 | -56.435 | 0.985 | 0.06378 | 4.379 |
| 5 | -55.132 | 1.012 | 0.03879 | 5.340 |

## Interpretation

- The posterior is not ignoring context. After one support episode, posterior variance drops from `1.0` to about `0.1885`, and KL increases from `0.0` to about `2.46`.
- Additional support episodes keep shrinking posterior variance and increasing KL.
- Despite that, K=1 return is worse than K=0 in this run.
- The current instability is therefore not simply "encoder does not use context." It is more likely a mismatch between inferred posterior samples and the policy/critic behavior learned under that latent.

## Current Research Implication

PEARL remains the strongest local method, but the next optimization should stay within the original PEARL route:

- tune KL strength and entropy temperature;
- inspect prior-policy vs posterior-policy returns separately;
- check whether posterior sampling is too confident too early;
- tune context size and update/data ratio;
- avoid adding non-paper mechanisms until the faithful posterior-policy coupling is understood.
