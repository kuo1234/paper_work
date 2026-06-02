# Run 015 RL2-PPO Baseline

## Purpose

This run upgrades the recurrent RL2 baseline from REINFORCE to clipped PPO-style training.

The implementation:

- uses a GRU policy/value network;
- feeds `(state, previous_action, previous_reward, done)` at every step;
- resets hidden state only between tasks;
- keeps hidden state across multiple episodes from the same task;
- performs clipped PPO updates over complete task sequences;
- uses reward scaling for PPO value-target stability;
- uses no task labels, oracle probes, PEARL posterior, or hand-crafted latent summaries.

This is still not a full RL2-TRPO reproduction, but it is a stronger baseline than Run 012.

## Command

```powershell
$env:PYTHONIOENCODING='utf-8'
rtk .\.venv-py312\Scripts\python.exe -B src\torch_rl2_ppo_experiment.py `
  --seeds 7,13,23 `
  --iterations 15 `
  --meta-batch 4 `
  --episodes-per-task 3 `
  --ppo-epochs 3 `
  --reward-scale 0.05 `
  --output-dir experiments\rl2_ppo_run015 `
  --device cuda
```

## Results

Mean +/- population std over 3 seeds:

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RL2-PPO | -69.927 +/- 11.566 | -71.509 +/- 13.394 | -71.571 +/- 13.447 | -71.572 +/- 13.449 | -71.572 +/- 13.449 |

Seed-level curves:

| Seed | K=0 | K=1 | K=2 | K=3 | K=5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | -85.824 | -90.147 | -90.288 | -90.292 | -90.293 |
| 13 | -58.641 | -59.263 | -59.302 | -59.302 | -59.303 |
| 23 | -65.315 | -65.116 | -65.122 | -65.122 | -65.122 |

## Comparison

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful Run 011 | -47.377 +/- 0.331 | -47.611 +/- 3.286 | -49.453 +/- 2.308 | -49.590 +/- 3.354 | -49.360 +/- 2.968 |
| RL2-recurrent Run 012 | -64.535 +/- 6.769 | -65.597 +/- 6.966 | -65.693 +/- 6.957 | -65.699 +/- 6.955 | -65.700 +/- 6.955 |
| RL2-PPO Run 015 | -69.927 +/- 11.566 | -71.509 +/- 13.394 | -71.571 +/- 13.447 | -71.572 +/- 13.449 | -71.572 +/- 13.449 |
| MAML-PG Run 013 | -65.438 +/- 10.321 | -65.243 +/- 12.414 | -65.564 +/- 17.272 | -65.413 +/- 15.859 | -67.144 +/- 16.314 |

## Interpretation

- Faithful PEARL remains the best method among the implemented neural baselines on this small latent navigation task.
- RL2-PPO is more algorithmically appropriate than the REINFORCE runner, but this short run still does not show useful recurrent adaptation from K=0 to K=5.
- Reward scaling reduced PPO value loss from hundreds to roughly `0.2-0.4`, so Run 015 is the cleaner RL2-PPO comparison than the unscaled Run 014.
- This still does not prove PEARL is better than paper-level RL2 on standard benchmarks. It only strengthens the current local benchmark evidence.

## Next Required Step

To approach paper-level comparison:

- train RL2-PPO longer and tune PPO hyperparameters;
- upgrade MAML-PG toward PPO/TRPO-style optimization;
- continue faithful PEARL longer training;
- eventually move the comparison to MuJoCo-style meta-RL tasks.
