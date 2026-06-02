# Experiment Run 009：Separated Context/RL Batches and Ensemble Evaluation

## 目的

Run 009 針對 Run 007/008 的瓶頸改資料流：

- context batch 和 RL batch 從同一 task 的 replay 中分開抽樣；
- 每次 context 只使用 replay 的 subset；
- evaluation 使用 posterior rollout ensemble；
- 同時保留 KL annealing、posterior sample averaging 與診斷 metrics。

## Sample Evaluation 指令

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_probabilistic_experiment.py --seeds 7,13,23 --iterations 20 --updates-per-iteration 20 --batch-size 256 --output-dir experiments/torch_pearl_probabilistic_run009 --device cuda --posterior sample --posterior-samples 8 --rollout-ensemble 3 --context-items 64 --kl-weight 0.001 --latent-weight 0.3 --bc-weight 80 --kl-anneal-steps 300
```

## Mean Evaluation 指令

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_probabilistic_experiment.py --seeds 7,13,23 --iterations 20 --updates-per-iteration 20 --batch-size 256 --output-dir experiments/torch_pearl_probabilistic_run009_mean --device cuda --posterior mean --posterior-samples 8 --rollout-ensemble 3 --context-items 64 --kl-weight 0.001 --latent-weight 0.3 --bc-weight 80 --kl-anneal-steps 300
```

## 結果

| Evaluation | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sample ensemble | -49.681 ± 3.232 | -52.926 ± 4.879 | -53.122 ± 4.863 | -51.948 ± 3.864 | -53.841 ± 5.078 |
| posterior mean | -49.681 ± 3.232 | -51.127 ± 3.009 | -51.127 ± 3.009 | -51.127 ± 3.009 | -51.127 ± 3.009 |

## 結論

- Run 009 明顯降低 sample evaluation 的不穩定性。
- posterior mean 版本只小幅退化，表示 sampling noise 已經不是最大問題。
- 剩餘問題是 posterior update 的方向沒有讓 actor 控制變好。
- actor-z sensitivity 仍偏低，約 `0.009` 到 `0.023`。

## 下一步

下一步應該改 actor/encoder coupling，而不是繼續調 evaluation：

1. actor input 改用 normalized latent，避免 z 尺度不穩。
2. 加入 FiLM / gating layer，讓 z 直接調制 hidden features。
3. actor loss 加上 paired-task contrastive action loss：不同 z 對同一 state 應輸出不同方向。
4. 移除或降低 oracle BC，避免 actor 學成忽略 z 的平均策略。
