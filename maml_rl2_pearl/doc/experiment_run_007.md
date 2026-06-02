# Experiment Run 007/008：Probabilistic Diagnostics and z-Sensitivity

## 目的

Run 006 顯示 probabilistic posterior 已接上，但 sample 評估不穩、mean 評估改善很小。Run 007/008 一次加入幾個診斷與穩定化改動：

- KL annealing
- posterior 多次 sample 平均
- latent prediction MSE 診斷
- posterior variance 診斷
- actor 對 z 的 sensitivity 診斷
- Run 008 額外加入 actor-z sensitivity regularizer

## Run 007 指令

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_probabilistic_experiment.py --seeds 7,13,23 --iterations 20 --updates-per-iteration 20 --batch-size 256 --output-dir experiments/torch_pearl_probabilistic_run007 --device cuda --posterior sample --posterior-samples 8 --kl-weight 0.001 --latent-weight 0.3 --bc-weight 80 --kl-anneal-steps 300
```

## Run 007 結果

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-probabilistic | -48.865 ± 2.655 | -56.105 ± 5.953 | -53.986 ± 5.253 | -53.752 ± 2.579 | -52.814 ± 1.485 |

診斷重點：

- `diagnostic_actor_z_sensitivity` 約 `0.0076` 到 `0.0161`，非常低。
- actor 幾乎不使用 z，這解釋了 context posterior 為什麼沒有轉成有效控制。

## Run 008 指令

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_probabilistic_experiment.py --seeds 7,13,23 --iterations 20 --updates-per-iteration 20 --batch-size 256 --output-dir experiments/torch_pearl_probabilistic_run008 --device cuda --posterior sample --posterior-samples 8 --kl-weight 0.001 --latent-weight 0.3 --bc-weight 80 --kl-anneal-steps 300 --z-action-weight 100
```

## Run 008 結果

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-probabilistic + z regularizer | -48.312 ± 2.837 | -57.077 ± 5.081 | -55.159 ± 2.734 | -52.397 ± 0.417 | -51.971 ± 1.998 |

診斷重點：

- `diagnostic_actor_z_sensitivity` 提升到約 `0.023` 到 `0.028`。
- 但 K=1 仍退化，表示問題不只 actor 忽略 z。
- posterior mean / variance 與 critic actor coupling 仍然不穩。

## 結論

Run 007/008 證明目前瓶頸是可診斷的：

1. actor 對 z 的依賴太弱；
2. 強迫 actor 使用 z 後，performance 仍未改善；
3. 因此下一步要改訓練資料流，而不是繼續加正則。

## 下一步

- 真正分離 context batch 與 RL batch。
- 一個 task 內抽不同 context subsets，讓 posterior 學會處理 context uncertainty。
- evaluation 時用多個 posterior samples 的 action ensemble，而不是只平均 z。
- 加入 latent prediction / action sensitivity 曲線到每個 run 的結果 JSON。
