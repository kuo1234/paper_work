# Experiment Run 005：Online PEARL-SAC-Style

## 目的

Run 005 修正 Run 004 固定離線 replay 的主要問題，加入 current actor online collection 與 per-task replay buffer。這比 Run 004 更接近 PEARL 的 meta-training 結構。

## 新增結構

- 每個 train task 有自己的 replay buffer。
- 每個 buffer 先用 probe / noisy-oracle episode bootstrap。
- 每輪更新後，用目前 actor 在每個 train task 上收集新 episode。
- 訓練 batch 從 task-matched replay buffer 取 context 與 RL samples。
- 使用 SAC-style twin Q / target Q / stochastic actor update。

## 指令

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_online_sac_experiment.py --seeds 7,13,23 --iterations 20 --updates-per-iteration 20 --batch-size 256 --output-dir experiments/torch_pearl_online_sac --device cuda
```

## 輸出

- `experiments/torch_pearl_online_sac/results.json`
- `experiments/torch_pearl_online_sac/summary.csv`

## 結果

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-online-SAC-style | -55.246 ± 2.360 | -46.244 ± 1.804 | -46.244 ± 1.804 | -46.244 ± 1.804 | -46.244 ± 1.804 |

## 結論

- Online replay 比 Run 004 offline replay 略好，且 variance 較低。
- 仍弱於 Run 003 supervised PEARL-neural。
- 長訓練單 seed 反而退化，表示問題在 encoder/critic/actor coupling，而不是步數不足。

## 下一步

1. 把 deterministic encoder 改成 probabilistic encoder，輸出 `mu, log_var`。
2. 明確分離 context batch 與 RL batch。
3. 加入 KL regularization 到 standard normal prior。
4. encoder 由 critic loss + KL 訓練，移除 latent supervision。
5. actor collection 用 posterior sampling，而不是 deterministic latent mean。
