# Experiment Run 011：Faithful PEARL Longer Training

## 目的

Run 010 只是 smoke。Run 011 將 faithful PEARL skeleton 增加到較長訓練，檢查在不使用非論文技巧的情況下是否開始出現 K-shot adaptation 訊號。

## 論文一致性修正

Run 011 前修正了資料收集流程：

- 每個 collection interval 對每個 train task 收集 prior-sampling episode。
- 接著用該 task replay context 推論 posterior，再收集 posterior-sampling episode。

這符合 PEARL 的 prior exploration / posterior adaptation 精神，不是額外創新。

## 指令

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_faithful_experiment.py --seeds 7,13,23 --iterations 30 --updates-per-iteration 20 --meta-batch 8 --context-size 64 --rl-batch-size 64 --warmup-episodes 4 --collection-interval 1 --output-dir experiments/pearl_faithful_run011 --device cuda
```

## 結果

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful Run 011 | -47.377 ± 0.331 | -47.611 ± 3.286 | -49.453 ± 2.308 | -49.590 ± 3.354 | -49.360 ± 2.968 |

## Seed-level observation

| Seed | K=0 | K=1 | K=5 | 判讀 |
| ---: | ---: | ---: | ---: | --- |
| 7 | -47.546 | -43.442 | -45.917 | K=1 有改善 |
| 13 | -46.915 | -47.919 | -49.003 | adaptation 變差 |
| 23 | -47.671 | -51.473 | -53.160 | adaptation 變差 |

## 結論

- faithful PEARL skeleton 已可進行較長 CUDA 訓練。
- 單一 seed 出現 K=1 改善，但 3 seeds 平均尚未穩定改善。
- 目前不能說 PEARL 已比 MAML / RL² 強。
- 目前也不能說 PEARL 不適合作為優化方向，因為訓練量、任務 benchmark、baseline 都還不足。

## 下一步

要回答最終問題，下一步不是再改 PEARL 架構，而是：

1. 增加 faithful PEARL training iterations 並保存 learning curve。
2. 實作或接入 faithful MAML / RL² baselines。
3. 在同一 task split、同一 K-shot protocol 下比較。
4. 至少 3 seeds，最好 5 seeds。
5. 再判斷 PEARL 是否值得作為後續優化方向。
