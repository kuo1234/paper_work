# Experiment Run 001：Lite Few-Shot Adaptation

## 目的

第一次實驗的目的不是重現正式 MAML-PPO、RL²-PPO、PEARL-SAC，而是先檢查三種 adaptation mechanism 在同一個 2D Goal Navigation protocol 下是否能產生可解讀的 adaptation curve。

## 指令

```powershell
rtk python -B src/run_experiment.py --seeds 7,13,23 --output-dir experiments/first_run
```

## 輸出

- `experiments/first_run/results.json`
- `experiments/first_run/curves.csv`
- `experiments/first_run/summary.csv`

## 方法

| Method | 訓練 / 調參 | 測試時 adaptation |
| --- | --- | --- |
| MAML-lite | train split 的 goal mean 作初始化；validation K=2 選 learning rate / steps | support 後用 reward-distance loss 更新 goal estimate |
| RL2-lite | train split 的 goal mean 作初始 memory；validation K=2 選 learning rate / steps / memory rate | support 後用 exponential memory update |
| PEARL-lite | train goals + grid 作 candidate latent goals；validation K=2 選 temperature / prior weight | context 後用 posterior-weighted candidate mean |

## 結果

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | -43.781 ± 7.232 | -40.335 ± 1.711 | -47.659 ± 2.680 | -47.612 ± 2.685 | -41.903 ± 4.341 |
| MAML-lite | -36.540 ± 3.060 | -9.575 ± 1.917 | -5.114 ± 0.891 | -3.811 ± 0.666 | -3.064 ± 0.529 |
| RL2-lite | -36.540 ± 3.060 | -15.264 ± 2.816 | -9.540 ± 1.882 | -6.604 ± 1.181 | -4.325 ± 0.758 |
| PEARL-lite | -36.540 ± 3.060 | -3.915 ± 0.754 | -3.915 ± 0.754 | -3.915 ± 0.754 | -3.915 ± 0.754 |
| Goal-direction oracle | -2.844 ± 0.497 | -2.844 ± 0.497 | -2.844 ± 0.497 | -2.844 ± 0.497 | -2.844 ± 0.497 |

## 結論

- 三個 lite 方法都明顯優於 Random，表示 dense reward + support probe 可以提供有效 task inference 訊號。
- MAML-lite 呈現穩定遞增曲線，K=5 接近 oracle。
- RL2-lite 遞增較慢，符合 memory update 逐步累積的預期。
- PEARL-lite 在 K=1 就大幅改善，但後續不再改善，表示目前 candidate posterior 在第一條 probe trajectory 後已經足夠集中。

## 限制

- 這不是正式神經網路 policy，也不是論文級 MAML/RL²/PEARL。
- support trajectory 使用 deterministic probe pattern，目的是讓第一次實驗能清楚觀察 adaptation，不代表最終方法的探索策略。
- PEARL-lite 的 K=1 後持平可能來自 grid candidate resolution 和 dense reward 設定，不應過度解讀成正式 PEARL 的行為。
