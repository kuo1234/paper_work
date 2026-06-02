# Experiment Run 002：Latent Goal + Wind

## 目的

Run 001 的 task 只有 hidden goal，dense reward + probe trajectory 很快就能解出任務。Run 002 把 task latent 擴成：

```text
z = (goal_x, goal_y, wind_x, wind_y)
```

環境 dynamics：

```text
next_state = state + clipped(action) + wind
```

reward：

```text
r = -distance(next_state, goal) - action_penalty
```

這讓方法必須同時推論 reward target 與 hidden dynamics，比單純 goal navigation 更接近 PEARL 的 task inference 問題。

## 指令

```powershell
rtk python -B src/run_latent_experiment.py --seeds 7,13,23 --output-dir experiments/latent_run
```

## 輸出

- `experiments/latent_run/results.json`
- `experiments/latent_run/curves.csv`
- `experiments/latent_run/summary.csv`

## 方法

| Method | adaptation mechanism |
| --- | --- |
| MAML-latent | support 後用 reward-distance gradient 更新 goal estimate，並由 transition residual 估 wind |
| RL2-latent | support 後把 inferred goal/wind 累積進 memory estimate |
| PEARL-latent | 用候選 latent tasks 的 reward likelihood + transition likelihood 做 posterior mean |

## 結果

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | -70.768 ± 4.415 | -68.600 ± 4.119 | -81.856 ± 5.176 | -69.604 ± 6.264 | -75.929 ± 4.801 |
| MAML-latent | -45.373 ± 2.316 | -9.457 ± 1.186 | -6.187 ± 1.278 | -5.329 ± 1.186 | -4.979 ± 1.133 |
| RL2-latent | -45.373 ± 2.316 | -21.726 ± 1.091 | -15.359 ± 0.535 | -11.775 ± 0.768 | -8.144 ± 1.308 |
| PEARL-latent | -45.373 ± 2.316 | -7.217 ± 0.234 | -7.217 ± 0.234 | -7.217 ± 0.234 | -7.217 ± 0.234 |
| Latent oracle | -4.040 ± 0.311 | -4.040 ± 0.311 | -4.040 ± 0.311 | -4.040 ± 0.311 | -4.040 ± 0.311 |

## 結論

- Run 002 比 Run 001 更適合觀察 task inference，因為 hidden wind 會讓未適應 policy 明顯偏離。
- PEARL-latent 最快利用第一條 support trajectory，K=1 後已有大幅改善。
- MAML-latent 的曲線更像 gradient adaptation：K 越大越接近 oracle。
- RL2-latent 目前是 explicit memory proxy，能改善但速度較慢；真正 RNN policy 可能需要用 PPO/A2C 訓練才能公平比較。

## 限制與下一步

- 目前仍是 dependency-free proxy，不是神經網路 actor-critic。
- 若要更接近論文，下一步應新增 PyTorch 依賴，實作 PEARL-style encoder + SAC actor/critic，並用 replay buffer 做 off-policy 更新。
- 若任務仍太容易，下一步可改 sparse reward、加入障礙物或使用 bimodal task distribution。
