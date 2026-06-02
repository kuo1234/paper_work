# Experiment Run 010：Faithful PEARL Skeleton Smoke

## 目的

Run 010 是回到 PEARL 原論文路線後的第一個乾淨 skeleton。它刻意移除前面 research variants 的非論文技巧：

- no true latent supervision
- no oracle behavior cloning
- no scripted probe context
- no analytic candidate posterior
- no actor-z sensitivity regularizer
- no hand-crafted summary context encoder

## 依據

- PEARL 論文：Rakelly et al., "Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables", ICML 2019. https://proceedings.mlr.press/v97/rakelly19a.html
- 官方 reference implementation： https://github.com/katerakelly/oyster

論文重點是 off-policy meta-RL、probabilistic context variable、posterior sampling、以及將 task inference 與 control disentangle。

## 已實作的 PEARL 核心

| PEARL component | Run 010 狀態 |
| --- | --- |
| per-task replay buffer | 已有 |
| transition set context encoder | 已有 |
| product-of-Gaussians posterior aggregation | 已有 |
| probabilistic z sampling | 已有 |
| actor conditioned on `(s, z)` | 已有 |
| Q-functions conditioned on `(s, a, z)` | 已有 |
| value / target value network | 已有 |
| context batch 與 RL batch 分離 | 已有 |
| critic loss + KL 訓練 encoder | 已有 |
| SAC-style actor/value/Q update | 已有 |
| meta-test K-shot context adaptation | 已有 |

## 仍未完成

- 尚未做長時間 meta-training。
- 尚未接正式 PEARL benchmark，例如 sparse point robot、Half-Cheetah-Vel、Ant-Goal。
- 尚未加入 MAML / RL² faithful baselines。
- 尚未驗證 sample efficiency 是否優於 MAML / RL²。

## 指令

這是 smoke run，不是正式結果：

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_faithful_experiment.py --seeds 7,13,23 --iterations 2 --updates-per-iteration 3 --meta-batch 4 --context-size 16 --rl-batch-size 16 --warmup-episodes 1 --output-dir experiments/pearl_faithful_smoke --device cuda
```

## 結果

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful smoke | -69.808 ± 2.429 | -71.842 ± 4.146 | -68.782 ± 6.225 | -73.823 ± 2.013 | -71.095 ± 4.075 |

## 判讀

- 程式架構已回到 PEARL 主線。
- smoke run 沒有學起來，這不代表 PEARL 不行，只代表目前訓練量非常小。
- 與前面 Run 002 analytic PEARL-latent 或 Run 003 supervised PEARL-neural 不可直接比較，因為那些版本用了非論文輔助訊號。

## 下一步

1. 增加 meta-training iterations。
2. 記錄 learning curve，而不是只看 final K-shot。
3. 確認 Q / V / policy / KL loss 是否穩定。
4. 再加入 MAML / RL² faithful baselines 做公平比較。
