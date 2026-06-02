# Faithful PEARL Reproduction 路線圖

本文件定義後續回到 PEARL 原論文路線時，應該保留、移除、重做的部分。

## 目標

建立一個更接近 PEARL 論文 Algorithm 1 的實作：

- probabilistic context encoder `q(z|c)`
- per-task replay buffer
- context batch 與 RL batch 分開抽樣
- actor / critic conditioned on z
- SAC-style off-policy update
- encoder 由 critic loss + KL regularization 訓練
- meta-test 時從 prior 開始，收集 context 後更新 posterior

## 要保留的部分

| 目前元件 | 保留原因 |
| --- | --- |
| `LatentWindNavigationEnv` | 可作為低成本 debugging benchmark |
| per-task replay buffer | PEARL 核心 |
| context/RL batch separation | PEARL 核心 |
| probabilistic encoder `mu/log_var` | PEARL 核心 |
| KL regularization | PEARL 核心 |
| CUDA PyTorch environment | 必要 |
| diagnostics | 可保留但不進主 loss |

## 要移除或關閉的部分

| 目前設計 | 原因 |
| --- | --- |
| true latent supervision | PEARL 不使用 task latent labels |
| oracle behavior cloning | PEARL actor 由 SAC objective 訓練 |
| noisy-oracle bootstrap | 會污染 online RL setting |
| actor-z hand-crafted sensitivity regularizer | 非 PEARL 原論文 |
| analytic candidate posterior | 可保留為 baseline，但不屬於 neural PEARL |
| hand-engineered context summary as final encoder | 應改 transition encoder + aggregation |

## 下一版實作：`torch_pearl_faithful_experiment.py`

建議新增一個新檔，不再把所有功能塞進現有 runner。

### 1. Transition Encoder

輸入單筆 transition：

```text
[s, a, r, s']
```

輸出每筆 transition 的 Gaussian factor：

```text
mu_i, log_var_i
```

再用 product-of-Gaussians 或 permutation-invariant aggregation 得到：

```text
q(z|c)
```

### 2. Context Batch / RL Batch

每次更新對每個 task 分開抽：

```text
context batch: 用於 encoder 推論 z
RL batch: 用於 critic / actor update
```

兩者來自同一 task buffer，但不是同一批 transitions。

### 3. Critic Loss Drives Encoder

移除：

```text
MSE(mu, true_latent)
```

改成：

```text
critic_loss(Q(s, a, z), target_q) + beta * KL(q(z|c) || p(z))
```

這是回到 PEARL 的關鍵。

### 4. Actor Loss

移除 oracle BC：

```text
MSE(actor(s,z), oracle_action)
```

改成 SAC actor objective：

```text
E[alpha * log_pi(a|s,z) - Q(s,a,z)]
```

### 5. Data Collection

初始資料可以用 random policy warmup，但不要用 oracle。

建議流程：

1. 每個 train task 用 random policy 收集少量 replay。
2. 每個 meta-iteration：
   - sample train tasks
   - sample context from task buffer
   - sample z from encoder
   - actor 用 z 在該 task 收集新 data
   - data 加入該 task replay buffer
   - 用 context batch + RL batch 更新 critic、actor、encoder

### 6. Meta-Test Protocol

對 test task：

1. `c = empty`
2. `z ~ prior`
3. collect trajectory
4. `c = c union trajectory`
5. `z ~ q(z|c)`
6. repeat K times

記錄 K=0,1,2,3,5 return。

## 建議里程碑

### Milestone A：Faithful Skeleton

- transition encoder
- product/mean aggregation
- SAC actor/critic
- per-task replay
- 無 latent supervision
- 無 BC

成功標準：

- 程式能跑完 1 seed。
- loss 不 NaN。
- K=1 不要求馬上變好。

### Milestone B：Toy Task Learning

成功標準：

- 在 hidden goal + wind task 上，K=1 或 K=2 平均優於 K=0。
- 至少 3 seeds。

### Milestone C：Ablation

比較：

- with / without KL
- posterior sample vs mean
- random warmup amount
- context size
- latent dimension

### Milestone D：更接近論文 Benchmark

加入：

- sparse 2D navigation
- bimodal goal distribution
- OOD goal/wind split
- 最後才考慮 MuJoCo / Gymnasium tasks

## 學習順序建議

你目前在學 meta-RL，建議照這個順序理解：

1. MAML：adaptation 是 gradient update。
2. RL²：adaptation 是 recurrent hidden state。
3. PEARL：adaptation 是 probabilistic task inference。
4. PEARL-SAC：把 task inference 接到 off-policy actor-critic。
5. Sparse reward / exploration：posterior sampling 的價值才會更明顯。

## 一句話策略

接下來主線應該少加創新，多做 faithful reproduction。所有創新都放進 ablation 或 diagnostics，不要混進主方法。
