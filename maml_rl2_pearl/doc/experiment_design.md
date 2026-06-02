# 第一版實驗設計

## Environment

使用 2D Goal Navigation。

- state：agent 目前位置 `(x, y)`。
- action：每一步的位移 `(dx, dy)`，會被限制在固定最大步長內。
- task：一個隱藏 goal location `(gx, gy)`。
- horizon：50 steps。
- dense reward：`-distance(position, goal)`。

第一版先用 dense reward，因為 sparse reward 會把探索困難與 adaptation mechanism 混在一起，不利於初學階段判讀。

## Task Split

建議第一版固定同一批 task split：

| Split | 數量 | 用途 |
| --- | ---: | --- |
| train | 40 | meta-training |
| validation | 10 | hyperparameter 選擇 |
| test | 20 | meta-testing |

三種方法必須使用同一組 train、validation、test goals。否則 adaptation curve 不能公平比較。

## K-Shot Evaluation Protocol

統一評估：

```text
K = 0, 1, 2, 3, 5
```

### MAML

- K=0：直接用 initialization policy 評估。
- K>0：先收集 K 條 support trajectories，做 inner-loop gradient update，再用 query trajectory 評估。

重點觀察：`theta -> theta'` 後 return 是否提升。

### RL²

- 每個 test task 一開始 reset RNN hidden state。
- 同一個 task 內連續跑 trajectories，不在 episode 之間 reset hidden state。
- 不做 test-time gradient update。

重點觀察：episode 2 是否比 episode 1 好，以及後續是否穩定。

### PEARL

- 一開始 context 為空，從 prior sample z。
- 每收集一條 trajectory，就把 transitions 加入 context。
- 用更新後的 posterior `q(z|c)` 產生下一條 trajectory。

重點觀察：context 增加後 z 是否更有用，return 是否提升。

## 必要指標

| 指標 | 公式或定義 | 用途 |
| --- | --- | --- |
| Adaptation speed | `R1 - R0` | 看一條 trajectory 後是否快速改善 |
| Final few-shot performance | `R5` | 看給足 5 條 trajectories 後的表現 |
| Stability | `mean ± std` over seeds | 避免只看單一 seed |
| Sample efficiency | K=2 test return vs meta-training steps | PEARL 是否真的更省 samples |

## 第一版最低可交付結果

第一版正式實驗最少應產出兩張圖：

1. Adaptation curve：x 軸為 K，y 軸為 average test return。
2. Sample efficiency curve：x 軸為 meta-training environment steps，y 軸為 K=2 average test return。

目前程式 scaffold 先提供共同環境與 protocol 介面，後續 MAML、RL²、PEARL 訓練器必須接到相同介面，避免每個方法各自定義測試方式。

## Run 001 Lite 設計

第一次訓練/測試先採用 lite proxy，而不是正式深度 RL：

| Method | 對應概念 | 第一版實作 |
| --- | --- | --- |
| MAML-lite | gradient-based adaptation | 學 train goal mean 作初始化，support 後用 reward-distance loss 對 goal estimate 做 gradient descent |
| RL2-lite | recurrent memory adaptation | 學 train goal mean 作初始 memory，support 後用 exponential memory update 累積 inferred goal |
| PEARL-lite | posterior context inference | 用 train goals + grid 作 latent candidates，根據 context reward likelihood 算 posterior mean |

三個 lite 方法在 support trajectory 使用固定 probe pattern 收集資訊，在 query trajectory 使用目前 goal estimate 控制。這個設計是為了讓 dense reward 能提供足夠 task inference 訊號；否則純 exploit 的 support rollout 很容易讓 MAML-lite / RL2-lite 沒有有效適應。

調參方式：

- 每個 seed 都用 train split 建立 prior。
- 用 validation split 的 K=2 return 選 hyperparameters。
- 最後只在 test split 回報 K=0,1,2,3,5 adaptation curve。
