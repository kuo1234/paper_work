# 實驗結果紀錄

## 目前狀態

目前已完成從 lite scaffold 到 faithful PEARL runner 的逐步實作，並新增第一個 neural RL2 recurrent baseline 與 neural MAML-PG baseline。尚未完成正式 MAML-PPO/TRPO、RL2-PPO/TRPO 與 MuJoCo benchmark 級別的論文復刻。

因此目前不應該宣稱：

- MAML、RL2、PEARL 在論文 benchmark 上的正式深度 RL return 數值。
- PEARL 已在本 repo 中證明 sample efficiency 較好。
- sparse reward 或 MuJoCo benchmark 已完成。

## 已完成的可驗證結果

- 已從 `plan.md` 固定第一版研究問題。
- 已建立 `doc/` 文件，分開保存結論、實驗設計、結果紀錄、過程問題與程式說明。
- 已建立 2D Goal Navigation scaffold，讓後續方法共用同一個 task split 與 K-shot evaluation protocol。
- 已執行 scaffold smoke test，確認 task split、trajectory collection 與 K-shot evaluator 可以跑通。
- 已執行 Run 001 lite experiment，輸出位於 `experiments/first_run/`。
- 已執行 Run 002 latent goal + wind experiment，輸出位於 `experiments/latent_run/`。
- 已建立 Python 3.12 CUDA PyTorch 環境，並執行 Run 003 neural PEARL-style experiment，輸出位於 `experiments/torch_pearl_cuda/`。
- 已執行 Run 004 offline PEARL-SAC-style actor-critic experiment，輸出位於 `experiments/torch_pearl_sac/`。
- 已執行 Run 005 online per-task replay SAC-style experiment，輸出位於 `experiments/torch_pearl_online_sac/`。
- 已執行 Run 006 probabilistic encoder + KL experiment，輸出位於 `experiments/torch_pearl_probabilistic/` 與 `experiments/torch_pearl_probabilistic_mean/`。
- 已執行 Run 007/008 diagnostics experiment，輸出位於 `experiments/torch_pearl_probabilistic_run007/` 與 `experiments/torch_pearl_probabilistic_run008/`。
- 已執行 Run 009 context/RL batch separation experiment，輸出位於 `experiments/torch_pearl_probabilistic_run009/` 與 `experiments/torch_pearl_probabilistic_run009_mean/`。
- 已執行 Run 010 faithful PEARL smoke，輸出位於 `experiments/pearl_faithful_smoke/`。
- 已執行 Run 011 faithful PEARL 較長訓練，輸出位於 `experiments/pearl_faithful_run011/`。
- 已執行 Run 012 neural RL2 recurrent baseline，輸出位於 `experiments/rl2_baseline_run012/`。
- 已執行 Run 013 neural MAML-PG baseline，輸出位於 `experiments/maml_baseline_run013/`。
- 已執行 Run 015 PPO-style RL2 recurrent baseline，輸出位於 `experiments/rl2_ppo_run015/`。
- 已執行 Run 016 faithful PEARL 加長訓練，輸出位於 `experiments/pearl_faithful_run016/`。
- 已執行 Run 017 faithful PEARL posterior diagnostics，輸出位於 `experiments/pearl_faithful_diag_run017/`。

## Scaffold Smoke Test

執行指令：

```powershell
rtk python src/goal_navigation.py --seed 7
```

輸出：

```text
Task split sizes: train=40, validation=10, test=20
Random smoke baseline: K=0: -53.021, K=1: -39.924, K=2: -49.974, K=3: -50.152, K=5: -43.077
Goal-direction oracle: K=0: -3.281, K=1: -3.281, K=2: -3.281, K=3: -3.281, K=5: -3.281
```

判讀：

- random baseline 只用來確認程式流程可以執行，不代表 adaptation 能力。
- goal-direction oracle 直接知道 hidden goal，因此不是合法 meta-RL baseline，只能當作環境與 reward 的 sanity check。
- oracle return 明顯高於 random baseline，表示 dense reward 與動作更新方向符合預期。

## 待補正式結果表

正式深度 RL 訓練完成後，建議用下表記錄 adaptation curve。

| Method | Seed count | R0 | R1 | R2 | R3 | R5 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MAML-PPO/TRPO | TBD | TBD | TBD | TBD | TBD | TBD | inner-loop update 後評估 |
| RL2-PPO/TRPO | TBD | TBD | TBD | TBD | TBD | TBD | 同 task 內不 reset hidden state |
| PEARL-SAC faithful | partial | -46.296 | -46.508 | -47.489 | -47.284 | -46.677 | Run 016，same latent navigation task，不是 MuJoCo |
| RL2 recurrent lightweight | partial | -64.535 | -65.597 | -65.693 | -65.699 | -65.700 | Run 012，REINFORCE + value baseline，不是 PPO/TRPO |
| RL2-PPO lightweight | partial | -69.927 | -71.509 | -71.571 | -71.572 | -71.572 | Run 015，PPO-style recurrent baseline，短訓練 |
| MAML-PG lightweight | partial | -65.438 | -65.243 | -65.564 | -65.413 | -67.144 | Run 013，policy-gradient MAML，不是 TRPO/PPO |

## Run 001 Lite 結果

執行指令：

```powershell
rtk python -B src/run_experiment.py --seeds 7,13,23 --output-dir experiments/first_run
```

輸出檔：

- `experiments/first_run/results.json`
- `experiments/first_run/curves.csv`
- `experiments/first_run/summary.csv`

平均結果為 3 seeds 的 mean ± population std：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | -43.781 ± 7.232 | -40.335 ± 1.711 | -47.659 ± 2.680 | -47.612 ± 2.685 | -41.903 ± 4.341 |
| MAML-lite | -36.540 ± 3.060 | -9.575 ± 1.917 | -5.114 ± 0.891 | -3.811 ± 0.666 | -3.064 ± 0.529 |
| RL2-lite | -36.540 ± 3.060 | -15.264 ± 2.816 | -9.540 ± 1.882 | -6.604 ± 1.181 | -4.325 ± 0.758 |
| PEARL-lite | -36.540 ± 3.060 | -3.915 ± 0.754 | -3.915 ± 0.754 | -3.915 ± 0.754 | -3.915 ± 0.754 |
| Goal-direction oracle | -2.844 ± 0.497 | -2.844 ± 0.497 | -2.844 ± 0.497 | -2.844 ± 0.497 | -2.844 ± 0.497 |

初步判讀：

- MAML-lite 從 K=0 到 K=5 穩定改善，符合「support 後做 gradient-style adaptation」的預期。
- RL2-lite 也會改善，但速度慢於 MAML-lite，符合 memory update 逐步累積的設計。
- PEARL-lite 在 K=1 後快速改善，但 K>1 幾乎持平，表示第一條 probe trajectory 已足以讓 posterior 候選集中；這是 lite 設計的結果，不代表正式 PEARL 一定會如此。
- Goal-direction oracle 接近上界，MAML-lite K=5 已接近 oracle，表示 task inference 在 dense reward + probe 設定下非常容易。
- Random 沒有穩定 adaptation curve，只作為 sanity baseline。

## 待補 sample efficiency 表

| Method | Env steps | K=2 test return mean | K=2 test return std | Notes |
| --- | ---: | ---: | ---: | --- |
| MAML-PPO | TBD | TBD | TBD | on-policy |
| RL²-PPO | TBD | TBD | TBD | on-policy recurrent |
| PEARL-lite / PEARL-SAC | TBD | TBD | TBD | off-policy replay |

## Run 002 Latent Goal + Wind 結果

執行指令：

```powershell
rtk python -B src/run_latent_experiment.py --seeds 7,13,23 --output-dir experiments/latent_run
```

輸出檔：

- `experiments/latent_run/results.json`
- `experiments/latent_run/curves.csv`
- `experiments/latent_run/summary.csv`

平均結果為 3 seeds 的 mean ± population std：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | -70.768 ± 4.415 | -68.600 ± 4.119 | -81.856 ± 5.176 | -69.604 ± 6.264 | -75.929 ± 4.801 |
| MAML-latent | -45.373 ± 2.316 | -9.457 ± 1.186 | -6.187 ± 1.278 | -5.329 ± 1.186 | -4.979 ± 1.133 |
| RL2-latent | -45.373 ± 2.316 | -21.726 ± 1.091 | -15.359 ± 0.535 | -11.775 ± 0.768 | -8.144 ± 1.308 |
| PEARL-latent | -45.373 ± 2.316 | -7.217 ± 0.234 | -7.217 ± 0.234 | -7.217 ± 0.234 | -7.217 ± 0.234 |
| Latent oracle | -4.040 ± 0.311 | -4.040 ± 0.311 | -4.040 ± 0.311 | -4.040 ± 0.311 | -4.040 ± 0.311 |

初步判讀：

- hidden wind 讓 K=0 明顯變差，任務比 Run 001 更能測出 adaptation 差異。
- PEARL-latent 的 posterior 在 K=1 後快速找到可用 latent，因此早期 adaptation 最快。
- MAML-latent 持續從 K=1 改善到 K=5，最後比 PEARL-latent 更接近 oracle。
- RL2-latent 保持漸進改善，但在目前 explicit memory proxy 下比 MAML-latent 和 PEARL-latent 慢。
- Random 在 hidden dynamics 下很差，確認此任務不容易靠隨機行為解決。

## Run 003 PyTorch CUDA PEARL-Style Neural 結果

環境：

```text
Python: .venv-py312
torch: 2.12.0+cu126
CUDA build: 12.6
GPU: NVIDIA GeForce GTX 1060 6GB
torch.cuda.is_available(): True
```

安裝依據 PyTorch 官方 local install 頁面，Windows + pip + CUDA 版本由 selector 決定；官方也建議用 `torch.cuda.is_available()` 驗證 CUDA 是否可用。來源：https://pytorch.org/get-started/locally/

執行指令：

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_experiment.py --seeds 7,13,23 --epochs 60 --batch-size 128 --output-dir experiments/torch_pearl_cuda --device cuda
```

輸出檔：

- `experiments/torch_pearl_cuda/results.json`
- `experiments/torch_pearl_cuda/summary.csv`

平均結果為 3 seeds 的 mean ± population std：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-neural | -46.431 ± 0.696 | -36.771 ± 3.198 | -36.771 ± 3.198 | -36.771 ± 3.198 | -36.771 ± 3.198 |

初步判讀：

- CUDA neural training path 已可用，且 context 後 K=1 比 K=0 改善。
- 目前 neural PEARL-style 結果遠弱於 Run 002 analytic PEARL-latent，表示 encoder/actor supervised pretraining 還不足以取代 posterior likelihood inference。
- K=1 後持平，因為目前 evaluation 每次 support 使用同一種 deterministic probe context，encoder 聚合特徵重複後不會新增資訊。
- 這是 PEARL-SAC 前置工程，不是完整論文級結果。

## Run 004 PyTorch CUDA Offline SAC-Style 結果

Run 004 新增：

- replay dataset
- context encoder
- tanh-squashed Gaussian actor
- twin Q critic
- target critic
- SAC-style TD target
- entropy actor objective
- behavior-cloning regularizer

執行指令：

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_sac_experiment.py --seeds 7,13,23 --epochs 80 --batch-size 256 --output-dir experiments/torch_pearl_sac --device cuda
```

輸出檔：

- `experiments/torch_pearl_sac/results.json`
- `experiments/torch_pearl_sac/summary.csv`

平均結果為 3 seeds 的 mean ± population std：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-SAC-style | -55.276 ± 2.092 | -47.624 ± 8.152 | -47.624 ± 8.152 | -47.624 ± 8.152 | -47.624 ± 8.152 |

初步判讀：

- 結構上已比 Run 003 更接近 PEARL-SAC，因為已經有 critic、target critic、stochastic actor 與 replay update。
- 表現上 Run 004 目前弱於 Run 003。Run 003 的 K=1 是 `-36.771 ± 3.198`，Run 004 的 K=1 是 `-47.624 ± 8.152`。
- 主要問題是 offline actor-critic 的 Q extrapolation：actor 會被不準的 Q 推向 replay distribution 外。把 BC regularization 提高到 200 後才穩定到目前結果。
- 這是工程升級成功、演算法效果尚未成功的狀態。下一步需要 online SAC collection，或加入 CQL / TD3+BC 風格的 conservative objective。

## Run 005 PyTorch CUDA Online SAC-Style 結果

Run 005 新增：

- per-task replay buffer
- bootstrap probe / noisy-oracle episodes
- current actor online collection
- task-matched context sampling
- SAC-style actor/critic updates on growing replay

執行指令：

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_online_sac_experiment.py --seeds 7,13,23 --iterations 20 --updates-per-iteration 20 --batch-size 256 --output-dir experiments/torch_pearl_online_sac --device cuda
```

輸出檔：

- `experiments/torch_pearl_online_sac/results.json`
- `experiments/torch_pearl_online_sac/summary.csv`

平均結果為 3 seeds 的 mean ± population std：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-online-SAC-style | -55.246 ± 2.360 | -46.244 ± 1.804 | -46.244 ± 1.804 | -46.244 ± 1.804 | -46.244 ± 1.804 |

比較：

| Method | K=0 | K=1 | 判讀 |
| --- | ---: | ---: | --- |
| PEARL-neural Run 003 | -46.431 ± 0.696 | -36.771 ± 3.198 | supervised imitation，表現目前最好 |
| PEARL-SAC-style Run 004 | -55.276 ± 2.092 | -47.624 ± 8.152 | offline replay，variance 較大 |
| PEARL-online-SAC-style Run 005 | -55.246 ± 2.360 | -46.244 ± 1.804 | online replay 略好於 Run 004，較穩但仍弱 |

初步判讀：

- Run 005 的 online replay 確實比 Run 004 稍好，且 K=1 variance 降低。
- 但 Run 005 仍弱於 Run 003，代表 SAC objective / encoder coupling 還沒穩定。
- 單 seed 長訓練 `iterations=50, updates=30` 反而讓 K=1 退化，顯示問題不是訓練步數不足，而是 deterministic encoder + critic/actor coupling 不穩。
- 下一步應改 probabilistic encoder、分離 context batch 與 RL batch，並加入 KL regularization，而不是單純加長訓練。

## Run 006 Probabilistic Encoder + KL 結果

Run 006 新增：

- context encoder 輸出 `mu, log_var`
- reparameterization trick：`z = mu + eps * std`
- KL regularization 到 standard normal prior
- online collection 時 posterior sampling
- evaluation 可選 posterior `sample` 或 `mean`

Sample posterior 評估：

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_probabilistic_experiment.py --seeds 7,13,23 --iterations 20 --updates-per-iteration 20 --batch-size 256 --output-dir experiments/torch_pearl_probabilistic --device cuda --posterior sample
```

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-probabilistic sample | -49.795 ± 2.844 | -59.765 ± 14.382 | -58.895 ± 12.151 | -53.561 ± 8.118 | -56.329 ± 8.663 |

Posterior mean 評估：

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_probabilistic_experiment.py --seeds 7,13,23 --iterations 20 --updates-per-iteration 20 --batch-size 256 --output-dir experiments/torch_pearl_probabilistic_mean --device cuda --posterior mean
```

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-probabilistic mean | -49.795 ± 2.844 | -49.484 ± 2.037 | -49.484 ± 2.037 | -49.484 ± 2.037 | -49.484 ± 2.037 |

初步判讀：

- Probabilistic posterior 機制已加入，但效果目前不佳。
- posterior sampling variance 很大，K=1 平均反而變差。
- posterior mean 穩定，但 adaptation 幅度很小。
- 這表示目前 actor 對 z 的使用不足，或 KL / latent supervision / critic loss 權重還不平衡。
- 下一步不是再加 feature，而是診斷 z 是否真的承載 goal/wind：要輸出 latent prediction error、posterior variance、以及 actor 對 z 的 sensitivity。

## Run 007/008 Probabilistic Diagnostics 結果

Run 007 加入：

- KL annealing
- posterior 8 samples 平均 z
- latent MSE / posterior variance / actor-z sensitivity 診斷

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Run 007 PEARL-probabilistic | -48.865 ± 2.655 | -56.105 ± 5.953 | -53.986 ± 5.253 | -53.752 ± 2.579 | -52.814 ± 1.485 |

Run 008 額外加入 actor-z sensitivity regularizer：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Run 008 PEARL-probabilistic + z regularizer | -48.312 ± 2.837 | -57.077 ± 5.081 | -55.159 ± 2.734 | -52.397 ± 0.417 | -51.971 ± 1.998 |

診斷：

- Run 007 actor-z sensitivity 約 `0.0076` 到 `0.0161`，actor 幾乎不看 z。
- Run 008 sensitivity 提升到約 `0.023` 到 `0.028`，但 K=1 仍退化。
- 所以瓶頸不只是 actor 忽略 z，也包括 posterior/critic/actor coupling 和 context sampling 方式。

下一步：

- 真正分離 context batch 和 RL batch。
- 對同一 task 抽多個 context subsets。
- evaluation 改成 action ensemble 或多 posterior rollouts 平均。
- 追蹤 z error / actor sensitivity 隨訓練步數變化，而不只看最後一點。

## 後續方向調整

目前已將偏離原論文的設計整理到 `doc/innovation_notes.md`。這些設計適合未來做 ablation 或論文優化題材，但不應繼續混入 faithful PEARL 主線。

回到論文路線的計畫寫在 `doc/faithful_pearl_roadmap.md`。下一步應新增 `torch_pearl_faithful_experiment.py`，移除 true latent supervision、oracle BC、手工 z regularizer，改用 transition encoder、critic loss + KL、policy-collected replay。

## Run 010 Faithful PEARL Smoke

Run 010 是目前第一個不混入非論文技巧的 PEARL runner。

已移除：

- true latent supervision
- oracle behavior cloning
- scripted probe context
- analytic candidate posterior
- actor-z sensitivity regularizer
- hand-crafted context summary encoder

新增 / 保留：

- transition encoder
- product-of-Gaussians posterior
- per-task replay
- context batch / RL batch 分離
- Q / V / target V / policy conditioned on z
- critic loss + KL 訓練 encoder

Smoke 指令：

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_faithful_experiment.py --seeds 7,13,23 --iterations 2 --updates-per-iteration 3 --meta-batch 4 --context-size 16 --rl-batch-size 16 --warmup-episodes 1 --output-dir experiments/pearl_faithful_smoke --device cuda
```

結果：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful smoke | -69.808 ± 2.429 | -71.842 ± 4.146 | -68.782 ± 6.225 | -73.823 ± 2.013 | -71.095 ± 4.075 |

判讀：

- faithful skeleton 已能跑完 CUDA smoke。
- 尚未學出 adaptation；目前訓練量太小，不能用來判斷 PEARL 是否比 MAML/RL² 強。
- 下一步要跑長訓練 learning curve，並補 MAML/RL² faithful baselines。

## Run 011 Faithful PEARL Longer Training

Run 011 修正資料收集，使每個 collection interval 先收 prior-sampling episode，再收 posterior-sampling episode。這是回到 PEARL 的 prior exploration / posterior adaptation 流程。

指令：

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_faithful_experiment.py --seeds 7,13,23 --iterations 30 --updates-per-iteration 20 --meta-batch 8 --context-size 64 --rl-batch-size 64 --warmup-episodes 4 --collection-interval 1 --output-dir experiments/pearl_faithful_run011 --device cuda
```

結果：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful Run 011 | -47.377 ± 0.331 | -47.611 ± 3.286 | -49.453 ± 2.308 | -49.590 ± 3.354 | -49.360 ± 2.968 |

判讀：

- seed 7 有 K=1 改善：`-47.546 -> -43.442`。
- seed 13 / 23 adaptation 變差。
- 3 seeds 平均尚未證明 PEARL 有穩定 K-shot adaptation。
- 因此目前仍不能回答「PEARL 是否比 MAML / RL² 強」。

下一步：

- 加 learning curve logging。
- 增加 faithful PEARL 訓練量。
- 實作 / 接入 MAML 和 RL² faithful baselines。
- 同 split、同 K-shot protocol 下比較。

更新：`torch_pearl_faithful_experiment.py` 已加入 `training_curve.csv` 輸出，會記錄每個 iteration 的 `q_loss`、`v_loss`、`policy_loss`、`kl_loss` 與每 task 平均 transitions。下一次長訓練可直接用這個檔案判斷 learning stability。

## Run 012 RL2 Recurrent Baseline

Run 012 新增第一個 neural RL2-style recurrent baseline。它使用 GRU policy，輸入 `(state, previous_action, previous_reward, done)`，hidden state 只在 task 之間 reset，並在同一 task 的多個 episodes/K-shot evaluation 中保留 recurrent memory。

指令：

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_rl2_baseline_experiment.py --seeds 7,13,23 --iterations 50 --meta-batch 8 --episodes-per-task 3 --output-dir experiments/rl2_baseline_run012 --device cuda
```

結果：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RL2-recurrent Run 012 | -64.535 ± 6.769 | -65.597 ± 6.966 | -65.693 ± 6.957 | -65.699 ± 6.955 | -65.700 ± 6.955 |

和 faithful PEARL Run 011 的同任務比較：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful Run 011 | -47.377 ± 0.331 | -47.611 ± 3.286 | -49.453 ± 2.308 | -49.590 ± 3.354 | -49.360 ± 2.968 |
| RL2-recurrent Run 012 | -64.535 ± 6.769 | -65.597 ± 6.966 | -65.693 ± 6.957 | -65.699 ± 6.955 | -65.700 ± 6.955 |

判讀：

- 目前 faithful PEARL 明顯優於這個輕量 neural RL2 baseline。
- 這是第一個「PEARL 是否比 RL2 強」的實證訊號，但還不是最終結論。
- RL2 目前是 REINFORCE + value baseline，不是完整 RL2-PPO/TRPO。
- MAML-PG baseline 已補上，但 MAML-TRPO/PPO 級 baseline 仍未完成，所以研究目標尚未完成。

## Run 013 MAML-PG Baseline

Run 013 新增第一個 neural MAML-style baseline。它 meta-learn Gaussian policy initialization，每個 task 用 support episode 做一次 policy-gradient inner update，再用 adapted policy 的 query loss 做 meta update。

指令：

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_maml_baseline_experiment.py --seeds 7,13,23 --iterations 10 --meta-batch 4 --output-dir experiments/maml_baseline_run013 --device cuda
```

結果：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAML-PG Run 013 | -65.438 ± 10.321 | -65.243 ± 12.414 | -65.564 ± 17.272 | -65.413 ± 15.859 | -67.144 ± 16.314 |

目前三個 neural baseline 的同任務比較：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful Run 011 | -47.377 ± 0.331 | -47.611 ± 3.286 | -49.453 ± 2.308 | -49.590 ± 3.354 | -49.360 ± 2.968 |
| RL2-recurrent Run 012 | -64.535 ± 6.769 | -65.597 ± 6.966 | -65.693 ± 6.957 | -65.699 ± 6.955 | -65.700 ± 6.955 |
| MAML-PG Run 013 | -65.438 ± 10.321 | -65.243 ± 12.414 | -65.564 ± 17.272 | -65.413 ± 15.859 | -67.144 ± 16.314 |

判讀：

- faithful PEARL 目前在這個 latent navigation task 上優於 MAML-PG 與 RL2-recurrent。
- MAML-PG variance 很大，seed 23 有 K-shot 改善，但 seed 7 明顯退化。
- 這仍不是最終論文級結論，因為 MAML 和 RL2 仍不是 TRPO/PPO 級 baseline，且 benchmark 不是 MuJoCo。

## Run 015 RL2-PPO Baseline

Run 015 把 RL2 recurrent baseline 從 REINFORCE 升級為 clipped PPO-style training。它仍然只用 recurrent memory 做 adaptation，不使用 PEARL posterior、task label、oracle probe 或 hand-crafted context summary。

指令：

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_rl2_ppo_experiment.py --seeds 7,13,23 --iterations 15 --meta-batch 4 --episodes-per-task 3 --ppo-epochs 3 --reward-scale 0.05 --output-dir experiments/rl2_ppo_run015 --device cuda
```

結果：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RL2-PPO Run 015 | -69.927 ± 11.566 | -71.509 ± 13.394 | -71.571 ± 13.447 | -71.572 ± 13.449 | -71.572 ± 13.449 |

加入 RL2-PPO 後的同任務比較：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful Run 011 | -47.377 ± 0.331 | -47.611 ± 3.286 | -49.453 ± 2.308 | -49.590 ± 3.354 | -49.360 ± 2.968 |
| RL2-recurrent Run 012 | -64.535 ± 6.769 | -65.597 ± 6.966 | -65.693 ± 6.957 | -65.699 ± 6.955 | -65.700 ± 6.955 |
| RL2-PPO Run 015 | -69.927 ± 11.566 | -71.509 ± 13.394 | -71.571 ± 13.447 | -71.572 ± 13.449 | -71.572 ± 13.449 |
| MAML-PG Run 013 | -65.438 ± 10.321 | -65.243 ± 12.414 | -65.564 ± 17.272 | -65.413 ± 15.859 | -67.144 ± 16.314 |

判讀：

- faithful PEARL 目前仍是本地 latent navigation benchmark 上最好的 neural method。
- RL2-PPO 的 value loss 經 reward scaling 後穩定，但短訓練仍沒有學出有效 K-shot recurrent adaptation。
- 目前可說「在這個自建 benchmark 和目前訓練預算下，PEARL 是值得繼續優化的方向」。
- 尚不能說「已完整證明 PEARL 在論文 benchmark 上比 MAML/RL2 強」。

## Run 016 Faithful PEARL Longer Training

Run 016 不改 PEARL 架構，只增加訓練量與 warmup data。這是檢查「目前 PEARL 是否只是訓練不夠」的控制實驗。

指令：

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_faithful_experiment.py --seeds 7,13,23 --iterations 45 --updates-per-iteration 30 --meta-batch 8 --context-size 64 --rl-batch-size 64 --warmup-episodes 6 --collection-interval 1 --output-dir experiments/pearl_faithful_run016 --device cuda
```

結果：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful Run 016 | -46.296 ± 1.697 | -46.508 ± 3.807 | -47.489 ± 2.390 | -47.284 ± 2.233 | -46.677 ± 1.937 |

與 Run 011 相比：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful Run 011 | -47.377 ± 0.331 | -47.611 ± 3.286 | -49.453 ± 2.308 | -49.590 ± 3.354 | -49.360 ± 2.968 |
| PEARL-faithful Run 016 | -46.296 ± 1.697 | -46.508 ± 3.807 | -47.489 ± 2.390 | -47.284 ± 2.233 | -46.677 ± 1.937 |

目前完整本地比較：

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-faithful Run 016 | -46.296 ± 1.697 | -46.508 ± 3.807 | -47.489 ± 2.390 | -47.284 ± 2.233 | -46.677 ± 1.937 |
| RL2-recurrent Run 012 | -64.535 ± 6.769 | -65.597 ± 6.966 | -65.693 ± 6.957 | -65.699 ± 6.955 | -65.700 ± 6.955 |
| RL2-PPO Run 015 | -69.927 ± 11.566 | -71.509 ± 13.394 | -71.571 ± 13.447 | -71.572 ± 13.449 | -71.572 ± 13.449 |
| MAML-PG Run 013 | -65.438 ± 10.321 | -65.243 ± 12.414 | -65.564 ± 17.272 | -65.413 ± 15.859 | -67.144 ± 16.314 |

判讀：

- PEARL 仍是目前本地 benchmark 最強方法。
- 增加訓練量後，PEARL 的所有 K 平均 return 都比 Run 011 更好。
- 但 K-shot adaptation 本身仍不穩：seed 7 和 23 的 K=1 比 K=0 好，seed 13 退化。
- 因此「PEARL 適合繼續作為優化方向」目前有支持；「已完全復刻論文並證明優於 MAML/RL2」仍不成立。

## Run 017 PEARL Posterior Diagnostics

Run 017 新增 evaluation-only posterior diagnostics，不改 PEARL 訓練架構。

診斷平均值：

| K | Diagnostic return | posterior mean norm | posterior var mean | posterior KL |
| ---: | ---: | ---: | ---: | ---: |
| 0 | -54.872 | 0.000 | 1.00000 | 0.000 |
| 1 | -57.644 | 0.986 | 0.18854 | 2.464 |
| 2 | -56.467 | 0.981 | 0.09629 | 3.630 |
| 3 | -56.435 | 0.985 | 0.06378 | 4.379 |
| 5 | -55.132 | 1.012 | 0.03879 | 5.340 |

判讀：

- context encoder 不是完全沒作用；support context 會讓 posterior variance 快速下降，KL 上升。
- 但 posterior 變得更確定時，return 不一定改善，K=1 反而退化。
- 目前 faithful PEARL 的瓶頸更像 posterior-policy coupling：encoder 有更新 belief，但 policy/critic 沒有穩定把這個 belief 轉成更好的行為。
- 下一步應在原 PEARL 架構內調整 KL、entropy temperature、context size、update/data ratio，並拆分 prior-policy return 與 posterior-policy return。

## Run 009 Context/RL Batch 分離結果

Run 009 新增：

- context subset 與 RL transition batch 分開抽；
- evaluation 使用 posterior rollout ensemble；
- sample evaluation 與 posterior mean evaluation 各自保存。

| Evaluation | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sample ensemble | -49.681 ± 3.232 | -52.926 ± 4.879 | -53.122 ± 4.863 | -51.948 ± 3.864 | -53.841 ± 5.078 |
| posterior mean | -49.681 ± 3.232 | -51.127 ± 3.009 | -51.127 ± 3.009 | -51.127 ± 3.009 | -51.127 ± 3.009 |

判讀：

- 相比 Run 007/008，Run 009 的 K-shot 曲線更穩。
- posterior mean 只小幅退化，表示 sampling noise 已大幅降低。
- 但 adaptation 仍未轉正，代表核心瓶頸已轉移到 actor/encoder coupling。
- 下一步應改 actor architecture，讓 z 直接調制 policy hidden representation，例如 FiLM 或 gating。
