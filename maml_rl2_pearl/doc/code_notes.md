# 程式說明與註解策略

## 目前程式範圍

[../src/goal_navigation.py](../src/goal_navigation.py) 是第一版 scaffold，負責：

- 產生固定 seed 的 train、validation、test goal tasks。
- 定義 2D point navigation environment。
- 定義 trajectory 資料格式。
- 提供 K-shot evaluation protocol 的共同結構。
- 用簡單 baseline 做 smoke test，確認環境與評估流程可執行。

[../src/run_experiment.py](../src/run_experiment.py) 是第一次 lite 訓練/測試 runner，負責：

- 以 train split 建立 prior 或 candidate set。
- 以 validation split 的 K=2 return 選 hyperparameters。
- 在 test split 上輸出 K=0,1,2,3,5 adaptation curve。
- 將結果寫入 `experiments/first_run/results.json`、`curves.csv`、`summary.csv`。

[../src/torch_pearl_experiment.py](../src/torch_pearl_experiment.py) 是 PyTorch CUDA neural runner，負責：

- 訓練 context encoder 與 conditioned actor。
- 使用 `.venv-py312` 裡的 CUDA PyTorch。
- 在 `experiments/torch_pearl_cuda/` 保存結果。
- 作為後續 PEARL-SAC critic / replay buffer 的接入點。

[../src/torch_pearl_sac_experiment.py](../src/torch_pearl_sac_experiment.py) 是 PyTorch CUDA offline SAC-style runner，負責：

- 產生 scripted behavior replay dataset。
- 訓練 context encoder、stochastic actor、twin Q critic 與 target critic。
- 使用 SAC-style TD target 與 entropy actor objective。
- 加上 behavior-cloning regularizer，控制離線 actor-critic 的 distribution shift。

[../src/torch_pearl_online_sac_experiment.py](../src/torch_pearl_online_sac_experiment.py) 是 PyTorch CUDA online SAC-style runner，負責：

- 維護 per-task replay buffers。
- 使用目前 actor online 收集新資料。
- 從同一 task 的 replay 中取 context 與 RL batch。
- 作為真正 PEARL probabilistic encoder + KL regularization 的下一個接入點。

[../src/torch_pearl_probabilistic_experiment.py](../src/torch_pearl_probabilistic_experiment.py) 是 PyTorch CUDA probabilistic PEARL-style runner，負責：

- 將 context encoder 改為 `mu/log_var` posterior。
- 用 reparameterization trick sample z。
- 加入 KL regularization。
- 支援 posterior sample 與 posterior mean evaluation。
- 支援 KL annealing、posterior sample averaging、actor-z sensitivity regularization 與診斷 metrics。
- 支援 context/RL batch 分離、context subset sampling、posterior rollout ensemble。

[../src/torch_pearl_faithful_experiment.py](../src/torch_pearl_faithful_experiment.py) 是回到 PEARL 原論文路線的 runner，負責：

- 使用 transition set encoder，而不是手工 context summary。
- 使用 product-of-Gaussians posterior aggregation。
- 使用 Q / V / target V / policy conditioned on z。
- 用 critic loss + KL 訓練 encoder。
- 不使用 true latent supervision、oracle BC、probe bootstrap、actor-z regularizer。
- 輸出 `training_curve.csv`，用於追蹤 Q/V/policy/KL loss 與 replay growth。
- 輸出 `posterior_diagnostics.csv`，用於 meta-test 時觀察 posterior mean norm、posterior variance 與 KL。

[../src/torch_rl2_baseline_experiment.py](../src/torch_rl2_baseline_experiment.py) 是第一個 neural RL2 recurrent baseline runner，負責：

- 使用 GRU policy 表示 task-level recurrent memory。
- policy input 為 `(state, previous_action, previous_reward, done)`。
- hidden state 只在 task 之間 reset，meta-test 的 K episodes 之間會保留。
- 使用 on-policy REINFORCE + value baseline 訓練。
- 不使用 PEARL posterior、task latent label、oracle probe 或 hand-crafted context summary。
- 目前是輕量 baseline，不是完整 RL2-PPO/TRPO 復刻。

[../src/torch_rl2_ppo_experiment.py](../src/torch_rl2_ppo_experiment.py) 是 PPO-style neural RL2 recurrent baseline runner，負責：

- 使用同樣的 GRU recurrent adaptation 介面。
- 收集完整 task sequences，保留同一 task 內跨 episode 的 hidden state。
- 使用 clipped PPO objective、value loss 與 entropy bonus 更新 policy/value。
- 支援 reward scaling，避免 PPO value target 過大導致訓練不穩。
- 不使用 PEARL posterior、task latent label、oracle probe 或 hand-crafted context summary。
- 目前是較強的 RL2 baseline，但仍不是完整 RL2-TRPO 官方復刻。

[../src/torch_maml_baseline_experiment.py](../src/torch_maml_baseline_experiment.py) 是第一個 neural MAML policy-gradient baseline runner，負責：

- meta-learn 一個 Gaussian policy 初始化。
- 每個 task 先收 support episode，再做一個 differentiable policy-gradient inner update。
- 使用 adapted policy 的 query episode loss 做 meta update。
- meta-test 時依 K 值執行 K 次 support inner update，再用 deterministic query return 評估。
- 不使用 PEARL posterior、RL2 recurrent memory、task latent label 或 oracle probe。
- 目前是 MAML-PG baseline，不是完整 MAML-TRPO/PPO 復刻。

## 為什麼不是完整 MAML、RL²、PEARL

完整 MAML-PPO、RL²-PPO、PEARL-SAC 都需要訓練 loop、policy network、optimizer、batching、replay buffer 與多 seed 實驗管理。第一版學習的目標是先把比較問題與 protocol 固定，避免在還沒定義公平評估方式前就開始堆模型程式。

因此 Run 001 使用三個 lite proxy：

- MAML-lite：對 goal estimate 做 gradient-style adaptation。
- RL2-lite：用 explicit memory estimate 模擬 recurrent hidden state。
- PEARL-lite：用候選 latent goals 與 reward likelihood 做 posterior inference。

目前 scaffold 會把三種方法需要接入的地方用註解標出：

- MAML：在 support trajectories 後做 inner-loop update。
- RL²：在同一 task 內保留 recurrent state。
- PEARL：把 trajectory transitions 加入 context，再更新 posterior over z。

## 註解原則

- 註解說明「為什麼這樣設計」，不是重述 Python 語法。
- 在會影響公平比較的地方加明確註解，例如 task split、K-shot protocol、hidden state reset 時機。
- 在尚未實作完整演算法的地方清楚標示為 extension point，避免把 baseline 誤解成正式方法。
