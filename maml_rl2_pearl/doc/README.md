# 第一版學習文件索引

本資料夾整理 `plan.md` 的第一版學習成果，目標是先把 MAML、RL²、PEARL 在 few-shot meta-RL 中的核心差異、第一個實驗設計、目前可驗證結果與過程問題固定下來。

## 文件

- [conclusion.md](conclusion.md)：第一版結論與三種方法的核心比較。
- [experiment_design.md](experiment_design.md)：2D Goal Navigation 實驗設定、測試 protocol 與指標。
- [results.md](results.md)：目前已完成的可驗證結果與尚未完成的正式訓練結果。
- [process_issues.md](process_issues.md)：學習與實作過程中需要注意的問題。
- [code_notes.md](code_notes.md)：目前程式 scaffold 的用途與註解策略。
- [experiment_run_001.md](experiment_run_001.md)：第一次 lite 訓練/測試紀錄。
- [experiment_run_002.md](experiment_run_002.md)：hidden goal + hidden wind 的進階 latent task 實驗。
- [experiment_run_003.md](experiment_run_003.md)：PyTorch CUDA PEARL-style neural encoder/actor 實驗。
- [experiment_run_004.md](experiment_run_004.md)：PyTorch CUDA PEARL-style offline SAC actor-critic 實驗。
- [experiment_run_005.md](experiment_run_005.md)：PyTorch CUDA online per-task replay SAC-style 實驗。
- [experiment_run_006.md](experiment_run_006.md)：probabilistic encoder + KL regularization 實驗。
- [experiment_run_007.md](experiment_run_007.md)：probabilistic diagnostics、KL annealing、actor-z sensitivity 實驗。
- [experiment_run_009.md](experiment_run_009.md)：context/RL batch 分離與 posterior ensemble evaluation 實驗。
- [experiment_run_010_faithful_pearl.md](experiment_run_010_faithful_pearl.md)：回到原論文架構的 PEARL faithful skeleton smoke。
- [experiment_run_011_faithful_pearl_long.md](experiment_run_011_faithful_pearl_long.md)：faithful PEARL 較長訓練與第一版 learning signal。
- [experiment_run_012_rl2_baseline.md](experiment_run_012_rl2_baseline.md)：第一個 neural RL2 recurrent baseline，供 faithful PEARL 初步比較。
- [experiment_run_013_maml_baseline.md](experiment_run_013_maml_baseline.md)：第一個 neural MAML-PG baseline，供 PEARL/RL2 初步比較。
- [experiment_run_015_rl2_ppo_baseline.md](experiment_run_015_rl2_ppo_baseline.md)：PPO-style RL2 recurrent baseline，提供較強 RL2 對照。
- [experiment_run_016_faithful_pearl_longer.md](experiment_run_016_faithful_pearl_longer.md)：faithful PEARL 加長訓練，不改架構檢查是否穩定改善。
- [experiment_run_017_pearl_posterior_diagnostics.md](experiment_run_017_pearl_posterior_diagnostics.md)：faithful PEARL posterior diagnostics，檢查 q(z|c) 是否真的使用 context。
- [innovation_notes.md](innovation_notes.md)：目前所有偏離原論文的設計、效果與未來研究價值。
- [faithful_pearl_roadmap.md](faithful_pearl_roadmap.md)：回到 PEARL 原論文路線的實作計畫。

## 第一版範圍

第一版先不做 zero-shot 主題，也不直接進入 MuJoCo benchmark。範圍限定在可解釋、可視覺化、方便對照 adaptation mechanism 的 2D Goal Navigation few-shot meta-RL。

目前新增的程式分成兩層：

- [../src/goal_navigation.py](../src/goal_navigation.py)：共同環境與 K-shot 評估協定。
- [../src/run_experiment.py](../src/run_experiment.py)：第一次 lite 訓練/測試 runner。
- [../src/latent_navigation.py](../src/latent_navigation.py)：hidden goal + hidden wind 任務環境。
- [../src/run_latent_experiment.py](../src/run_latent_experiment.py)：Run 002 進階 latent task runner。
- [../src/torch_pearl_experiment.py](../src/torch_pearl_experiment.py)：PyTorch CUDA neural PEARL-style runner。
- [../src/torch_pearl_sac_experiment.py](../src/torch_pearl_sac_experiment.py)：PyTorch CUDA offline SAC-style runner。
- [../src/torch_pearl_online_sac_experiment.py](../src/torch_pearl_online_sac_experiment.py)：PyTorch CUDA online per-task replay runner。
- [../src/torch_pearl_probabilistic_experiment.py](../src/torch_pearl_probabilistic_experiment.py)：PyTorch CUDA probabilistic PEARL-style runner。
- [../src/torch_pearl_faithful_experiment.py](../src/torch_pearl_faithful_experiment.py)：不含非論文技巧的 faithful PEARL runner。
- [../src/torch_rl2_baseline_experiment.py](../src/torch_rl2_baseline_experiment.py)：neural RL2 recurrent baseline runner。
- [../src/torch_maml_baseline_experiment.py](../src/torch_maml_baseline_experiment.py)：policy-gradient MAML baseline runner。
- [../src/torch_rl2_ppo_experiment.py](../src/torch_rl2_ppo_experiment.py)：PPO-style RL2 recurrent baseline runner。

Run 003 已加入 PyTorch CUDA 訓練路徑，但仍不是完整 PEARL-SAC。它先訓練 context encoder 與 conditioned actor，下一步才接 critic、entropy regularization 與 replay buffer。

Run 004 已加入 replay dataset、stochastic actor、twin Q critic、target critic 與 SAC-style TD update；目前仍是離線資料版本，尚未完成 online SAC collection。

Run 005 已加入 online actor collection 與 per-task replay buffer，是目前最接近 PEARL 訓練結構的版本；效果仍未追上 supervised Run 003。

Run 006 已加入 `mu/log_var` posterior、reparameterization sampling 與 KL regularization；目前 sample 評估不穩，mean 評估穩定但改善小。

Run 007/008 已加入診斷與 z-sensitivity regularizer，確認目前瓶頸不是單一正則可解，而是 context/RL batch 與 posterior/actor coupling 仍需重做。

Run 009 已分離 context/RL batch 並加入 rollout ensemble，降低 sampling noise；剩餘瓶頸集中在 actor 是否有效使用 z。

Run 010 新增 faithful PEARL skeleton，移除 true latent supervision、oracle BC、probe bootstrap 等非論文技巧；目前只完成 smoke，不代表論文復刻完成。

Run 011 對 faithful PEARL 做較長訓練，單一 seed 出現 K=1 改善，但跨 seed 平均尚未穩定優於 K=0。

Run 012 新增 neural RL2 recurrent baseline；目前小規模設定中 faithful PEARL Run 011 優於 RL2 Run 012，但仍缺 MAML 與更強 RL2-PPO/TRPO baseline，不能視為最終論文級結論。

Run 013 新增 neural MAML-PG baseline；目前同一 latent navigation task 上 PEARL Run 011 也優於 MAML-PG Run 013，但 MAML 尚未升級到 TRPO/PPO 級訓練。

Run 015 新增 PPO-style RL2 baseline；reward scaling 後 PPO 訓練穩定，但短訓練下仍未追上 PEARL Run 011，也未呈現穩定 K-shot recurrent adaptation。

Run 016 在不改 faithful PEARL 架構下增加訓練量；整體 return 優於 Run 011 並繼續優於 MAML/RL2 baselines，但 3-seed 平均的 K-shot posterior adaptation 仍不穩。

Run 017 新增 posterior diagnostics；結果顯示 q(z|c) 會隨 support context 明顯收斂，posterior variance 下降、KL 上升，但 return 仍可能退化，瓶頸更像 posterior-policy coupling，而不是 encoder 完全沒讀 context。
