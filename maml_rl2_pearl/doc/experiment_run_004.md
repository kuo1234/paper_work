# Experiment Run 004：PyTorch CUDA Offline PEARL-SAC-Style

## 目的

Run 004 把 Run 003 的 supervised neural encoder/actor 升級成 actor-critic 訓練結構：

- replay dataset
- context encoder
- stochastic Gaussian actor
- twin Q critic
- target critic
- SAC-style TD target
- entropy actor objective
- behavior-cloning regularization

這仍不是完整 PEARL-SAC，因為 replay buffer 目前由 scripted behavior policy 離線產生，還不是 SAC actor online collection。

## 指令

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_sac_experiment.py --seeds 7,13,23 --epochs 80 --batch-size 256 --output-dir experiments/torch_pearl_sac --device cuda
```

## 輸出

- `experiments/torch_pearl_sac/results.json`
- `experiments/torch_pearl_sac/summary.csv`

## 結果

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-SAC-style | -55.276 ± 2.092 | -47.624 ± 8.152 | -47.624 ± 8.152 | -47.624 ± 8.152 | -47.624 ± 8.152 |

## 結論

- CUDA actor-critic training path 可執行。
- K=1 比 K=0 改善，表示 context encoder 仍有提供 adaptation 訊號。
- 整體弱於 Run 003 supervised PEARL-neural，表示目前 offline SAC-style update 尚未學出好策略。

## 問題

離線 SAC 容易遇到 Q extrapolation。actor 在 replay buffer 支援不足的 action 區域被 Q 高估引導，導致 query policy 退化。Run 004 使用 `bc_weight=200` 才避免更嚴重崩壞。

## 下一步

1. 改成 online SAC：每輪用目前 actor 收集 replay。
2. 加入 conservative Q loss，降低 unseen action 的 Q。
3. 或先做 TD3+BC-style actor objective，穩定離線 actor update。
4. encoder loss 從 latent supervision 改成 critic-driven loss + KL regularization。
