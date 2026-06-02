# Experiment Run 006：Probabilistic Encoder + KL

## 目的

Run 006 把 Run 005 的 deterministic encoder 升級成 PEARL-like posterior：

```text
q(z|c) = N(mu(c), diag(exp(log_var(c))))
z = mu + eps * std
```

並加入 KL regularization：

```text
KL(q(z|c) || N(0, I))
```

## 指令

Posterior sampling：

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_probabilistic_experiment.py --seeds 7,13,23 --iterations 20 --updates-per-iteration 20 --batch-size 256 --output-dir experiments/torch_pearl_probabilistic --device cuda --posterior sample
```

Posterior mean：

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_probabilistic_experiment.py --seeds 7,13,23 --iterations 20 --updates-per-iteration 20 --batch-size 256 --output-dir experiments/torch_pearl_probabilistic_mean --device cuda --posterior mean
```

## 結果

| Evaluation | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| posterior sample | -49.795 ± 2.844 | -59.765 ± 14.382 | -58.895 ± 12.151 | -53.561 ± 8.118 | -56.329 ± 8.663 |
| posterior mean | -49.795 ± 2.844 | -49.484 ± 2.037 | -49.484 ± 2.037 | -49.484 ± 2.037 | -49.484 ± 2.037 |

## 結論

- Probabilistic encoder、reparameterization、KL regularization 都已接入 CUDA training path。
- posterior sampling 目前 variance 太大，平均表現變差。
- posterior mean 穩定，但 adaptation 幅度很小。
- 目前問題很可能是 actor 沒有充分利用 z，或 KL / critic / latent auxiliary loss 權重不平衡。

## 下一步

1. 記錄 `mu/log_var` 對 true latent 的 error。
2. 量測 actor 對 z 的 sensitivity。
3. 分離 context batch 與 RL batch，而不是同一 batch 同時提供 context 和 RL sample。
4. 調低 KL 或逐步 anneal KL。
5. 用多次 posterior samples 平均 action，降低 evaluation variance。
