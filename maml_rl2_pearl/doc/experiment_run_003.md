# Experiment Run 003：PyTorch CUDA PEARL-Style Neural Encoder

## 目的

Run 003 的目標是把專案推進到 PyTorch + CUDA，建立 PEARL-style neural path：

- context encoder：`c -> z`
- conditioned actor：`(state, z) -> action`
- CUDA training
- K-shot support/query evaluation

這仍不是完整 PEARL-SAC。它先用 supervised target 訓練 encoder 與 actor，下一步再接 SAC critic、entropy regularization 與 replay buffer。

## PyTorch / CUDA 環境

使用 Python 3.12 venv：

```powershell
rtk py -3.12 -m venv .venv-py312
rtk .\.venv-py312\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

驗證：

```text
torch 2.12.0+cu126
cuda build 12.6
cuda available True
device NVIDIA GeForce GTX 1060 6GB
cuda matmul ok (1024, 1024) cuda:0
```

官方依據：

- PyTorch local install：https://pytorch.org/get-started/locally/
- 官方驗證方式包含建立 tensor 與 `torch.cuda.is_available()`。

## 指令

```powershell
rtk .\.venv-py312\Scripts\python.exe -B src\torch_pearl_experiment.py --seeds 7,13,23 --epochs 60 --batch-size 128 --output-dir experiments/torch_pearl_cuda --device cuda
```

## 輸出

- `experiments/torch_pearl_cuda/results.json`
- `experiments/torch_pearl_cuda/summary.csv`

## 結果

| Method | K=0 | K=1 | K=2 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEARL-neural | -46.431 ± 0.696 | -36.771 ± 3.198 | -36.771 ± 3.198 | -36.771 ± 3.198 | -36.771 ± 3.198 |

## 結論

- PyTorch CUDA 可用，已能在 GTX 1060 上訓練 neural encoder/actor。
- Neural context encoder 的 K=1 adaptation 有改善，但幅度仍小。
- 目前 supervised imitation 不是論文 PEARL 的 critic-driven latent training，因此不能當作正式 PEARL 結果。

## 下一步

1. 新增 replay buffer。
2. 新增 Q-function / value function。
3. 將 encoder loss 改成 critic loss + KL regularization。
4. actor 改成 stochastic Gaussian policy。
5. 使用 SAC update 替代 supervised oracle action imitation。
