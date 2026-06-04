# BTS PoC Review Notes

## 整體結論

目前 `bts-poc/` 已經到「**可交給本地 A6000 / DGX Spark 跑第一輪 PoC**」的程度。

在這個沙箱中我已驗證：

- 所有 Python 檔都通過 `py_compile`（語法正確）
- `data.generate` 能成功生成資料
- 生成出的歧義規格 support 統計合理（例如 tiny dataset 中 support counter 約 `{2: 86, 1: 71, 4: 43}`）
- train / eval / run 的檔案結構完整

無法在沙箱內完整 smoke test 訓練的唯一原因是：**這個環境沒有安裝 PyTorch**。這不是程式本身的 bug；你在本地 GPU 機器上安裝 `torch` 後即可跑。

---

## Blocking / 重要問題

### 1. `train.py` / `eval/phenomena.py` 沒在這裡做過真正 forward/backward
- 原因：沙箱缺 `torch`
- 風險：可能還有 runtime shape bug（靜態檢查抓不到）
- 對策：在 A6000 / DGX 上先跑最小 smoke test：
  - `python -m data.generate --out-dir data/tiny --n-episodes 200 ...`
  - `python train.py --train-jsonl data/tiny/train.jsonl --test-jsonl data/tiny/test.jsonl --epochs 1 --batch-size 32 --d-model 64 --layers 2`

### 2. oracle posterior 的收斂規則目前是「站到某個候選物件上才收斂」
- 這是刻意簡化的 PoC 設計，但意味著「現象 2：entropy 隨觀察下降」可能會比較晚才發生。
- 對策：如果第一版曲線太平，可在第二版把 `compatible_tasks_given_state()` 改成更細緻的規則，例如：
  - agent 接近某物件時先排除部分不相容任務
  - 只要物件進入可視範圍就更新 posterior

---

## Should Fix（建議很快補上）

### 3. phenomenon 3 目前需要 belief / single 兩個 checkpoint 都先訓好
- 已在 README 補上指令，但 `run.sh` 還沒有一鍵串起來。
- 對策：之後可加一個 `bash run.sh --compare` 模式。

### 4. `train.py` 的評估指標目前偏向 loss / entropy，沒有內建 rollout success
- 現象 3 的 success rate 是在 `eval/phenomena.py` 裡做 rollout，沒有在 train loop 每個 epoch 顯示。
- 對策：PoC 初版可以接受；若要更方便調參，之後在每個 epoch 也抽少量 rollout 計 success。 

### 5. `EpisodeDataset` 目前只吃 `steps`，沒有把 `final_obs` 放進模型訓練
- 這沒有 bug，只是目前訓練監督只來自每一步的 action 與 posterior。
- 對策：PoC 先不動；若後面想做更強的 belief 收斂，可用 `final_obs` 增加 auxiliary target。

---

## Nice to Have

### 6. 加 `requirements-cuda.txt` / conda 環境說明
目前只有 `requirements.txt`（torch + matplotlib）；若你要在 A6000 / DGX 上快速建環境，可補 CUDA 對應版本。

### 7. 加更多 seed / support 統計圖
目前資料生成合理，但可多畫一張「support 大小分布」圖，讓論文更好講。

---

## 建議的第一輪本地執行順序

1. 安裝 PyTorch
2. 生成 tiny dataset
3. 1 epoch smoke train（belief）
4. 1 epoch smoke train（single-point）
5. 跑 `eval/phenomena.py` 看三現象圖是否有方向性
6. 若方向對，再開大 `n_episodes` / `epochs`

---

## 一句話總結

**這套 PoC code 現在已經足夠讓你在本地 GPU 機器上做第一輪驗證。** 真正剩下的不確定性不是「會不會跑不起來」，而是「這個簡化版 oracle posterior 是否已足以讓現象 2 清楚出現」——但這是一個合理且可快速迭代的研究風險，不是工程阻塞。
