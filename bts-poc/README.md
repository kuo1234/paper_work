# BTS PoC

Belief-from-Specification (BTS) 的最小可跑 PoC。

## v4 變更（eval-only 不對稱懲罰 + avg_return）

- **env 加「踩錯物件」不對稱懲罰**：rollout 時若 policy 在不確定下走到非 target 物件，給 `WRONG_OBJECT_REWARD` 並結束 episode。這是 **eval-only 的環境動態，不進訓練 loss**；expert 直達 target 不踩錯，故不影響既有資料與訓練。
- **Phenomenon 3 新增 `avg_return`**：`phenomenon3_belief_vs_single.json` 同時保留 `success_rate` 與新增 `belief/single_avg_return`、`success_gap`、`return_gap`，以 **`return_gap` 為主判準**（猜錯的代價放大後，belief-aware 與 single 的差距更易顯現）。
- 因 penalty 為 eval-only，**v4 不需重訓**，直接用 v3 checkpoint 重跑 eval。

## v3 變更（partial reveal + P2 metric）

- **hint 改 partial reveal**：hint tile 不再揭露完整 `hint_task`，改成隨機揭露 target 的**單一屬性**（color 或 shape）。揭露後候選集合通常從 ~4 縮到 ~2（仍是 2-peak 分布），逼 single-point baseline 在剩餘候選裡 argmax 賭一個，讓 belief-aware policy 有機會勝出。
  - `vectorize_obs()` 改為編碼「揭露屬性 + kind 指示」，不再帶完整 task one-hot（obs_dim 因此改變，須重新生成資料並重訓；`train.py` 會動態 infer 維度）。
  - `oracle_posterior()` 揭露後用屬性過濾候選，不直接收斂成單點。
- **Phenomenon 2 判讀修正**：從「first vs last 單格」改成「初始 vs 收斂段平均 drop + mean_curve 線性斜率」，避免尾段 survivorship 噪音誤判。`phenomenon2_entropy_decay.json` 現含 `per_episode_mean_drop`、`slope`、`supported`。

## 目標

用最小成本驗證三個現象：

1. 歧義規格 → 初始 belief entropy 更高
2. 隨觀察進來，belief entropy 下降
3. belief / risk-aware 版本在歧義規格下勝過單點 baseline

## 結構

```text
bts-poc/
  envs/gridworld.py        # 環境 + 歧義規格 + oracle posterior + BFS expert
  data/generate.py         # 產生 offline dataset (jsonl)
  models/transformer.py    # 小型 in-context transformer + belief head
  train.py                 # L_IC + L_belief 訓練
  eval/phenomena.py        # 量三現象 + 畫圖
  configs/poc.yaml         # 預設設定
  run.sh                   # belief / single-point baseline 一鍵跑
```

## 環境安裝

最小依賴：

```bash
pip install torch matplotlib
```

建議 Python 3.10+。

## 執行

### 1) belief 版本（主方法）

```bash
bash run.sh
```

### 2) 單點 baseline（模擬 T2DA 式單點）

```bash
bash run.sh --single
```

### 3) 比較現象 3（belief vs single-point）

先各自跑完 belief 與 single 兩個版本後：

```bash
python eval/phenomena.py \
  --ckpt runs/bts_poc_belief/best.pt \
  --baseline-ckpt runs/bts_poc_single/best.pt \
  --test-jsonl data/gridworld_mixed/test.jsonl \
  --out-dir runs/bts_poc_compare/figs
```

## 輸出

- `data/gridworld_mixed/`
  - `train.jsonl`
  - `test.jsonl`
  - `meta.json`
- `runs/bts_poc_belief/` 或 `runs/bts_poc_single/`
  - `best.pt`
  - `metrics.jsonl`
  - `figs/phenomenon1_entropy_gap.png`
  - `figs/phenomenon2_entropy_decay.png`

## 成功判準

若方法合理，應看到：

- `phenomenon1_entropy_gap.json` 裡 `ambiguous_mean > exact_mean`
- `phenomenon2_entropy_decay.png` 呈現遞減趨勢
- belief 版最終成功率 / action loss 優於 single-point baseline（這點目前需要額外做 rollout 對比，可在下一版補）

## 目前限制（有意簡化）

- 規格只做語言屬性向量，不用真的 LLM encoder
- 觀察只做低維 state，不用影像
- belief head 只做類別後驗，不做高斯/ensemble
- phenomena.py 目前實作現象 1、2；現象 3 的 rollout 對比下一版補

這些都是刻意為了先用最小成本驗證「歧義規格 → belief → 收斂」核心鏈。

## 下一步（若這版跑通）

1. 補現象 3：真正 rollout belief vs single-point 的成功率對比
2. 在 toy env 加圖片規格
3. 換成 CALVIN / 真 encoder（見 `idea/poc_plan.md`）
