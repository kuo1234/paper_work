---
type: experiment-report
aliases:
  - "M1b-6 eval score debug handoff"
  - "3D-DA CALVIN score 偏低 debug"
tags:
  - 研究主線
  - CALVIN
  - 3D-Diffuser-Actor
  - GB10
  - eval
  - debug
  - handoff
summary: "M1b-6 接手紀錄：3D-DA ABC→D eval 分數遠低於論文（100-seq avg seq len ~0.57 vs 論文 ~2.6-3.0）。flags 已逐項對齊官方 test_trajectory_calvin.sh，無不符。剩餘嫌疑：FPS CPU fallback 等價性、partial-validation 完整性、checkpoint 載入。session 出問題需重開，此檔供下一 session 直接接 debug。"
---

# M1b-6：eval 分數遠低於論文 — debug 接手紀錄（2026-06-08）

> 上游：[[m1b_abcd_validation_eval_report]]、[[m1a_calvin_smoke_report]]
> 狀態：**debug 進行中，session 出問題需重開，本檔為接手點**
> 遠端：spark-3994 / GB10 / 容器 `bts_m1` / 3D-DA branch `bts-gb10-calvin-smoke` commit `0c6685b`

---

## 0. 一句話現況

pipeline 跑得通，但 **分數遠低於論文**。需釐清是「環境/patch 造成行為偏差」還是「eval 設定/資料不完整」。**這是 M1b-6 的唯一待解問題。**

---

## 1. 最新結果：100-sequence eval（old checkpoint）

```text
checkpoint: diffuser_actor_calvin.pth (released CALVIN old w/history)
dataset:    task_ABC_D/validation (partial extracted, ~28.8GB)
NUM_SEQUENCES=100, num_history=3
1/5 : 36.0% | 2/5 : 13.0% | 3/5 : 4.0% | 4/5 : 3.0% | 5/5 : 1.0%
avg sequence length ≈ 0.57
```

**對比論文 3D Diffuser Actor ABC→D ≈ 2.6–3.0（avg seq len）。** 差距巨大，不是統計雜訊能解釋的（100 seq 已足夠看出量級差）。

歷史小批量（同樣偏低，趨勢一致）：

```text
old 10-seq:        40 / 30 / 10 / 10 / 0
old 50-seq:        48 / 22 / 14 /  8 / 2
old 100-seq:       36 / 13 /  4 /  3 / 1   ← 樣本越大越低，符合「真的偏低」而非僥倖
no-history 50-seq: 48 / 20 / 12 /  6 / 4
```

---

## 2. 已排除 / 已確認的事

### 2.1 flags 已逐項對齊官方 `test_trajectory_calvin.sh` ✅
逐項比對，**完全一致**，無漏 flag：

```text
num_history            = 3
interpolation_length   = 20
fps_subsampling_factor = 3
embedding_dim          = 192
quaternion_format      = wxyz
rotation_parametrization = 6D
diffusion_timesteps    = 25
gripper_loc_bounds     = tasks/calvin_rel_traj_location_bounds_task_ABC_D.json
gripper_loc_bounds_buffer = 0.01
relative_action        = 1
use_instruction        = 1
lang_enhanced          = 1
action_dim             = 7
text_encoder           = clip, text_max_length 16
backbone               = clip
```

> 注意：官方 script 是 6 GPU × 跑滿 1000 seq；我這裡 1 GPU。GPU 數不應改變單一 rollout 行為，只影響吞吐 / 切分。但**尚未驗證「單 GPU 切分是否導致 sequence 取樣/順序錯位」**——這仍是嫌疑（見 §3）。

### 2.2 graphbolt warning 是 benign ✅
log 內 `graphbolt disabled` 警告與分數無關，dgl 只用 `farthest_point_sampler`。

### 2.3 pipeline 本身可跑完整 60-step rollout、彙總正確 ✅
M1a / M1b 都跑完，無 crash；task log 顯示確實在正式 validation 任務上。

### 2.4 100-seq eval 進程確認正常完成 ✅
PID 4128737 透過 nohup 跑完，非 crash；之前 SSH polling 斷線是本地連線問題，不是遠端中斷。

---

## 3. 剩餘嫌疑（依優先序，下一 session 從這裡接）

### 嫌疑 A：FPS CPU fallback 非等價（**最高優先**）
- aarch64 dgl wheel 的 `farthest_point_sampler` 只有 CPU 算子。
- patch（`encoder.py` run_fps）把 input 轉 `float64().cpu()` 算 FPS、再搬回 GPU。
- **風險點**：
  1. dtype 從 float32 → float64 再回來，FPS 取的點 index 可能與官方 CUDA float32 FPS **不同**（FPS 對數值很敏感，選點差一個就改變 scene token 子集）。
  2. CPU/GPU FPS 演算法的 tie-breaking / 起始點行為可能不同（起始 index 給 0，但浮點距離比較順序仍可能分歧）。
- **驗證方法**：
  - 拿同一個 batch 的 `context_features`，分別跑「CPU float64 FPS」vs「CPU float32 FPS」vs（若能）GPU FPS，比較 sampled_inds 是否一致 / 重疊率。
  - 若 index 差很多 → 高度可疑，需自實作與官方等價的 FPS（或編 dgl CUDA aarch64）。
  - 反向驗證：把 `fps_subsampling_factor` 設成不下采樣（或 npts 很小使 FPS 退化），看分數是否回升——若回升，幾乎坐實 FPS 是元兇。

### 嫌疑 B：partial-extracted validation 不完整 / 缺 metadata
- 我們只 partial extract validation（~28.8GB / 99k files），沒下全包。
- npz_count 顯示 99022（central directory 宣稱 99045，wrote=99010 skipped=35）——**有 35 個 skipped**，需確認 skip 的是不是必要檔。
- **風險點**：eval 用到的 `lang_annotations/auto_lang_ann.npy`、`statistics.yaml`、episode 連續性（task chaining 需要連續 episode 的初始狀態）若有缺漏，會讓某些 sequence 直接失敗 → 拉低平均。
- **驗證方法**：
  - 確認 skipped 35 檔清單（extractor log）；
  - 對照官方 validation 應有的 episode 數 / 命名連續性（ep_start_end_ids.npy 之類）；
  - 抽查 fail 的 sequence 是否集中在某些 episode 範圍（若集中 → 資料缺漏）。

### 嫌疑 C：checkpoint 載入未完全吻合
- 載入時若有 missing/unexpected keys 被靜默吞掉，會讓部分權重是隨機初始。
- **驗證方法**：在載 checkpoint 處 print `load_state_dict` 的回傳（missing_keys / unexpected_keys），確認為空。

### 嫌疑 D：單 GPU sequence 切分 / 取樣錯位
- 官方多 GPU；單 GPU 時 sequence 的選取若依賴 rank/world_size 切片，可能取到「非標準前 N 條」或順序錯。
- 但 §1 顯示 10/50/100 趨勢一致地低，較不像單純取樣偏差；列為較低優先。
- **驗證方法**：確認 evaluate_policy 取 sequence 的邏輯是否吃 `RANK/WORLD_SIZE`，單 GPU 是否取到正規 eval 序列。

### 嫌疑 E：relative_action / quaternion / loc_bounds 座標慣例
- 雖然 flag 對齊，但 partial-extract 用的 `statistics.yaml`（`--calvin_gripper_loc_bounds`）來自 validation 自身；要確認與官方訓練/eval 期望的 bounds 一致（座標係/單位）。
- **驗證方法**：比對 `statistics.yaml` 內容與官方 task_ABC_D 期望值。

---

## 4. 立即可跑的下一步（建議順序）

1. **驗 FPS 等價性**（嫌疑 A）：寫小腳本 dump 一個 batch 的 FPS index，比 CPU-fp64 vs CPU-fp32 vs（嘗試）GPU。最有可能是元兇。
2. **查 skipped 35 檔 + validation 完整性**（嫌疑 B）：確認沒缺關鍵 metadata / episode。
3. **print checkpoint load_state_dict 回傳**（嫌疑 C）：5 分鐘可排除。
4. 若以上都乾淨，再查 §3-D/E 與單 GPU 取樣。

> 經驗判斷：分數差到 ~0.57 vs ~2.8（差 ~5×），最可能是「scene token 取樣（FPS）被 patch 改壞」或「資料/權重有缺」這類**會系統性破壞行為**的因素，而非單一 flag 細節。優先 A、B、C。

---

## 5. 遠端狀態快照（重開 session 後接手用）

```text
host:       p76141495@192.168.65.11 (spark-3994, GB10, aarch64, CUDA13)
container:  bts_m1  (image bts_m1_env:latest)
3D-DA repo: ~/bts/3d_diffuser_actor  branch bts-gb10-calvin-smoke  commit 0c6685b
validation: ~/bts/calvin/dataset/task_ABC_D/validation  (partial extracted, tactile 已刪)
checkpoints:
  ~/bts/3d_diffuser_actor/train_logs/diffuser_actor_calvin.pth          (old w/history)
  ~/bts/3d_diffuser_actor/train_logs/diffuser_actor_calvin_nohistory.pth (no-history, num_history=1)
eval launcher: ~/bts/run_eval_abc100.sh  (NUM_SEQUENCES sed-edit, 其餘 flag 同 §M1a)
extractor:     ~/bts/extract_validation_stream.py
官方參考:      ~/bts/3d_diffuser_actor/scripts/test_trajectory_calvin.sh
```

關鍵 patch 位置（皆已 commit 在 branch 0c6685b）：

```text
diffuser_actor/utils/encoder.py        run_fps：FPS 在 CPU(float64) 算後搬回 GPU  ← 嫌疑 A
online_evaluation_calvin/evaluate_utils.py  prepare_visual_states：camera 依 class name 選
容器內 site-packages（非 repo，重建容器要重打）：
  /usr/local/lib/python3.12/dist-packages/pyhash.py            (FNV1-32 shim)
  /usr/local/lib/python3.12/dist-packages/dgl/graphbolt/__init__.py (load_graphbolt try/except)
```

---

## 6. 給下一 session 的一句話

**接 M1b-6：3D-DA ABC→D 100-seq 只有 avg seq len ~0.57（論文 ~2.8）。flags 已全對齊官方，問題不在 flag。先驗 FPS CPU fallback 等價性（嫌疑 A），再查 partial-validation 完整性（B）與 checkpoint load keys（C）。遠端容器 bts_m1、branch 0c6685b、validation 已就緒。**
