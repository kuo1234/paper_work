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

---

## 7. 第二輪 debug 結果（2026-06-08，補記）— **所有 component 嫌疑全數排除，定錨為硬體/環境**

> 重要：遠端路徑修正為 **/workspace/bts/...**（不是 ~/bts）。容器 bts_m1 Up、GB10 idle。

### 7.1 逐項排除（皆遠端容器實測）

| 嫌疑 | 結論 | 證據 |
|---|---|---|
| **C** checkpoint 載入 | ✅ 排除 | 實跑 `load_state_dict`：**0 missing / 0 unexpected / 0 shape mismatch**。`module.` 7-char strip 正確、strict=True 本會 raise。|
| **B** validation 完整性 | ✅ 排除 | 1087 個 lang-task eval window **0 missing frame**。**且發現 eval 的 initial_state 來自 `get_sequences()` 程序生成、不讀 npz frame**（npz 只供 lang embedding），資料完整性對 rollout 行為基本無關。|
| **A** FPS（dtype）| ✅ 排除 | git diff 證明上游**原本就 `.to(float64)`**，patch 只改 device。實測 CPU FPS fp64-vs-fp32 index **overlap=1.000 且完全 deterministic**。CUDA FPS 在此 aarch64 wheel 不存在。|
| **A** FPS（反向 rollout 測）| ✅ 排除 | 同 20 條 sequence：factor=3 **avg 0.95** vs factor=1（不下採樣）**avg 0.55**（更糟非回升）。若 FPS 是元兇移除下採樣應回升。|
| **D** 單 GPU 取樣 | ✅ 排除 | WORLD_SIZE=1 → 乾淨取 sequence 0..N-1。|
| **F** GB10 EGL 幾何 | ✅ 排除 | live render depth 與**資料集存的 depth 幾乎一致**（static mean 4.278 vs 4.285）。point cloud extent 合理、finite。|

外加逐行確認對齊官方：scene config（calvin_scene_D / calvin_table_D）、proprio 8-dim(pos3+wxyz quat4+grip1) 與 `gripper[...,:7]`、quaternion_format=wxyz 與 `convert_rot` 內部慣例、relative_to_absolute、gripper_loc_bounds（JSON keyed by {A,B,C,D}→multi-task union）、act bounds clip、EP_LEN=60/EXECUTE_LEN=20、autocast=fp16。**無一不符。**

### 7.2 關鍵反轉
失敗**不在 chaining**：task-1 本身只有 ~36%（論文 ~93.8%）。代表「每一步 action 品質系統性偏掉」。no-history(nhist=1) 與 old(nhist=3) 分數幾乎相同也佐證 history 非關鍵。

### 7.3 決定性外部證據（GitHub issues）
- **#102「Error Result on Calvin dataset」**：完全相同症狀——分數極低、**換 no-history checkpoint 也一樣低**；maintainer 說 eval code「looks correct」。最終解法：**「V100 server 一直失敗，換到 4090 workstation 重配環境後成功重現」**＝GPU/環境問題，非 code/data/flag。
- **#94**：multi-task 權重測試需 `single_task_gripper_loc_bounds=0`（但此 flag 只在 RLBench eval；CALVIN 走 task=None→multi-task union，已正確）。
- **#66**：avg seq len 算法＝Σ(每條完成數)/N，與我們一致。

### 7.4 定錨結論 + 下一步
pipeline 對官方忠實、component 全清白 → 分數低**最可能是 GB10（Blackwell / aarch64 / CUDA13 / torch 2.7 nv25.04）這套 exotic 硬體在 diffusion inference 某 op 的 silent 數值問題**（與 #102 的 V100-fail / 4090-pass 同類）。

建議下一步（**不要再找 flag/data/code bug，已證實不在那**）：
1. **換主流 GPU 對照**（最直接）：把同 checkpoint + 同 dataset 放到 x86 CUDA（4090 / A100）跑同 20 條 sequence，看是否回到 ~2.x。能一刀切開「硬體 vs 其他」。
2. 若只能留在 GB10：逐 op 比對 inference 數值——關 autocast 改全程 fp32、`cudnn.benchmark=False` / `deterministic=True`、單步 dump diffusion 去噪中間值與一張參考卡比對；重點查 attention / FPS gather / pytorch3d 旋轉 op 在 Blackwell 上的數值。

### 7.5 第二輪用到的遠端產物
```text
/tmp/check_load.py     checkpoint load keys 驗證
/tmp/check_val.py      validation lang-window 完整性
/tmp/fps_probe.py      FPS 決定性 + fp64/fp32 overlap
/tmp/geom_probe.py     一步 depth/pcd/proprio 幾何 dump
/tmp/cmp_depth.py      live vs stored depth 比對
/tmp/run_eval_fpsexp.sh  參數化 launcher: <N> <fps_factor> <tag>
/workspace/bts/eval_fpsexp_base3_logs/result.txt  factor=3 N=20 (sum19)
/workspace/bts/eval_fpsexp_fps1_logs/result.txt   factor=1 N=20 (sum11)
```
