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

---

## 8. 第三輪：選項 2 = fp32 + TF32-off 重跑（2026-06-08）— **精度也排除**

在 GB10 上強制全程 fp32 重跑同 20 條 sequence，判別是否為 Blackwell 上的 fp16/TF32 數值問題。

patch（`/tmp/run_eval_fp32.sh`，跑完自動還原 source）：
```python
with torch.cuda.amp.autocast(enabled=False):          # 關 fp16 autocast
torch.backends.cuda.matmul.allow_tf32 = False         # 關 TF32 matmul
torch.backends.cudnn.allow_tf32 = False               # 關 cudnn TF32
torch.set_float32_matmul_precision('highest')
```

結果（同 20 條，apples-to-apples）：

| run | 設定 | sum/20 | avg |
|---|---|---|---|
| base3 | fp16 + TF32 on | 19 | **0.95** |
| **fp32** | **autocast off + TF32 off** | **18** | **0.90** |
| fps1 | fp16, fps_factor=1 | 11 | 0.55 |

→ **fp32 沒有回升（0.95→0.90，noise 等級，per-seq pattern 幾乎一致）。精度 / autocast / TF32 不是元兇。**

**推論收窄**：三個同 subset 數據點 fp16=0.95 / fp32=0.90 / fps1=0.55，前兩者穩定 ~0.9 且 robust to precision。若仍屬硬體類（#102 的 V100-fail/4090-pass），機制**不是精度**，而是某 op 在 Blackwell/aarch64 上算錯（attention / scatter-gather / pytorch3d 旋轉 / dgl kernel），或根本不是硬體。

**最決定性的剩餘實驗 = §7.4 選項 1：主流 GPU(4090/A100/x86 CUDA) 對照**。fp32 已排除精度後，GB10 上要再判別只能逐 op 比對參考卡，本質就等同跨硬體對照。建議優先安排一張 x86 CUDA 卡跑同 checkpoint + 同 20 條，一刀切開「GB10 硬體 vs 其他」。

遠端產物：`/tmp/run_eval_fp32.sh`、`/workspace/bts/eval_fp32_run1_logs/result.txt`（sum18）。

---

## 9. 第四輪：diffusers scheduler 線（2026-06-08）— **clip_sample 初看有效，但 100-seq 否決**

選項 1（主流 GPU）不可行（只有本地+GB10），改追硬體無關的 **diffusers 版本/scheduler** 線（符合 #102「reconfigure environment」）。

### 9.1 發現
- 裝的是 **diffusers 0.37.1，repo setup.py 不 pin**。repo 最後 commit **2024-08-17**，era-correct 應是 0.27–0.30。
- repo `DDPMScheduler(...)` **沒設 clip_sample → 吃 default `True`**：每個去噪步把預測 x0 clip 到 [−1,1]。
- hypothesis：position 有 `normalize_pos` 到 [−1,1]（clip 大致無害），但 **rotation 6D（idx 3:9）沒 normalize**，去噪中間 x0 若超出 [−1,1] 會被 clip → 扭曲旋轉。

### 9.2 clip_sample=False 實測：20-seq 假陽性，100-seq 否決
| run | N | 設定 | sum/N | avg |
|---|---:|---|---:|---:|
| base3 | 20 | clip_sample=True（default）| 19/20 | 0.95 |
| noclip | 20 | clip_sample=False | 23/20 | 1.15 |
| original | 100 | clip_sample=True（default）| 57/100 | 0.57 |
| **noclip100** | **100** | **clip_sample=False** | **45/100** | **0.45** |

20-seq 的 +21%（seq4 1→4、seq17 1→3）被 100-seq 誠實比較否決：**clip_sample=False 沒有回升，反而略低（0.57→0.45）**。結論：**clip_sample 不是解法；20-seq 是樣本噪聲**。

### 9.3 純降 diffusers 版本行不通 / 目前未找到 scheduler 主因
- **scheduler 數學 0.30→0.37 完全相同**（timesteps[24..0] / betas / alphas_cumprod / clip_sample=True / timestep_spacing=leading 全一致）。
- era-correct **≤0.27 在現代 huggingface_hub(0.36) 下 import 失敗**（`cached_download` 已移除）→ 要降 diffusers 得連 hub 一起降，風險高（會動到 transformers/CLIP），**未做**。已把 diffusers 還原 0.37.1。
- 結論：diffusers scheduler 線目前**未找到主因**；clip_sample 已排除。

### 9.4 pyhash shim 已驗正確
shim 的 `fnv1_32`（mul-then-xor）對齊官方 FNV-1 32-bit 測試向量（'a'=0x050c5d7e 等全 match）。即使與真 pyhash 有別，scene shuffle 只換兩個 table-block 位置、產生「不同但合法」的 episode（oracle 看實際 sim state），**非掉分機制**。排除。

### 9.5 CPU vs GPU 單步對照：模型 inference / kernel bug 也排除
在沒有第二張 GPU 的情況下，做 GB10-only 最決定性測試：**同一個真實 eval obs + 同一 checkpoint + 同一 CLIP embedding + 同一固定 diffusion initial noise + 每步 variance noise**，CPU vs GPU(fp32/TF32-off) 比單次 `model.step`。

結果：
- raw policy trajectory：**max_abs=9.6e-8, mean_abs=2.1e-8**。
- full eval postprocess（quat→Euler + relative_to_absolute；含 pytorch3d transform）後 absolute action：**max_abs=1.9e-7**。

→ **Blackwell/aarch64 forward kernel / attention / pytorch3d / dgl gather 算錯基本排除**。模型對同一 obs 的輸出在 CPU/GPU 一致。

### 9.6 下一步（目前最合理）
既然模型 inference/device 數值已排除，低分若不是 checkpoint 本身，就更像 **CALVIN env / pybullet / action execution / task oracle / packaged dataset/env config 版本差異**，不是模型 forward。下一步建議：
1. 比對 CALVIN / pybullet / package versions 與官方/issue #102 成功環境；
2. 做 action execution / oracle sanity（固定一個已知 initial_state，檢查 env.step 控制模式、physics timestep、task oracle 對 state delta 的判定是否合理）；
3. 回 dataset/env config 線：partial-extract validation 與官方 packaged_ABC_D 在 `.hydra/merged_config.yaml`、calvin_env assets、scene/task config 上是否完全一致（注意 eval initial_state 來自 `get_sequences()`，但 env config/asset 仍來自 validation）。

遠端產物：`/tmp/run_eval_noclip.sh`（patch DA 兩個 scheduler 加 clip_sample=False，跑完還原 source）、`/workspace/bts/eval_noclip_logs/`（N=20 sum23）、`/workspace/bts/eval_noclip100_logs/`（N=100 sum45）、`/tmp/sched_probe.py`、`/tmp/pyhash_check.py`、`/tmp/fnv_variants.py`、`/tmp/cpu_gpu_step_compare.py`、`/tmp/cpu_gpu_full_step_compare.py`。

---

## 10. 第五輪：沒有第二台機器下的剩餘 sanity（2026-06-08）

### 10.1 `calvin_env main` 對照：排除 env commit 差異
3D-DA README 明確要求：`cd calvin_env; git checkout main`。遠端原本是 detached `1431a46`，本地 `main/origin/main` 是 `797142c fix bug in button during rollouts`，中間含幾個看似相關 commit：
- `27a27a2 fix bug with actions being modified after step`
- `6e7ceaa fix bug with wrong object sizes after changing a scene`
- `5a0eb8a add control in joint space`
- `797142c fix bug in button during rollouts`

做法：`/tmp/run_eval_envmain.sh` 暫時 checkout `calvin_env main`，跑同 20 條 base eval，再 trap 還原 `1431a46`。

結果：**逐 sequence 完全等於 base3**（sum19/20；`0 0, 1 5, 2 0, ...` 一模一樣）。→ `calvin_env` commit / action execution 差異**不是主因**。

### 10.2 language tokenization：排除 text length / truncation 問題
實際 eval wrapper (`DiffusionModel.encode_instruction`) 設 `tokenizer.model_max_length=16` 後，embedding shape 是 **(1, 16, 512)**。`new_playtable_validation.yaml` 34 個 annotation 經 CLIP tokenizer 都回傳 len=16，沒有 >16 長句/未 truncation 問題。

validation `auto_lang_ann.npy` 的 `language.emb` 是 **(1087,1,384)**（CALVIN 自己的 embedding），不是 3D-DA 使用的 CLIP 16×512；不能直接拿來比。3D-DA repo 的 `instructions/calvin_task_ABC_D/` 在遠端不存在，因此無法直接比訓練時預存 CLIP embedding。

### 10.3 DGL CPU FPS vs canonical FPS：排除 CPU FPS 實作錯誤
實作純 Python canonical farthest-point sampling（start_idx=0、逐步 argmax），對多個 shape/dtype 比 DGL CPU：
- fp32/fp64，B=1/3/2，N=128/256/1000，C=192/64/16，k=N//3
- **equal=True / overlap=1.000** 全部通過。

→ DGL CPU FPS 至少等於 canonical FPS；剩下的 CUDA-vs-CPU tie-breaking 不可直接測，但已被 factor=1 反向 rollout + canonical check 大幅降權。

### 10.4 stored frame vs live render：render/assets 大體正常
reset env 到 validation `.npz` 的 `robot_obs`/`scene_obs` 後 live render 與 stored frame 比：
- episode 0/1/40/219635：RGB/depth 幾乎 pixel/depth 對齊（static RGB mean_abs ~0.8–1.4，gripper RGB ~0.001；depth ~1e-4）。
- 中段 episode 1000/37682 差異較大，可能是 storage state 不能完整重建所有 dynamic state；但開頭與跨 split frame 能對齊，**不支持 render/asset 全局錯誤**。

### 10.5 目前狀態
在「只有本地 + GB10」限制下，已排掉：checkpoint、flags、validation missing、FPS dtype/canonical/factor=1、EGL depth/render、fp16/TF32、CPU-vs-GPU model inference、calvin_env main vs detached、language token length。

仍未能解釋 0.57 vs 論文 ~2.5。最誠實的剩餘假設：
1. **released checkpoint / public eval reproduction 本身高度環境敏感**（issue #102 的 V100 fail→4090 pass 仍是唯一外部 clue），但我們已無第二台機器可切開；
2. **我們仍漏掉某個 high-level rollout 差異**（pybullet 3.2.7 / Python 3.12 / control dynamics / task oracle / packaged dataset-env config），但目前沒有單一強嫌疑；
3. **比較對象可能不是這個 checkpoint/公開腳本的可重現結果**，需向作者 issue/成功環境索取精確 env lock 或 checkpoint checksum。

若繼續本機 debug，下一個最有資訊量的是：修 `/tmp/dump_first_traj.py`（補 `interpolation_length=20`）dump failure/success 的 19-waypoint absolute action path + gripper，判斷模型是否輸出「合理但 physics/oracle 失敗」還是「模型本身語義/動作就很差」。

---

## 11. 外部 GPT fix plan review + action replay sanity（2026-06-08）

使用者提供 `C:\Users\kuo\Downloads\3d_da_calvin_low_score_fix_plan.md`，重點建議轉向 CALVIN env / controller / pybullet / metadata。審稿結論：**大方向對，但前兩個高優先修法需降權**。

### 11.1 `scene_info` 修補包：不應視為高機率主因
遠端 read-only 檢查：
```text
validation/scene_info.npy: 不存在
scene_info_fix/task_ABC_D_scene_info.zip: bytes=525, names=['training/scene_info.npy']
```

3D-DA online eval 的 `scene_info` 來源是 runtime：
```python
env = hydra.utils.instantiate(render_conf.env, ..., use_scene_info=True)
start_info = env.get_info()   # env.scene.get_info()
```
並不是讀 `validation/scene_info.npy`。官方 fix zip 只補 `training/scene_info.npy`，對目前 online eval 大概率**沒有影響**（除非有其他離線 training/data path）。

### 11.2 `use_nullspace` 已經是 true
```text
/workspace/bts/calvin/dataset/task_ABC_D/validation/.hydra/merged_config.yaml
85:  use_nullspace: true
```
所以 GPT plan 的「修 `use_nullspace: true`」已排除。

### 11.3 dataset action replay sanity：支持 env/action/controller 線，但需小心解讀
寫了 `/tmp/action_replay_sanity.py` 與 `/tmp/action_replay_modes.py`：
- 從 `auto_lang_ann.npy` 取前 10 個 unique task language windows；
- reset 到起始 frame 的 `robot_obs`/`scene_obs`；
- replay validation `.npz` 的 expert actions；
- 用 `new_playtable_tasks.yaml` task oracle 檢查 target task；
- 比較 live next obs vs stored next frame。

官方 replay convention 參考：`calvin_env/scripts/record_video_icra.py` 使用 `np.split(data['actions'], [3,6])` 作 absolute action；`reset_env_rendered_episode.py` 也有 `rel_actions` reset-every-32 variant。因此測了多種 mode：
```text
stored_start_end_hit: 1/10
abs action, reset at segment start: 3/10
abs action, reset every 32 frames: 4/10
rel action, reset at segment start: 3/10
rel action, reset every 32 frames: 4/10
```
成功例：open_drawer、turn_on_led、close_drawer，另 place_in_slider 在 reset32 下成功。多數 block/slider/lightbulb task 不 hit。

**判讀**：這是目前第一個直接支持「問題在 CALVIN env/action/controller/oracle/dataset protocol」的實驗：expert action replay 也不穩。但仍不能過度斷言 pybullet 壞，因為 `auto_lang_ann` frame range 與 task oracle 的 exact start/end 可能不完全對齊（stored start/end 只有 1/10 hit 也提示 annotation window 不是完美 oracle segment）。

### 11.4 更新後的下一步優先序
在沒有第二台 GPU 的限制下，下一步不再優先 scene_info/use_nullspace，而是：
1. **找官方 CALVIN replay/eval protocol 的正確 segment source**：不要只用 `auto_lang_ann.info.indx`，查官方 benchmark eval / dataset replay 是否有 `ep_start_end_ids`、`episode_lookup`、或 task-specific successful segments。
2. **跑官方 CALVIN baseline / scripted oracle sanity（若有 checkpoint/script）**：切分「CALVIN env 本身低」vs「3D-DA wrapper 低」。
3. **更嚴謹的 action replay**：用官方 script 的 chunking/prev_info 邏輯，輸出 all-task oracle detections，而不是只檢查 annotation target task。
4. **dump 3D-DA predicted 19-waypoint action path + 對照 dataset expert action distribution**：看 policy 輸出是語義錯，還是合理 action 被 env/controller/oracle 吃掉。

遠端產物：
```text
/tmp/action_replay_sanity.py
/tmp/action_replay_modes.py
/tmp/action_replay_sanity.out   (local tee copy in shell cwd if preserved)
/tmp/action_replay_modes.out    (local tee copy in shell cwd if preserved)
```

---

## 12. 第七輪：oracle all-task scan 修正 action replay 解讀（2026-06-08）

為避免把 `auto_lang_ann` window 誤當 oracle ground truth，寫 `/tmp/oracle_replay_scan.py`：對前 12 個 unique task annotation window 同時檢查：
1. stored state scan：reset 到 frame s 與 frame j，不 replay action，查 target / any task oracle；
2. abs replay：`np.split(data['actions'], [3,6])`；
3. rel replay：`data['rel_actions']`；
4. 記錄 first target hit 與 first any-task hit。

結果摘要：
```text
n=12
stored_target_hits: 1/12
stored_any_hits:    8/12
abs_target_hits:    3/12
abs_any_hits:       7/12
rel_target_hits:    3/12
rel_any_hits:       6/12
```

關鍵觀察：很多 annotation target 與 oracle actual hit **不一致**：
- target `turn_on_led`，stored state 偵測到 `turn_on_lightbulb`；abs/rel replay 才偵測到 `turn_on_led`。
- target `lift_blue_block_slider`，stored state 偵測到 `lift_pink_block_slider`。
- target `lift_pink_block_table`，stored/replay 偵測到 `lift_red_block_table`。
- target `turn_on_lightbulb`，stored state 偵測到 `turn_on_led`。
- target `turn_off_led`，stored state 偵測到 `turn_off_lightbulb`。

**修正解讀**：上一輪「expert action replay target success 只有 3–4/10」不能直接推論 env/controller 壞；主要原因是 `auto_lang_ann.language.task` / `info.indx` 不能直接當 task-oracle ground truth。`stored_any_hits=8/12` 且 `abs_any_hits=7/12` 反而顯示 env/action/oracle 能偵測不少任務，並非全局壞。

更重要：**3D-DA online eval 不使用 `auto_lang_ann.language.task`**。它用 `get_sequences()` 生成 canonical task names，再從 `calvin_models/conf/annotations/new_playtable_validation.yaml` 取 language instruction。因此 `auto_lang_ann` label/window mismatch 不是 3D-DA online eval 低分的直接 root cause。

更新後結論：
1. GPT plan 的「action replay」方向有價值，但必須使用官方 replay/benchmark segment，不可直接用 `auto_lang_ann` target 當 oracle truth。
2. 目前 action replay sanity **沒有證明 env/controller 壞**；它只證明 annotation windows/labels 與 task oracle 對齊很差（可能是 CALVIN dataset annotation 已知問題）。
3. 剩餘最合理下一步：dump 3D-DA online eval 的真實 predicted actions / videos / oracle traces，而不是從 `auto_lang_ann` replay 推論。具體：對 first 20 eval sequences 中成功(如 seq1=5)與失敗(如 seq0=0)各 dump 每個 subtask 的 language、19-waypoint absolute action、executed actions、robot/object state、oracle hit，判斷 policy 是不是一開始就輸出錯。

遠端產物：
```text
/tmp/oracle_replay_scan.py
oracle_replay_scan.out   (本地 working dir tee 檔，若未清除)
```

---

## 13. crash 後補驗：safe launcher + restore trap（2026-06-08）

使用者檢查前一個 smoke log 發現：只跑到 4 條 summary，沒有 final `Load 5/1000` / `EXIT`；本地 command 用 `tee | grep` 且沒有 `pipefail`，因此 local pipeline exit code 0 不能代表 lab eval 完整跑完。

### 13.1 restore 狀態
重開 session 後先查遠端容器：
```text
host/container: p76141495@192.168.65.11 / bts_m1
repo: /workspace/bts/3d_diffuser_actor
online_evaluation_calvin/evaluate_policy.py: NUM_SEQUENCES = 1000
running evaluate_policy/run_eval jobs: none
```

也就是 lab repo 已回到 canonical `NUM_SEQUENCES=1000`，沒有卡在前一輪 smoke 的 `5`。

### 13.2 robust launcher
建立 `/tmp/run_eval_safe.sh`，修正兩個前一輪包裝問題：
1. `set -Eeuo pipefail` 下使用 `PYTHONPATH="$(pwd):${PYTHONPATH:-}"`，避免 `PYTHONPATH` 未定義時中斷；
2. 對 `evaluate_policy.py` 做 backup 後設定 `trap restore EXIT INT TERM`，不論正常結束、error、interrupt、TERM 都還原；
3. 不再用 `grep` pipe 判斷成功，stdout/stderr 全量寫到 outer log，launcher 自己最後輸出 `EXIT=... RESTORED_NUM_SEQUENCES=...`。

### 13.3 safe N=4 實跑
用 safe launcher 跑：
```text
/tmp/run_eval_safe.sh 4 safe4_trap1 29566 > /workspace/bts/eval_safe4_trap1.outer.log 2>&1 &
```

結果完整跑完：
```text
summary lines: 5  # 4 條 per-sequence summary + 1 條 final summary
result.txt:
0 1
1 2
2 5
3 1
final: EXIT=0 TAG=safe4_trap1 N=4 RESTORED=1 RESTORED_NUM_SEQUENCES=1000
post-check evaluate_policy.py: NUM_SEQUENCES = 1000
```

最後 log 有 final reprint：
```text
Load 4/1000 episodes...
1/5 : 100.0% | 2/5 : 50.0% | 3/5 : 25.0% | 4/5 : 25.0% | 5/5 : 25.0% ||
```

**解讀**：這輪只驗證 launcher/restore/log 完整性，不解讀 N=4 score（sum=9/4=2.25，樣本太小且 first-4 偏幸運）。後續 lab eval 一律用 `/tmp/run_eval_safe.sh` 類型的 trap launcher，並以 outer log 的 `EXIT=` 與 `RESTORED_NUM_SEQUENCES=1000` 為完成條件，不再用 `tee | grep` 的 pipeline exit code。

遠端產物：
```text
/tmp/run_eval_safe.sh
/workspace/bts/eval_safe4_trap1.outer.log
/workspace/bts/eval_safe4_trap1_logs/result.txt
```


---

## 13. 最終收斂：停止 3D-DA closed-loop reproduction，全面轉向圖片路線（2026-06-08）

### 13.1 A6000 reference confirmation

使用 lab A6000（x86_64 / RTX A6000 46GB）做 bounded reference run。因 lab render 走 Mesa llvmpipe，速度很慢，但 N=10 已乾淨完成：

```text
A6000 final10 per-sequence:
0 1
1 0
2 4
3 0
4 2
5 0
6 0
7 0
8 1
9 2

summary:
1/5 : 50.0%
2/5 : 30.0%
3/5 : 10.0%
4/5 : 10.0%
5/5 : 0.0%
avg seq len = 1.0
```

這比 GB10 100-seq avg 0.57 略高，但沒有恢復論文級；而且和 GB10 早期 10-seq smoke 非常接近：

```text
GB10 old 10-seq: 40 / 30 / 10 / 10 / 0
A6000 N=10:      50 / 30 / 10 / 10 / 0
```

### 13.2 決策

因此不再把 root cause 歸因於 GB10/aarch64/Blackwell 特有 forward 錯誤。更合理的解釋是：

1. public checkpoint + public eval 的 reproducibility 本身不穩；或
2. CALVIN / pybullet / control / env package 版本組合仍有高層差異；或
3. 論文數字與 public checkpoint/script 並非可直接等同。

已做的 debug 已足以排除主流低階嫌疑。繼續深追 3D-DA closed-loop reproduction 的邊際價值低，會拖慢 BTS 主線。

**正式停止 M1b-6 3D-DA closed-loop reproduction debug。** 3D-DA 仍可作為參考，不再作為 BTS 下一階段必要 backbone。

### 13.3 新方向

BTS 全面轉向 **image/perception-side CALVIN**：

- 不再依賴 point cloud / DGL FPS / 3D-Diffuser-Actor closed-loop reproduction。
- 保留 BTS 核心 novelty：belief / object-attribute binding。
- 新主張收斂為：structured belief improves visual-language manipulation under attribute-object binding ambiguity。
- 下一步閱讀與設計重點：CALVIN image baselines、Diffusion Policy / ACT / VLA backbone、object-centric representation、attribute-object binding diagnostics。
