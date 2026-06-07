---
type: experiment-report
aliases:
  - "M1b ABC-D eval smoke"
  - "CALVIN validation partial extraction"
tags:
  - 研究主線
  - CALVIN
  - 3D-Diffuser-Actor
  - GB10
  - eval
summary: "M1b 以 HTTP Range 從 555GB task_ABC_D.zip 只抽取 validation (~28.8GB)，並在正式 task_ABC_D/validation 上跑 3D-DA eval smoke 成功。10-seq 結果：1/5=40%, 2/5=30%, 3/5=10%, 4/5=10%, 5/5=0%；50-seq 結果：1/5=48%, 2/5=22%, 3/5=14%, 4/5=8%, 5/5=2%。"
---

# M1b：正式 task_ABC_D validation partial extract + eval smoke（2026-06-07）

> 上游：[[m1a_calvin_smoke_report]]
> 目的：不直接下載 555GB 全包，先取得正式 ABC→D validation，跑小批量 eval smoke，檢查速度與穩定性。
> 遠端：spark-3994 / GB10 / 容器 `bts_m1` / 3D-DA branch `bts-gb10-calvin-smoke`

---

## 0. 結論 ✅

M1b 成功完成三件事：

1. **只從 555GB `task_ABC_D.zip` 抽出 validation**，不用下載整包：
   - ZIP server 支援 HTTP Range；
   - validation 在 ZIP 內是連續區段；
   - 抽出大小約 **28.79GB uncompressed**，共 **99,045 files**。

2. **正式 `task_ABC_D/validation` 10-sequence online eval smoke 跑完**：

```text
Load 10/1000 episodes...
1/5 : 40.0% | 2/5 : 30.0% | 3/5 : 10.0% | 4/5 : 10.0% | 5/5 : 0.0% ||
disconnecting id 0 from server
Destroy EGL OpenGL window.
```

3. **正式 `task_ABC_D/validation` 50-sequence eval smoke 也跑完**：

```text
Load 50/1000 episodes...
1/5 : 48.0% | 2/5 : 22.0% | 3/5 : 14.0% | 4/5 : 8.0% | 5/5 : 2.0% ||
disconnecting id 0 from server
Destroy EGL OpenGL window.
```

50-seq 比 10-seq 更穩，且全程無 crash；速度約 3.4–3.7 step/s。

---

## 1. 官方資料大小與 partial extraction 判斷

官方 ZIP header：

```text
task_D_D.zip      177.4 GB
task_ABC_D.zip    555.3 GB
task_ABCD_D.zip   704.0 GB
calvin_debug       1.3 GB
```

`task_ABC_D.zip` 支援 HTTP Range：

```text
Accept-Ranges: bytes
HTTP/1.1 206 Partial Content
Content-Range: bytes 0-99/555309812705
```

讀取 central directory（只抓尾端 + 218MB central directory）後發現：

```text
entries_parsed:        1,894,126
validation_entries:       99,055
validation compressed:    28.625 GB
validation uncompressed:  28.792 GB
training compressed:     526.272 GB
```

更重要的是 validation local file data 在 ZIP 裡是**連續區段**：

```text
offset_min_gb: 526.446
offset_max_gb: 555.080
span_gb:        28.634
merge_gap 0: chunks=1, range_gb=28.635, overfetch_gb=0.01
```

所以可用一個連續 Range stream 抽出 validation。

---

## 2. Partial extraction 實作結果

先用 subset extractor 驗證：成功抽出 5 個 `.npz` + metadata：

```text
statistics.yaml
.hydra/{config,hydra,overrides,merged_config}.yaml
lang_annotations/{embeddings.npy, auto_lang_ann.npy}
```

接著用 streaming extractor 抽完整 validation：

```text
entries 99045 compressed_gb 28.625 range_gb 28.635
progress 99045/99045 wrote=99010 skipped=35 uncompressed_gb=28.79 elapsed_min=52.9 rate_MBps=9.1
DONE /home/p76141495/bts/calvin/dataset/task_ABC_D
```

遠端位置：

```text
~/bts/calvin/dataset/task_ABC_D/validation/
```

驗證：

```text
npz_count 99022
camera_keys_before ['static', 'gripper', 'tactile']
camera_keys_after  ['static', 'gripper']
```

如 M1a，一樣刪掉 tactile camera（3D-DA 不用，避免 tacto/pyrender）。

---

## 3. 3D-DA source patches 已固化在遠端 repo

遠端 3D-DA repo：

```text
repo:   ~/bts/3d_diffuser_actor
branch: bts-gb10-calvin-smoke
commit: 0c6685b fix(calvin): make eval smoke work on GB10 aarch64
```

包含：

1. **camera order by class name**
   - 不假設 `env.cameras[0]=static, [1]=gripper`。

2. **DGL FPS CPU fallback**
   - aarch64 dgl wheel 的 `FarthestPointSampler` 無 CUDA 算子，FPS 在 CPU 算後搬回 GPU。

仍屬 hotfix / compatibility patch；M2 正式化要整理成乾淨 patchset 或 Dockerfile。

---

## 4. Eval smoke 結果（10-seq + 50-seq）

共同設定：

```text
checkpoint: diffuser_actor_calvin.pth (released CALVIN old w/history)
dataset:    task_ABC_D/validation (partial extracted)
GPU:        GB10, EGL render
```

10-sequence summary：

```text
Load 10/1000 episodes...
1/5 : 40.0% | 2/5 : 30.0% | 3/5 : 10.0% | 4/5 : 10.0% | 5/5 : 0.0% ||
```

50-sequence summary（更穩定）：

```text
Load 50/1000 episodes...
1/5 : 48.0% | 2/5 : 22.0% | 3/5 : 14.0% | 4/5 : 8.0% | 5/5 : 2.0% ||
```

部分 task log 顯示 eval 確實跑在正式 validation 任務上，例如：

```text
task: press the button to turn on the led light
task: go push the pink block left
task: lift the blue block from the sliding cabinet
task: push the sliding door to the left side
task: pull the handle to open the drawer
task: use the switch to turn on the light bulb
...
```

速度：單 sequence 60 steps 約 17 秒，約 **3.4–3.7 step/s**。50 sequences 約 16–20 分鐘量級完成。

---

## 5. 該如何解讀數字？

**只用來判斷 pipeline，不用來宣稱性能。**

原因：
- 50/1000 sequences 仍偏小，統計還不足以對論文數字；
- 使用 released `diffuser_actor_calvin.pth`（README 稱 old w/history），不是 2024-08 no-history 新權重；
- 有 compatibility patch（DGL FPS CPU fallback）但理論上只影響速度，不應改行為；
- official script 原本 6 GPU / 1000 sequences，此處 1 GPU / 50 sequences。

但此 smoke 成功代表：
**正式 ABC→D validation eval pipeline 已可在 GB10 上跑完。**

---

## 6. 下一步建議

### 6.1 先跑 100 sequences，再考慮 1000
不要直接 1000。已完成 50-seq；下一步建議：

- `NUM_SEQUENCES=100`：預估 ~30–40 min；
- 若穩定，再考慮完整 `NUM_SEQUENCES=1000`。

50-seq 結果已足以證明 pipeline 穩定；100-seq 用來得到較穩定的初步性能估計。

### 6.2 下載 / 測試新 no-history checkpoint
README 2024-08 提到新的 CALVIN no-history checkpoint：

```text
diffuser_actor_calvin_nohistory.pth
```

可能比目前 old w/history 權重更好，且 script 可能用 `train_trajectory_calvin_nohistory.sh` 對應不同參數（`num_history=0` 或類似）。建議在跑長 eval 前先比較兩個 checkpoint 的設定。

### 6.3 正式化環境
M1a/M1b 仍有手動 hotfix。M2 前應轉成：

- Dockerfile / container build script；
- reproducible patch set；
- 版本鎖定（transformers 4.44.2、torchdata 0.7.1、NGC pytorch 25.04 等）；
- remote ZIP partial extractor 保留成 script。

### 6.4 若要跑完整論文 eval
因 1000 sequences 估計數小時，且目前只 1 GPU。可先估算：

- 10 seq = 幾分鐘；
- 100 seq = ~10×；
- 1000 seq = ~100×，可能 4–6 小時（粗估）。

---

## 7. 核心 take-away

M1b 的關鍵不是 10-seq 分數，而是三個事實：

1. **不必下載 555GB 全包也能正式 eval**：validation 可 partial extract，實際 ~29GB。
2. **GB10 上的 CALVIN + 3D-DA 正式 validation eval 跑得通**。
3. **3D-DA 作為 BTS/CALVIN backbone 的可行性進一步升級**：已從 debug smoke 推進到正式 validation 小批量 smoke。
