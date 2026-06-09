# BTS PoC — 進度與接手紀錄（PROGRESS）

> 這份是 single source of truth。新 session 接手請**先讀這份**，再讀 `experiments/REPORT.md`（完整實驗認識）、`HANDOFF_FOR_CLAUDE_CODE.md` / `README.md`。
> 最後更新：2026-06-09（image/VLA route：controlled benchmark + LIBERO diagnostics scaffold）

---

## 0. 一句話現況

**主線已從舊 toy / 3D-DA CALVIN 轉到 image/VLA route。Controlled benchmark 證明 structured belief 修 generic policy 的 OOD wrong-object failure；OpenVLA-7B 真實 baseline 已 spark-only 在 GB10 跑通且忠實（LIBERO-Spatial 9/10），並發現綁定錯誤出現在 object-identity 軸（LIBERO-Object wrong-type 1/10），spatial 軸 OpenVLA 不 mis-bind。**

最新主要產物：

- `../idea/bts_image_vla_spec.md` — image/VLA 完整 spec（§34 spark-only 決策、§35 OpenVLA baseline+敘事 pivot）。
- `experiments/LIBERO_SPATIAL_OPENVLA_BASELINE_REPORT.md` — OpenVLA baseline + ambiguity/object probes。
- `experiments/CONTROLLED_IMAGE_BINDING_REPORT.md` — controlled benchmark 結果。
- `experiments/REPRODUCIBILITY.md`（§3b spark-native OpenVLA GPU eval）。

核心新結果：

```text
controlled: generic pixel_xy OOD success ≈ 0.244; structured BTS OOD = 1.000
OpenVLA spark-only GB10: load+predict 忠實，無 silent 失敗
  LIBERO-Spatial success 9/10=0.90, distractor-instance contact 0.00
  relation-stripped probe: success 0.90->0.20, distractor 仍 0.00（歧義=放棄抓取非綁錯）
  LIBERO-Object success 23/30=0.767, wrong-object-type 3/30=0.10
    cases: cream_cheese->tomato_sauce, butter->basket, chocolate_pudding->orange_juice
```

LIBERO scaffold：

```text
spark 原生 GPU venv ~/openvla-spark/.venv 跑 OpenVLA（torch 2.12+cu130, EGL render OK）
LIBERO Docker bts_libero 僅作 CPU diagnostics（torch.cuda=False）
rollout diagnostics 含 wrong-instance + wrong-object-type 指標
external policy adapter: fn(obs, context)->7D；openvla_policy_adapter 已實作忠實前處理
```

舊 toy conclusion（保留歷史）：

**P1、P2 真實且穩健（5 seeds 驗證）；P3 在當前 fully-observable toy env 下未通過統計檢驗。**

- P1（規格歧義 → belief 寬）、P2（觀察 → belief 收斂）：5 seeds 都穩，是真效應。
- P3（belief-aware policy > single-point）：v5 的單 seed `+0.092` 被 v6 多 seed 證明是**噪音**——dual return_gap `−0.014 ± 0.072`（5 seeds 正負亂跳），且 ablation 顯示第二 hint 無因果貢獻。
- 結論：核心推論鏈（spec=prior → observation=likelihood → belief 收斂）成立；但「belief 對**決策**有額外價值」在這個 fully-observable + BC + greedy 的 toy env **沒站住**（belief 對 single 是冗餘資訊，ctx 可繞過）。
- **尚未上 CALVIN**。下一步方向未定（見 `experiments/REPORT.md` 第 6 節）。

---

## 1. 執行環境（重要：用 SSH GPU 機器）

- **本地專案**：`C:\Users\kuo\Desktop\paperwork\bts-poc`
- **遠端 GPU**：`ssh p76141495@192.168.65.11`
  - hostname `spark-3994`，GPU `NVIDIA GB10`，Python 3.12，torch `2.12.0+cu130`，CUDA 可用
  - 遠端專案路徑：`/home/p76141495/bts-poc`
  - venv：`/home/p76141495/bts-poc/.venv`（已裝好 requirements）
- **工作流程**：本地改 code → `scp` 同步到遠端 → 在遠端 venv 跑 → `tar over ssh` 把 `runs/` 同步回本地
  - 本地沒有 `rsync`，用 `tar -czf - ... | ssh ... tar -xzf -`（或反向）
- 啟動 venv：`cd /home/p76141495/bts-poc && . .venv/bin/activate`

---

## 2. 三現象目前數字（v6 多 seed，最新且為準）

來源：`experiments/summary.json`（5 seeds × {single, dual}，每組每 seed 重訓）

| 現象 | single (n=5) | dual (n=5) | 結論 |
|---|---|---|---|
| **P1** 歧義→entropy 高 | `+1.023 ± 0.044` | `+0.986 ± 0.039` | ✅ 真實穩健 |
| **P2** slope | `−0.027 ± 0.012` | `−0.035 ± 0.010` | ✅ 真實穩健 |
| **P2** drop | `+0.441 ± 0.021` | `+0.535 ± 0.061` | ✅ 真實穩健 |
| **P3** return_gap | `+0.011 ± 0.097` | `−0.014 ± 0.072` | ❌ 均值≈0、跨 seed 變號 |

- dual return_gap 每 seed：`[−0.011, −0.104, +0.075, +0.033, −0.063]`（不同號）→ **無效應 + 噪音**。
- ablation：dual（−0.014）未優於 single（+0.011），**第二 hint 無因果貢獻**。
- **v5 的單 seed `+0.092` 已證明是 seed 噪音**（之前 v3→v4→v5 的上升趨勢是被單 seed 誤導）。

詳見 `experiments/REPORT.md`。

### 舊的單 seed 數字（v5，已被 v6 推翻，僅存查）
- v5 dual 單 seed：P3 return_gap `+0.092`（n_eval=120）— 不可信，未跨 seed。

歷代 P3 趨勢：v3 success_gap `+0.014` → v4 return_gap `+0.054` → **v5 return_gap `+0.092`**。

---

## 3. 演進史（每版改了什麼、為什麼）

### v1（原始交接版）
- Ambiguous-Spec GridWorld，spec 可精確/歧義，oracle posterior「踩到候選物件才收斂」。
- 結果：**P1 成立、P2/P3 不成立**。根因：observation 不提供 disambiguation 資訊，belief 沒理由收斂，single 不會輸。

### v2（hint-based，使用者改）
- 加 hint tile：歧義時 expert 先去 hint 再去 target；hint **直接揭露完整 `hint_task`**。
- 結果：**P1 成立、P2 本質成立但被 metric 誤判、P3 不成立**。根因：hint 給完整答案 → belief 變冗餘 → single 靠 ctx 就能贏。

### v3（partial reveal + 修 P2 metric）✅ 三現象首次全成立
- **hint 改 partial reveal**：隨機只揭露 color 或 shape **其一**，揭露後候選多停在 ~2（2-peak），不崩成單點。逼 single argmax 賭一個。
- `vectorize_obs` 移除完整 task one-hot，改編碼「揭露屬性 + kind 指示」(obs_dim 變 → 須重生資料 + 重訓，train.py 動態 infer 維度)。
- **P2 metric 改對**：從「first vs last 單格」（被尾段 survivorship 噪音誤導）改成 `per_episode_mean_drop` + 線性 `slope` + `supported`。
- 結果：P1 ✅、P2 ✅、P3 ✅ 但 gap 很小（success_gap `+0.014`）。

### v4（改法 B：eval-only 不對稱懲罰）✅ 把 P3 從弱訊號→中訊號
- **env.step 加「踩錯物件」懲罰**：rollout 時 policy 走到非 target 物件 → `WRONG_OBJECT_REWARD=-1.0` 並結束 episode。
- **eval-only**：reward 不進 loss（這是 offline imitation：BC + belief KL）。expert 直達 target 不踩錯，故不影響資料/訓練。**v4 不需重訓**，直接用 v3 checkpoint 重跑 eval。
- **P3 加 avg_return**：json 同時有 success_rate 與 avg_return，主判準改 `return_gap`。
- 結果：return_gap `+0.054` > success_gap `+0.029`，penalty 放大了差距。

### v5（雙 hint：讓 2-peak 可再縮小）✅ 把 P3 推到中強訊號
- 結構性診斷：v3/v4 單 hint 揭露單一屬性後殘留 2-peak **資訊上不可再縮小**（除非踩物件），belief 與 single 在「往哪走」無資訊差異 → P3 有天花板。
- **改成兩個 hint tile**，揭露互補屬性（一 color、一 shape）；揭露 0/1/2 個 → 候選 4/2/1（驗證：揭兩個後 122/122 全收斂到 1）。
- expert 改 `start → hint1 → hint2 → target`；`run.sh` horizon 15→20（雙 hint 路徑更長，避免 expert 截斷，從 53/122 降到 4/183）。
- `vectorize_obs` 擴成兩組 hint 區塊（抽出 `_encode_hint` helper），obs_dim 再變 → **須重生資料 + 重訓**。
- **沒動 `models/transformer.py`**：保留 single 吃 argmax one-hot 的弱點（這正是 gap 來源，改它就是 metric gaming）。
- 結果：return_gap `+0.092`（≈2.2× success_gap `+0.042`），P2 變兩階段收斂、更漂亮。
- ⚠️ **此 `+0.092` 後被 v6 證明是單 seed 噪音。**

### v6（多 seed + single/dual ablation）❌ P3 未通過統計檢驗（關鍵 de-risk）
- 動機：v5 是單 seed 且 train.py 根本沒設 torch seed（不可重現）；上 CALVIN 前先確認 P3 是真效應還是噪音。
- 加 `--seed`（train.py 可重現）、`n_hints∈{1,2}` ablation 開關（single/dual 共用模型架構，obs_dim 不變）、`experiments/run_matrix.sh` + `summarize.py`。
- 跑 5 seeds × {single, dual} = 20 訓練。
- 結果：**P1/P2 跨 seed 穩健成立；P3 dual return_gap `−0.014 ± 0.072`（跨 seed 變號），ablation 顯示第二 hint 無貢獻。v5 的 `+0.092` 是噪音。**
- 診斷：fully-observable + BC + greedy argmax 下，belief 對 single 是冗餘資訊（ctx 可繞過）→ P3 在此類 toy env 結構上難成立。
- 沒動 `models/transformer.py`、loss、v4/v5 機制；新實驗寫到 `data/exp/`、`runs/exp/`，舊 `runs/bts_poc_*` 保留。

---

## 4. 目前程式狀態（檔案 → 角色）

- `envs/gridworld.py`：env + 雙 hint 兩階段揭露 + oracle posterior + BFS expert + v4 不對稱懲罰
  - reward 常數：`STEP_REWARD=-0.01`、`SUCCESS_REWARD=1.0`、`WRONG_OBJECT_REWARD=-1.0`
  - hint1/hint2 各揭露互補屬性；`compatible_tasks_given_state`：依序用已揭露的每個 hint 屬性過濾候選（→ 4/2/1）
  - `expert_trajectory`：歧義時 `start→hint1→hint2→target`
  - `_on_any_wrong_object`：v4 判斷踩錯；`_encode_hint`：vectorize_obs 的單 hint 編碼 helper
- `data/generate.py`：offline dataset；record 含 `hint1_pos/kind/value` 與 `hint2_pos/kind/value`
- `models/transformer.py`：tiny in-context transformer + belief head + policy head + `SinglePointBaseline`
- `train.py`：`L_IC + λ·L_belief` teacher-forcing；`EpisodeDataset` 會保留 raw record（給 eval rollout 重建用）
- `eval/phenomena.py`：P1 entropy gap / P2 entropy decay metric / P3 `rollout_episode` 回 (success, return)

### 已修的 runtime bug（保留，勿回退）
1. `eval/phenomena.py` 開頭加 project root 到 `sys.path`（讓 `python eval/phenomena.py` 直接可跑）
2. `eval/phenomena.py` 從 `train.py` import `EpisodeDataset/collate_fn`（原本指向 data.generate 會 fail）
3. `EpisodeDataset.__getitem__` 用 `item.update(rec)` 保留原始 episode 欄位（P3 rollout 要重建 env）

---

## 5. 標準執行指令

```bash
# 進遠端、啟 venv
ssh p76141495@192.168.65.11
cd /home/p76141495/bts-poc && . .venv/bin/activate

# tiny smoke（驗證無 runtime/shape bug）
python -m data.generate --out-dir data/tiny --n-episodes 200 --size 7 --horizon 10 --n-objects 4 --spec-mode mixed --test-holdout-tasks green_triangle
python train.py --train-jsonl data/tiny/train.jsonl --test-jsonl data/tiny/test.jsonl --out-dir runs/smoke_belief --epochs 1 --batch-size 32 --d-model 64 --layers 2 --heads 4 --lambda-belief 0.5
python train.py --train-jsonl data/tiny/train.jsonl --test-jsonl data/tiny/test.jsonl --out-dir runs/smoke_single --epochs 1 --batch-size 32 --d-model 64 --layers 2 --heads 4 --lambda-belief 0.5 --single-point-baseline
python eval/phenomena.py --ckpt runs/smoke_belief/best.pt --baseline-ckpt runs/smoke_single/best.pt --test-jsonl data/tiny/test.jsonl --out-dir runs/compare/figs

# full run（會重生 data + 重訓兩版，各 15 epoch）
bash run.sh            # belief 版
bash run.sh --single   # single-point 版
python eval/phenomena.py --ckpt runs/bts_poc_belief/best.pt --baseline-ckpt runs/bts_poc_single/best.pt --test-jsonl data/gridworld_mixed/test.jsonl --out-dir runs/bts_poc_compare/figs

# 注意：若只改了 eval（如 v4 penalty，eval-only），不需重訓，直接重跑最後一行 eval。
```

成功判準：
- P1：`phenomenon1` 的 `ambiguous_mean > exact_mean`
- P2：`phenomenon2` 的 `per_episode_mean_drop > 0` 且 `slope < 0`
- P3：`phenomenon3` 的 `return_gap > 0`（主），`success_gap` 為輔

---

## 6. 下一步選項（尚未做）

要把 P3 從「中訊號」推到「強訊號」，需動到訓練或 rollout（比 v4 範圍大）：

- **改法 A'（expert 負面示範）**：讓部分 ambiguous episode 的 expert「不去 hint 直接猜」，讓 BC 真正學到「先去 hint」的價值差異。動到 `data/generate.py` + expert。
- **改法 risk-aware rollout**：belief 版在高不確定時主動選保守動作（不只 argmax）。動到 `eval` rollout 策略。
- **改法 C（不建議優先）**：policy head 不吃 ctx、改更依賴 belief。偏 metric gaming。

scope 原則：每次只動該動的，先規劃（plan mode）再做；不重寫設計；不輕易上 CALVIN。

---

## 7. 規劃文件位置

- 完整 v3/v4 設計與驗證計畫：`C:\Users\kuo\.claude\plans\federated-mixing-hejlsberg.md`
