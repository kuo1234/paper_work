# BTS PoC — 進度與接手紀錄（PROGRESS）

> 這份是 single source of truth。新 session 接手請**先讀這份**，再讀 `HANDOFF_FOR_CLAUDE_CODE.md` / `README.md` / `REVIEW_NOTES.md`。
> 最後更新：2026-06-04（v4 完成）

---

## 0. 一句話現況

三現象（P1/P2/P3）在 toy env 上**全部穩定成立**，核心假設鏈獲得支持。P3 在加入不對稱風險（v4）後從弱訊號變成中等訊號。**尚未上 CALVIN**（依設計，要先在 toy env 看到三現象才值得搬）。

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

## 2. 三現象目前數字（v4，最新）

來源：`runs/bts_poc_compare/figs/*.json`

| 現象 | 指標 | 結果 | 支持 |
|---|---|---|:--:|
| **P1** 歧義 spec → entropy 高 | exact `0.026` vs amb `1.074`，gap `+1.049` | 強 | ✅ |
| **P2** 觀察 → entropy 收斂 | drop `+0.467`，slope `−0.040`，曲線單調降 | 強 | ✅ |
| **P3** belief > single | success_gap `+0.029`；**return_gap `+0.054`**（主判準） | 中 | ✅ |

P3 細節（`phenomenon3_belief_vs_single.json`）：
- belief success `0.216` / single `0.187`
- belief avg_return `−0.389` / single `−0.444`
- `return_gap` ≈ 1.9× `success_gap` → penalty 確實放大了 belief 的價值

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

---

## 4. 目前程式狀態（檔案 → 角色）

- `envs/gridworld.py`：env + partial-reveal hint + oracle posterior + BFS expert + v4 不對稱懲罰
  - reward 常數：`STEP_REWARD=-0.01`、`SUCCESS_REWARD=1.0`、`WRONG_OBJECT_REWARD=-1.0`
  - `compatible_tasks_given_state`：揭露後用單一屬性過濾候選（→ 2-peak）
  - `_on_any_wrong_object`：v4 判斷踩錯
- `data/generate.py`：offline dataset；record 含 `hint_pos/hint_task/hint_attr_kind/hint_attr_value`
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
