# 接手點 — Expression-Decomposed CRS（更新 2026-06-18 深夜，全量三 split + 紅隊裁決）

> Decomposed CRS = 跨 split 全量 LTT 純帕累托改善 + subgroup 機制證據齊備。**紅隊裁決：站得住，但 matched 只能當 mechanism evidence，不是最終主證據；要成為正式第二主結果需補 robustness + strong baseline 等 6 項（§9）。** 下一步＝跑紅隊 §9 排序（robustness 跑中）。

## 一句話狀態（2026-06-18 深夜）

全量 dump 三 split 全完成（val 875 / testA 1243 / testB 974 真拆）。完整 LTT + matched + subgroup 已做：
- **LTT 主表**：val R1 0.191→0.165；testA R1 0.260→0.244 且 size 2.02→1.96（雙贏）；testB size 3.50→3.34。三 split n_feas 全健康(112/58/56=frozen)。
- **matched（mechanism evidence）**：三 split size~3 +19~23pp。
- **subgroup**：長句受益更大（機制）、雙目標甜蜜點、testA n_gt==1 負信號（待 §6 系統審計）。
- 子集所有瑕疵（R2 升、可行崩潰）證實為小樣本假象。

詳見 `vlm_as_decision_negative_result.md §4e`。**紅隊裁決見 `decomp_redteam_questions.md`（已含完整架構簡報 + 攻擊點 + 紅隊回覆）。**

## ★下一步：紅隊 §9 待補實驗（任務清單 #1-#7）★

claim 已降調（#1 done）。剩 6 項實驗，紅隊排序（1/3 是最硬防線）：
1. **random5 + image-disjoint robustness**（#2，跑中 `src/decomp_robust.py`）— 含 R_total（#5）
2. recall-size frontier 四指標 + calib/eval λ 分離（#3）`src/decomp_matched.py` 待擴
3. **strong full+NMS / top-K baseline**（#4）— 排除 threshold=0 稻草人
4. overall target failure R_total（#5，併入 #2）
5. 誠實成本表 VLM calls + detector forwards（#6）
6. 負增益 subgroup 系統失敗審計 50-100 例（#7）

**最終 claim（紅隊認可，禁用「decisive/決定性」）**：
> Candidate selection alone fails to close the oracle gap. Expression decomposition changes the candidate-pool construction process and improves the recall-size frontier for compositional multi-target queries. When wrapped by CRS/LTT, it can reduce answered-target FNR or set size while preserving no-target and deferral risk control.

## 重跑全量分析指令
```bash
ssh -i ~/nvsync.key p76141495@192.168.65.11 \
  'cd ~/selective-grounding && ./.venv/bin/python src/decomp_maintable.py'           # 三 split LTT 主表
ssh -i ~/nvsync.key p76141495@192.168.65.11 \
  'cd ~/selective-grounding && for sp in val testA testB; do ./.venv/bin/python src/decomp_matched.py $sp; done'  # matched
ssh -i ~/nvsync.key p76141495@192.168.65.11 \
  'cd ~/selective-grounding && ./.venv/bin/python src/decomp_robust.py'              # robustness + R_total (慢, ~30min+)
ssh -i ~/nvsync.key p76141495@192.168.65.11 \
  'cd ~/selective-grounding && for sp in val testA testB; do ./.venv/bin/python src/decomp_subgroup.py $sp; done'  # subgroup
```

## dump 檔案狀態
- 全量（現用）：`dump/gdino_gref_{val,testA,testB}_decomp.jsonl`（14229/19200/16063 行）
- 子集 pilot 備份：`dump/gdino_gref_{sp}_decomp_sub.jsonl`（勿刪）
- launcher：`run_dd2_full.sh`（全量，已跑完）

## 已驗證結果（2026-06-18，存 §4e）
- **matched oracle 三 split**（繞過 LTT，size~3）：val 0.70→0.90 / testA 0.74→0.89 / testB 0.60→0.85（+15~24pp，零例外）
- **LTT 主表**：val sz 3.24→3.23 R1 0.191→0.162 defer 0.42→0.30 n_feas 84（健康）；testA R1 0.260→0.189 n_feas=2；testB EMPTY
- 機制：per-part query 分數乾淨 → 同 size 留對框；val 三疑慮（CI 重疊/R2 反常/全域位移）已全解消

## 關鍵數字回顧（已驗證，存 vlm_as_decision_negative_result.md §4b/4c/4d/4e）
- 上界：完美分解 R1 = 0.006/0.011/0.019（val/testA/testB），L3=L2（定位不是瓶頸）
- 拆解品質：within±1 0.98、malformed 0、no-target 誤拆 0.000、single 誤拆 0.000
- e2e pilot（200 val）：full-expr R1 0.335 → decomp top-1 0.172（同 size）→ top-2 0.075（size 4.11）
- routing 權衡：v1 保守（誤拆 0，multi 拆 33%）vs v2 中性（multi 含並列拆 100% 但 no-target 誤拆 0.56）→ **用 v1**

## 所有腳本（spark `~/selective-grounding/`）
- dump：`decomp_dump3.py`（子集版，現用）、`decomp_dump2.py`（全量版）
- LTT 接線：`src/decomp_maintable.py`、`src/crs_protocol.py`（單一事實來源，勿改）
- 前置檢查：`dump/{decomp_ceiling,decomp_quality,routing_check,routing_split,routing_v2}.py`
- 啟動：`run_dd3.sh`

## 教訓（省得明天重踩）
- spark venv 是 uv 建的，**沒有 pip**，要裝套件用 `VIRTUAL_ENV=$PWD/.venv ~/.local/bin/uv pip install`。
- 背景啟動別用裸 `nohup ... &` 直接打 ssh（exit code 255、log 易被舊殘留誤導）；用 `run_*.sh` 包起來再 `bash run_*.sh` 最穩。
- GDINO 在 transformers 5.11 的 post_process 參數是 `threshold=`（非舊版 `box_threshold=`）。
- 確認 process 真死：`ps aux | grep X | grep -v grep | wc -l`，別只信 pgrep（會抓到自己的 ssh 指令字串）。
