# 接手點 — Expression-Decomposed CRS（更新 2026-06-18 晚，全量 dump 跑中）

> 方法層跨三 split 全 GO（matched +15~24pp）；LTT 小子集 testA n_feas=2 / testB EMPTY。**全量 dump 已背景跑中（threshold=0，三 split 串跑，~數十小時）**，完成後重跑 maintable，LTT 應有健康可行區間 → 寫進論文。

## 一句話狀態（2026-06-18 晚）

跨 split 確認跑完，分裂成兩個故事：
- **方法層（matched oracle，繞過 LTT）= 三 split 全 GO**：同 size recall val +20pp / testA +15pp / testB +24pp，零例外。核心主張坐實。
- **校準層（LTT 主表）小子集力不從心**：testA n_feas=2、testB EMPTY feasible。matched 證明非方法失效，是統計力問題。

→ 已啟動**全量 dump** 補統計力。詳見 `vlm_as_decision_negative_result.md §4e`。

## 進行中：全量 dump（背景跑）

- **launcher**：`run_dd2_full.sh`（三 split 依序 val→testA→testB 串跑），driver log `ddump_full_driver.log`，各 split log `ddump_full_{sp}.log`。
- **腳本**：`decomp_dump2.py`（全量版，不抽樣，跑全部 canonical rows）。
- **threshold=0（刻意不調 0.05）**：保持與既有 baseline dump（`gdino_gref_{sp}.jsonl`，撐起所有既有 CRS 主結果 3.24/2.02/3.50）同口徑，公平比較優先於速度。代價＝慢（val 14229 rows，三 split 串跑 ~數十小時）。
- **輸出**：覆寫 `gdino_gref_{sp}_decomp.jsonl`。子集 pilot 已備份成 `gdino_gref_{sp}_decomp_sub.jsonl`（勿刪，是今天 §4e 證據來源）。

### 確認全量 dump 狀態
```bash
ssh -i ~/nvsync.key p76141495@192.168.65.11 \
  'cd ~/selective-grounding && cat ddump_full_driver.log; \
   grep -a "DONE\|canonical" ddump_full_*.log; \
   wc -l dump/gdino_gref_*_decomp.jsonl; \
   ps aux | grep decomp_dump2 | grep -v grep | wc -l'
```
- `=== ALL FULL DONE ===` + 三 split 都有 `DONE {sp}: decomposed=...` + process=0 → 完成。
- 注意：`pgrep`/`ps aux` 數字若含自己的 ssh 字串會假高，認 `grep decomp_dump2 | grep -v grep | wc -l`。

## 完成後：重跑分析（dump 在手時）
```bash
ssh -i ~/nvsync.key p76141495@192.168.65.11 \
  'cd ~/selective-grounding && ./.venv/bin/python src/decomp_maintable.py'           # LTT 主表(三 split)
ssh -i ~/nvsync.key p76141495@192.168.65.11 \
  'cd ~/selective-grounding && for sp in val testA testB; do ./.venv/bin/python src/decomp_matched.py $sp; done'  # matched(決定性)
```
- **GO 判讀**：全量後 LTT 三 split 都有健康 n_feas（≫2）、R1 降 / defer 降 / R2 守 β=0.3、matched 仍 +15~24pp → 寫進論文第二主結果章。
- **若仍 EMPTY**：非統計力問題，回去查該 split 的 routing / no-target 分布。

## 已驗證結果（2026-06-18 子集 pilot，存 §4e）
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
