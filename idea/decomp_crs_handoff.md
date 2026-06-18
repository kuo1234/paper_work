# 接手點 — Expression-Decomposed CRS（更新 2026-06-18 晚）

> 方法層跨三 split 全 GO（matched +15~24pp）；LTT 校準層在小子集上 testA n_feas=2 / testB EMPTY。**下一步＝全量 dump**（小子集統計力不足，非方法問題）。

## 一句話狀態（2026-06-18 晚）

跨 split 確認跑完，分裂成兩個故事：
- **方法層（matched oracle，繞過 LTT）= 三 split 全 GO**：同 size recall val +20pp / testA +15pp / testB +24pp，零例外。核心主張坐實。
- **校準層（LTT 主表）小子集力不從心**：testA n_feas=2、testB EMPTY feasible。matched 證明非方法失效，是統計力問題（子集 calib 砍半後 no-target 撐不起三風險聯合可行區間）。

詳見 `vlm_as_decision_negative_result.md §4e`（含「跨 split 確認」段）。

## 下一步：全量 dump（解 LTT 力不從心）

LTT 需要更大 calib set。子集（1500 tp）本是 pilot，matched 已給方法可行的決定性證據，現在投資源跑全量：

1. **解 GDINO threshold=0 慢的瓶頸**（全量 ~14229 在 threshold=0 要 10+ 小時）：
   - 選項 a：threshold 調 0.05（少候選、快很多），重驗 matched 不退化即可。
   - 選項 b：batch 推論。
   - 腳本起點：`decomp_dump2.py`（全量版，目前用 threshold=0）。
2. 全量 dump 後重跑 `src/decomp_maintable.py` → 三 split LTT 應都有健康 n_feas。
3. 若全量 LTT 三 split 同向 → 寫進論文第二主結果章。

## 確認指令（dump 狀態）
```bash
ssh -i ~/nvsync.key p76141495@192.168.65.11 \
  'cd ~/selective-grounding && grep "ALL DONE\|DONE" ddump_*.log 2>/dev/null; \
   wc -l dump/gdino_gref_*_decomp.jsonl 2>/dev/null; \
   ps aux | grep decomp_dump | grep -v grep | wc -l'
```

## 重跑分析（dump 在手時）
```bash
ssh -i ~/nvsync.key p76141495@192.168.65.11 \
  'cd ~/selective-grounding && ./.venv/bin/python src/decomp_maintable.py'           # LTT 主表(三 split)
ssh -i ~/nvsync.key p76141495@192.168.65.11 \
  'cd ~/selective-grounding && for sp in val testA testB; do ./.venv/bin/python src/decomp_matched.py $sp; done'  # matched(決定性)
```

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
