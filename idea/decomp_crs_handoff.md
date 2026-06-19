# 接手點 — Expression-Decomposed CRS（更新 2026-06-18 深夜，全量三 split 完成）

> ★Decomposed CRS = 跨 split 全量無瑕疵 GO，論文第二主結果數字已定。★ 全量三 split LTT n_feas 全健康(112/58/56)，decomp 純帕累托改善零例外。下一步＝subgroup 分析（GPT (B)§6，現在樣本量夠了）。

## 一句話狀態（2026-06-18 深夜）

全量 dump 三 split 全完成（val 875 / testA 1243 / testB 974 真拆）。完整 LTT + matched：
- **LTT 主表**：val R1 0.191→0.165；testA R1 0.260→0.244 且 size 2.02→1.96（雙贏）；testB size 3.50→3.34。**三 split n_feas 全健康(112/58/56=frozen)**，子集時 testA=2/testB EMPTY 完全消失。
- **matched**：三 split size~3 +19~23pp，大樣本穩固。
- 子集所有瑕疵（R2 升、可行崩潰）證實為小樣本假象。

詳見 `vlm_as_decision_negative_result.md §4e`「全量三 split 定論」段。**這是論文第二主結果的最終數字。**

## 下一步：subgroup 分析（GPT (B)§6，現在可做）

全量後樣本量夠了（真拆 875/1243/974），可安全做分組風險分析回答「decomposition 贏在哪種 query」：
- 分組：no-target / single / multi-target / long / conjunction / high-n_gt
- 每組報 R1/R2/R3 + set size + 真拆佔比
- 目標證明：decomposition 主要幫 compositional multi-target；vanilla CRS 已足夠 simple query
- 裁決依據見 `decomp_vs_gpt_framing_verdict.md §5.1`（唯一值得加的新工作）

順手可補（GPT (B)§7）：Pareto 加 detector-forwards 軸（decomp 多跑 N 次 detector 的真實 compute 成本）。

## 重跑全量分析指令
```bash
ssh -i ~/nvsync.key p76141495@192.168.65.11 \
  'cd ~/selective-grounding && ./.venv/bin/python src/decomp_maintable.py'           # 三 split LTT 主表
ssh -i ~/nvsync.key p76141495@192.168.65.11 \
  'cd ~/selective-grounding && for sp in val testA testB; do ./.venv/bin/python src/decomp_matched.py $sp; done'  # matched
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
