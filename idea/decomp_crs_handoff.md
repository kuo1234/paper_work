# 接手點 — Expression-Decomposed CRS（更新 2026-06-18）

> val 已 ★乾淨 GO★。testA/testB decomp dump 整夜跑中，醒來確認完成後跑 maintable 做跨 split 確認。

## 一句話狀態（2026-06-18）

val LTT 主表 + matched oracle 分析全數確認 Decomposed CRS = **乾淨 GO**：同 size 下 recall +17~20pp（決定性，繞過 LTT），R1 0.191→0.162、defer 42%→30%。詳見 `vlm_as_decision_negative_result.md §4e`。
**現在卡在 testA/testB decomp dump**（整夜依序跑 testA→testB），完成後跑 maintable 即得跨 split 確認。

## 明天第一件事：確認 testA/testB dump 完成
```bash
ssh -i ~/nvsync.key p76141495@192.168.65.11 \
  'cd ~/selective-grounding && grep "ALL DONE" ddump_test_driver.log; \
   wc -l dump/gdino_gref_testA_decomp.jsonl dump/gdino_gref_testB_decomp.jsonl 2>/dev/null; \
   ps aux | grep decomp_dump3 | grep -v grep | wc -l'
```
- `ALL DONE` 出現 + 兩檔都有行數 + process 數=0 → 完成，跑下方 maintable。
- 只有 testA 檔 → testB 還在跑，看 `tail ddump_testB.log`。
- 啟動器：`run_dd3_test.sh`（已寫好，依序 testA→testB 各 1500，driver log = `ddump_test_driver.log`）。

## 完成後：跑 LTT 主表（跨 split 確認）

```bash
ssh -i ~/nvsync.key p76141495@192.168.65.11 \
  'cd ~/selective-grounding && ./.venv/bin/python src/decomp_maintable.py'
```
maintable 會自動偵測 testA/testB decomp dump（`build_decomp`），三 split 都印 Frozen vs Decomp。同時可重跑 `src/decomp_matched.py`（目前寫死 val，要看 testA/testB 需把檔名參數化）。

### 判讀（跨 split go/no-go）
- **GO**：testA/testB 與 val 同向（同 size recall 勝、defer 降、R2 守 β=0.3）→ 寫進論文第二主結果章。
- **NO**：若某 split 反向或 R2 爆 → 回去看該 split 的 matched 分析定位原因。

## val 已驗證結果（2026-06-18，存 §4e）
- 主表：sz 持平 3.24→3.23、R1 0.191→0.162、R2 0.159→0.209（守 β）、defer 0.424→0.299
- matched oracle（255 真拆 case，繞過 LTT）：同 size recall **+17~20pp**（size~2: 0.64→0.81；~3: 0.70→0.90；~4: 0.74→0.94）
- 機制：per-part query 分數乾淨 → 同 size 留對框；三疑慮（CI 重疊/R2 反常/全域位移）全解消

## 若三 split 全 GO，後續
1. 全量 val（子集只是 pilot）——需解 GDINO threshold=0 慢（10+ 小時）：調 0.05 或 batch。
2. 寫成畢業論文第二主結果章。

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
