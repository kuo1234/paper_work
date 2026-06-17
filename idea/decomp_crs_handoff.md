# 接手點 — Expression-Decomposed CRS（2026-06-16 晚）

> 明天回來先讀這份。背景 dump 整夜跑，醒來直接接 LTT。

## 一句話狀態

Expression-Decomposed CRS 三道門全過（上界 R1 可達 0.006 / VLM 拆解品質 within±1 0.98 / e2e pilot R1 0.34→0.075）。
**現在卡在「生成完整 decomp 候選 dump」這一步**，背景跑中，完成後跑一行指令就出主表。

## 背景 dump 狀態（整夜跑）

- **process**：spark `~/selective-grounding/`，`decomp_dump3.py val 1500`，nohup 啟動（斷線不死）。
- **產出檔**：`dump/gdino_gref_val_decomp.jsonl`（schema 同 gdino_gref_val.jsonl，多 `decomposed`/`n_parts` 欄位）。
- **範圍**：subset=10405 行（no_target 8905 全 passthrough + target-present 抽樣 1500 個拆解）。為何子集：全量 14229 在 GDINO threshold=0（900 候選 post_process 慢）下要 10+ 小時，故抽 1500 tp（LTT 只需 ≥200，足夠）。
- **進度快照（睡前）**：848/10405 行，RUNNING。

### 明天第一件事：確認 dump 完成
```bash
ssh -i ~/nvsync.key p76141495@192.168.65.11 \
  'cd ~/selective-grounding && grep DONE ddump_val.log; wc -l dump/gdino_gref_val_decomp.jsonl; pgrep -f decomp_dump3 && echo RUNNING || echo DONE'
```
- 若 `DONE val: decomposed=N passthrough=M` 出現 → 完成，接下一步。
- 若還 RUNNING → 再等；若數字不再增加且無 DONE → 可能卡某筆，看 `tail ddump_val.log` 排查。

## 完成後：跑 LTT 主表（已寫好，一行指令）

```bash
ssh -i ~/nvsync.key p76141495@192.168.65.11 \
  'cd ~/selective-grounding && ./.venv/bin/python src/decomp_maintable.py'
```

`src/decomp_maintable.py` 已寫好放在 spark。它會輸出三組對照（同一份 records 的共同 key 上公平比較）：
- **COMPOSE frozen**：full-expression GD 候選池（= 現有 Frozen CRS）
- **COMPOSE decomp**：decomp-union GD 候選池（= 新方法）
- 各報 sz / R1 / R2 / defer + bootstrap CI + n_feas

### 判讀（go/no-go 條件）
- **GO**：decomp 的 R1 或 set size 明顯優於 frozen，且 R2/R3 沒爆（守 β=0.3 / γ=0.5）。
- **R2 安全有保障**：no-target 用 v1 保守 routing，誤拆率實測 0.000（150 個全不拆），所以 decomposition 不碰 no-target，R2 理論上應與 frozen 持平。這是這步要驗證的核心。
- **注意可比性**：decomp dump 只含 1500 tp 子集，maintable 用 `set(owl)&set(decomp)` 共同 key，frozen 也只在這些 key 上算 → 公平。但 val 整體 tp 是 5324，子集 1500 約 28%，CI 會比全量寬。

## 若 val GO，後續（明天之後）
1. testA / testB 也跑 decomp dump（`decomp_dump3.py testA 1500` / `testB 1500`）+ maintable。
2. 全量 val（若要進論文主表，子集只是 pilot）——需解 GDINO 慢的問題（threshold 調 0.05 或 batch）。
3. 寫成畢業論文第二主結果章。

## 關鍵數字回顧（已驗證，存在 vlm_as_decision_negative_result.md §4b/4c/4d）
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
