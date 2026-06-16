# CRS 畢業論文延伸 — Cardinality / Candidate Importance 可行性檢查 handoff

> 日期：2026-06-16　分支：feature/bts-poc-v4　資料：spark `~/selective-grounding/dump/gdino_gref_{val,testA,testB}.jsonl`
> 目的：在投入訓練任何 head 之前，用**現有 dump（零 GPU、純分析）**判斷「Cardinality-aware CRS / Candidate Importance」這條畢業論文延伸值不值得做、該怎麼做。
> 本文件供徵詢外部意見（GPT red-team）用，故完整交代脈絡。

---

## 0. 脈絡：這份檢查在整個研究的哪裡

**已完成（投稿版，不動）**：Frozen CRS（Cross-Base Conformal Referring Sets）。一條完整主線：
frozen grounding base 不可靠 → 用 LTT 把點預測改成 risk-controlled referring set，
聯合校準三風險 R1（answered-target FNR）/ R2（no-target false selection）/ R3（target deferral），
給 distribution-free 有限樣本保證。主結果 COMPOSE（OWL-ViT gate + GroundingDINO box）
三 split set size = 3.24 / 2.02 / 3.50 框，三風險全守。13 章中文初稿已完稿。

**本檢查要評估的（畢業論文延伸層）**：在 CRS 之上加一個 learned decision layer，讓輸出集合
更小更準。候選方向（來自 future-research 文件）優先序原為：
1. Cardinality-aware CRS（預測「選幾個」k̂）
2. Candidate Importance head（預測「選哪些」p_match）
3. Diversity-preserving selection
4. Box pruning / merging
5. VLM global verification

**本檢查的方法論**：沿用先前 BTS/M1 的紀律——「先用 dump 算上界 / informativeness 做 go/no-go，
再決定要不要動工」，不在未驗證前訓練任何模型。

**關鍵資料事實**：gRefCOCO target-present 樣本 n_gt 分布——
- val：幾乎全是 n_gt=2（5324 中僅 27 個 ≥3）→ **cardinality 在 val 無變異，不可學也不可驗**。
- testA：n_gt 1:5917 / 2:5940 / 3:1813 / … 一路到 16（有變異）。
- testB：n_gt 1:5646 / 2:3957 / 3:932 / … 到 14（有變異）。
→ 推論：cardinality 研究主場是 testA/testB，不是 val。

dump schema（每筆 canonical row）：`no_target, n_gt, top1_score, top2_score, margin12,
score_entropy, score_mean_topk, score_std_topk, n_cands, pred_boxes_xyxy, pred_scores, gt_boxes_xyxy`。
無預存 per-candidate IoU，但有預測框 + GT 框，IoU 現場可算。候選池為 base 經
PRED_KEEP=50 + PRED_MIN_SCORE=0.01 預處理後的 stored pool。

---

## 1. 五個檢查與結果（全部 calib/eval = ref_id parity 切分，與 CRS 一致）

### CHECK 1 — Oracle-k 上界：cardinality 是不是瓶頸？

對每個 target-present query，把 k 餵成真值 n_gt，比較三種選框方式的 R1（answered-target FNR）：

| split | 1a oracle-k + raw-score top-k | 1b oracle-k + oracle 排序 | floor（選全部候選） | gap 1a−1b |
|---|---|---|---|---|
| val | R1=0.261 (size 2.01) | R1=0.006 | 0.006 | **0.255** |
| testA | R1=0.239 | R1=0.011 | 0.011 | **0.227** |
| testB | R1=0.308 | R1=0.019 | 0.019 | **0.290** |

**解讀**：就算知道要選幾個（oracle-k），用 raw score 選 top-k，R1 仍 0.24–0.31；
換成 oracle 排序（知道哪些框對），同 size=k 下 R1 掉到 0.006–0.019（= 候選池不可化約召回 floor，
與 CRS 章報的 recall floor 吻合）。
→ **「選幾個」幾乎沒價值，「選哪些（ranking）」才是全部價值。gap 0.23–0.29 全在 ranking。**

### CHECK 2 — frozen 特徵能不能預測 k？

用 query-level 特徵對 n_gt 算 count MAE / within±1 / corr：

| split | constant(median=2) MAE | within±1 | 最強特徵 |corr(feat, n_gt)| |
|---|---|---|---|
| val | 0.010 | 99.9% | 全部 <0.07（無變異，不可學） |
| testA | 0.808 | 92.7% | score_mean_topk 0.529 |
| testB | 0.864 | 92.5% | score_mean_topk 0.430 |

**解讀**：val 無變異；testA/B 有變異但 frozen 特徵對 k 僅中等相關（0.43–0.53），
且 constant baseline 的 within±1 已 92% → cardinality head 能贏 constant 的空間僅 ~7–8%。
→ **cardinality head 難學、天花板低。**

**CHECK 1+2 合論：cardinality 不是瓶頸，且不好學。文件的第一優先（cardinality）應降級。**

### CHECK 3 — candidate importance 可學性（global / pooled AUROC）⚠️ 陷阱

每個候選框算 p_match label（IoU≥0.5 任一 GT 為正），用 per-box frozen 特徵
（score, score_norm, rank, rank_frac, area_frac, overlap_count, mean_iou_others, score_gap_top1）
訓 logistic regression（calib-fit → eval-test），把**所有 query 的候選 pool 在一起**算 AUROC：

| split | candidate rows | 正例率 | 多特徵 logreg AUROC | raw score 單獨 |
|---|---|---|---|---|
| val | 244K | 29.8% | 0.757 | 0.626 |
| testA | 608K | 26.9% | 0.731 | 0.635 |
| testB | 480K | 24.0% | 0.735 | 0.610 |

表面看 = GO（0.73–0.76 >> raw 0.61）。**但這是 pooled AUROC，是陷阱（見 CHECK 4/5）。**

### CHECK 4 — learned p_match 接回 top-k：實際 R1 落點

把 CHECK 3 的 learned p_match 拿來選 top-k（size=k），量真實 R1：

| split | raw-score top-k（現況） | learned p_match top-k | oracle 排序（上界） |
|---|---|---|---|
| val | R1=0.272 | R1=0.393 ❌ | 0.007 |
| testA | R1=0.242 | R1=0.303 ❌ | 0.012 |
| testB | R1=0.303 | R1=0.364 ❌ | 0.017 |

**解讀**：learned ranker 接回 top-k 後 R1 **比 raw score 更差**（倒退 21–46% of gap）。
與 CHECK 3 的高 AUROC 直接矛盾。

### CHECK 5 — per-query AUROC（逐 query 內部算再平均）：解開矛盾

| split | raw score per-query AUROC | learned p_match per-query AUROC | 最強單特徵 |
|---|---|---|---|
| val | 0.692 | 0.734 | overlap_count 0.672 |
| testA | 0.684 | 0.734 | overlap_count 0.639 |
| testB | 0.682 | 0.729 | overlap_count 0.611 |

**解讀**：per-query 下 learned（0.73）確實 > raw（0.69）——candidate importance **per-query 排序是真的有 gain**。
那 CHECK 4 為什麼更差？

---

## 2. 核心發現（最重要，給 GPT 重點看這段）

### 發現 A：cardinality（選幾個）不是瓶頸，candidate ranking（選哪些）才是
oracle-k 都還有 R1=0.24–0.31，瓶頸全在 ranking（CHECK 1）。文件原優先序 1（cardinality）→ 2（candidate）
**應對調**。cardinality 降為可選旋鈕。

### 發現 B：pooled AUROC 是陷阱，會給假 GO
CHECK 3（pooled AUROC 0.73）看起來 GO，但 CHECK 4（接回 top-k）R1 反而更差。
原因：pooled AUROC 可以靠「分出哪些 query 整體容易 / 難」拿高分，對 per-query 選框無用。
**教訓：candidate selection 的評估指標必須是 per-query 的覆蓋（R1 / coverage），不能用 pooled AUROC。**

### 發現 C：最具資訊量的特徵（擁擠度）同時最反多樣性 —— 這是真正的障礙
learned ranker per-query AUROC 真的較高（CHECK 5），但接回 top-k 卻傷 R1（CHECK 4）。真因：
- AUROC 獎勵「把所有 TP 排在 neg 前」；top-k R1 要的是「覆蓋到所有**不同**的 GT」。兩件事。
- learned 最依賴 `overlap_count`/`mean_iou_others`（框擁擠度），它學到「擠成一團的框更可能是 TP」（邊際對），
  但擁擠度**集中在同一個 GT 上**：n_gt=2 query 若 GT_easy 周圍 10 個重疊框、GT_hard 只 1 孤框，
  learned 把 10 個擠框排前面（AUROC 高），top-2 全落 GT_easy → 漏 GT_hard → FNR=0.5。
- `overlap_count` 對孤獨的 GT_hard 框反向懲罰。
→ **最強訊號（擁擠度）反多樣性。** 這解釋為何 per-query 排序變好但覆蓋變差。

---

## 3. 定案方向（本檢查的結論）

1. **cardinality head 不當主升級**（CHECK 1+2）——降為可選旋鈕。
2. **candidate importance 可學且 per-query 真有 gain**（CHECK 5）——不是死路。
3. **但單靠 top-k by p_match 會自毀**（CHECK 4）——因為最強訊號反多樣性（發現 C）。
4. 因此文件**第三層 diversity-preserving selection 不是 optional 打磨，而是讓 candidate importance 有用的前提**。
5. **評估指標必須鎖 per-query 覆蓋（R1 / coverage），AUROC 會騙人**（發現 B 是活例）。

→ 畢業論文真正命題收斂為：
> **candidate importance + diversity/coverage-aware selection，以 R1（覆蓋）為目標聯合校準，
> 並仍接回 CRS / LTT 給三風險保證。** 而非「學一個 p_match 然後 top-k」。

cardinality 可作為 selection 的一個 budget 旋鈕（k̂ + margin，margin 由 LTT 校準），但不是核心貢獻。

---

## 4. 想請教 GPT 的問題（red-team 點）

1. **發現 C（擁擠度反多樣性）是否致命？** diversity-aware selection（如 farthest-point over box centers、
   NMS-cluster 代表、score-quantile diversity）能不能在不知道 GT 的情況下，於推論時把覆蓋救回來？
   還是說「哪個是不同 GT」本身就需要 GT 才知道，使這條路注定要更強的語義特徵？
2. **frozen by-products 的 per-query 排序天花板就在 0.73 嗎？** dump 裡只有 score + 框幾何。
   要再上去似乎只能靠 dump 沒有的特徵（VLM crop 語義 p_match、cross-base agreement、relation/attribute residual）
   —— 那就需要重跑 GPU dump，不再是便宜檢查。值得為此重跑嗎？還是 0.73 + diversity 已足夠撐畢業論文？
3. **VLM global verification（文件第五層）會不會其實該前移？** 它正好補 CRS 已知 limitation
   （長 expression 下 OWL gate 退化、R2 惡化到 0.40–0.61）。對小 compact set 做 set-level 語義驗證，
   是否比 candidate importance head 更高 ROI、且更避得開 True/False Verification(2509.09958) 的近鄰威脅？
4. **畢業論文的最小可防守貢獻組合**該怎麼劃？目前傾向：
   (a) candidate importance + diversity-aware coverage selection（主），
   (b) 接回 CRS/LTT 保證（已有），
   (c) VLM verification 補長句 limitation（應用延伸）。
   這樣夠不夠一本畢業論文？cardinality 完全砍掉會不會可惜？

---

## 5. 復現資訊
- 腳本（在 spark `~/selective-grounding/dump/`）：`cardinality_check.py`(CHECK 1+2)、
  `candidate_check.py`(CHECK 3)、`check4.py`(CHECK 4)、`check5.py`(CHECK 5)。
- 全程零 GPU、純讀 dump jsonl。calib/eval = ref_id parity。logreg 為手寫 GD（標準化 + 300 iter）。
- 數字可一鍵復現：`ssh spark 'cd ~/selective-grounding/dump && python3 check5.py'`。
