---
type: spec-doc
aliases:
  - "Selective Grounding 實作規格"
  - "M0/M1 dump schema + baseline + protocol"
tags:
  - 研究主線
  - 規格
  - selective-prediction
  - referring-grounding
  - CLIP
  - belief
summary: "Selective / Uncertainty-Aware Referring Grounding 的五份可實作規格：related-work 差異表與 novelty claim、M0/M1 dump schema、C-min no-target baseline stack、B operator spec、split/calibration/threshold protocol。"
---

# Selective Grounding — 實作規格（5 份）

> 配套決策文件：[[thesis_selective_grounding_plan]]。本檔是把計畫落成可寫 code / 可寫論文的規格。
> 凡需查證但本 session 無法連網的條目，以 `〔待補〕` 標記，**不得用未經查證的引用充數**。

---

## 1. Related Work 差異表 + Final Novelty Claim

### 1.1 重要更正（VIRO 狀態）

舊計畫把 VIRO 寫成 *ICLR'26 withdrawn / unpublished*。**此描述已過期**：VIRO 已被 **CVPR 2026 接收**，必須當作**已發表競品**對待，不能再用「withdrawn」當作弱化它的理由。

- 正式引用（已查證 arXiv:2601.12781）：
  > Hyejin Park, Junhyuk Kwon, Suha Kwak, Jungseul Ok.
  > **VIRO: Robust and Efficient Neuro-Symbolic Reasoning with Verification for Referring Expression Comprehension.**
  > arXiv:2601.12781 (v1 2026-01-19, v2 2026-03-20). Accepted to CVPR 2026.
- 影響：VIRO 與本研究在 *framing* 上高度重疊（abstention + no-target + verification），差異必須建立在**方法路線**而非「它沒發表」。
- VIRO 的具體賣點（查證自 abstract）：在 neuro-symbolic REC pipeline（LLM/VLM 把 query 拆成可執行 program）中，**每個 operator 內嵌輕量 verifier**（VIRO=Verification-Integrated Reasoning Operators），逐步驗證 object existence / spatial relation，解決「cascading error → high-confidence false positive」；以 **verification-aware abstention** 處理 no-target；report **61.1% balanced accuracy**（target-present + no-target 合併）、program failure rate ≤0.3%、可泛化到 egocentric 真實資料。abstract 未點名 RefCOCO/gRefCOCO 具體 split 數字。

### 1.2 撰寫紀律

- 全文**禁用** "first" / "unique" / "novel framework"。改用可被審稿人核對的相對陳述：
  "we study", "we present a frozen-base, post-hoc …", "in contrast to X which Y, we Z"。
- 貢獻錨點維持三件、且都是**可被實驗證偽**的相對主張，不是 priority claim：
  1. **Frozen-base**：不更新任何 base 參數。
  2. **Post-hoc**：belief 由 base 免費/低成本副產品 + 輕量 calibrator 導出。
  3. **Risk-calibrated belief policy**：以 risk-coverage / calibration 量化的 ANSWER / ABSTAIN / INTERVENE 三態決策，並報 oracle gap。

### 1.3 差異表（取代舊表）

| 工作 | 任務 | 動 base？ | abstain / no-target | 機制路線 | 與本研究的關係 |
|---|---|---|---|---|---|
| CLIP-VG | REC | 是（curriculum fine-tune） | 否 | 把 base 練更準 | 可當本研究的 frozen base #1 |
| OWL-ViT | open-vocab detection | —（預訓練） | 否 | open-vocab detector | 可當 frozen base #2，驗 base-agnostic |
| RefFormer (NeurIPS'24) | REC | 是 | 否 | query-based grounding | 可當 frozen base #3（復現風險，列後） |
| LIHE / gRefCOCO 系 | GREC | 是 | 部分（顯式建 0/multi referent） | 專門架構 + 標籤監督 | 它把 no-target 當分類目標專訓；本研究以 post-hoc belief 在**凍結** base 上導出，可掛任意 base |
| HieA2G (AAAI'25) | GREC | 是 | 是（Adaptive Grounding Counter） | trained counter + 階層 align | trained counting head；本研究 no-target gate 不重訓、近零成本，定位為 post-hoc 對照 |
| SIMMC2.0 clarification | 對話式 grounding | — | 類似（多輪澄清） | 對話 loop + 使用者回覆 | 需互動；本研究單輪自動 abstain/消歧 |
| **VIRO** (Park et al., CVPR'26; arXiv:2601.12781) | REC + no-target | 否（凍結 LLM/VLM 組件） | 是（**verification-aware abstention**） | **LLM/VLM query→program 拆解 + operator 內嵌 verifier 逐步符號驗證** | framing 最近；**方法路線正交**：VIRO 走程式拆解 + operator-level 驗證器（驗 object existence / spatial relation）抑制 cascading error，本研究走 frozen base + 單一輕量 post-hoc calibrator，無程式合成、無 per-operator 符號驗證、base-agnostic、近零訓練。兩者可互補對照 |
| **本研究** | REC + GREC | **否** | **是（核心）** | frozen base + post-hoc risk-calibrated belief policy | — |

> 表內每一列的「機制路線」是審稿人區辨本研究的關鍵欄，務必填實，不要只比「動不動 base」。

### 1.4 Final Novelty Claim（可貼進論文 intro / abstract）

> We study selective referring grounding under a **frozen base**: rather than retraining a grounding model to be more accurate, we attach a **post-hoc, risk-calibrated belief policy** that decides, per query, whether to **answer**, **abstain**, or run a grounding-specific **candidate-contrastive intervention**. The belief is derived from cheap by-products of any base model (candidate geometry, prompt-induced identity stability, relation/attribute residuals, cross-model agreement) plus a lightweight calibrator with no base updates. We quantify the policy on **risk–coverage** (REC) and on **no-target / generalized REC** (gRefCOCO), and we report the **oracle gap** to bound how much signal remains. In contrast to architecture-level GREC methods (LIHE, HieA2G) that train dedicated no-target/counting heads, and to neuro-symbolic pipelines that verify LLM-synthesized programs operator-by-operator (VIRO), our policy is a single base-agnostic, near-training-free calibrator.

差異一句話（防「只是加 threshold」與「跟 VIRO 重複」）：

> Unlike a single confidence threshold, the policy is a learned multi-signal calibrator with a conditional, grounding-specific intervention; unlike VIRO's per-operator neuro-symbolic verification over LLM-synthesized programs, our belief comes from a **single lightweight post-hoc calibrator** on cheap base by-products, with no program decomposition or operator-level symbolic executors, and transfers across frozen bases.

---

## 2. M0 / M1 Offline Dump Schema

**設計目標**：一次 dump 同時支撐 (a) RefCOCO correctness AUROC、(b) gRefCOCO no-target AUROC、(c) risk-coverage、(d) B intervention ablation、(e) cross-base transfer。

### 2.1 Granularity（每筆 row 是什麼）

一 row = **(sample, base_model, prompt_variant)** 三元組。

- `prompt_variant = "canonical"` 為主 row，承載 correctness / no-target / risk-coverage 主分析。
- 同義改寫 / discriminative / neutral prompt 各自為附 row（同 `sample_uid`、不同 `prompt_variant`），供 cross-prompt consistency 與 B operator 還原 ΔS。
- cross-base transfer：同一 `sample_uid` 跨多個 `base_model` 各一組 row；分析時以 `sample_uid` join。

存成 **JSONL（一 row 一行）** + 分區目錄 `dump/{dataset}/{base_model}/{split}.jsonl`。大張量（每候選的 per-prompt score 向量）可外掛 `.npz`，row 內存指標。

### 2.2 欄位表

#### (A) 識別與環境
| 欄位 | 型別 | 說明 |
|---|---|---|
| `sample_uid` | str | 全域唯一：`{dataset}:{split}:{image_id}:{ref_id}`，cross-base join key |
| `dataset` | str | `refcoco` / `refcoco+` / `refcocog` / `grefcoco` |
| `split` | str | `train` / `val` / `testA` / `testB`（gRefCOCO 用其官方 split 名） |
| `image_id` | int | 原始 COCO image id |
| `ref_id` | int | referring 標註 id（gRefCOCO 可為 no-target 標註 id） |
| `sent_id` | int | 句子 id（一 ref 多句時） |
| `expression` | str | referring expression 原文 |
| `image_path` | str | 相對路徑 |
| `image_w` / `image_h` | int | 影像尺寸（IoU / dispersion 正規化用） |
| `base_model` | str | `clip-vg` / `owl-vit` / `refformer` |
| `base_version` | str | checkpoint / commit，確保可復現 |
| `prompt_variant` | str | `canonical` / `paraphrase_k` / `disc_<phenotype>` / `neutral` |
| `dump_run_id` | str | dump 批次 id（追溯哪次 forward） |

#### (B) Base 輸出（候選）
| 欄位 | 型別 | 說明 |
|---|---|---|
| `candidates` | list[obj] | top-K 候選，每個 `{cand_id, bbox:[x,y,w,h], raw_score}`；K 固定（建議 K=10）並記於 meta |
| `K` | int | 實際候選數（base 可能 <K） |
| `pred_cand_id` | int | argmax 候選 id（standard 系統的輸出） |
| `pred_bbox` | list[float] | argmax bbox |
| `score_vec_ref` | str/null | 指向 `.npz` 的 key：本 row 各候選分數向量（cross-prompt 用） |

#### (C) Ground truth 與 correctness
| 欄位 | 型別 | 說明 |
|---|---|---|
| `is_no_target` | bool | gRefCOCO no-target 標註；RefCOCO 系恆 `false` |
| `num_referents` | int | GT referent 數（RefCOCO=1；gRefCOCO 可 0/1/多） |
| `gt_bboxes` | list[list] | GT 框（multi-target 為多個） |
| `gt_cand_ids` | list[int] | 候選中被 IoU≥0.5 對上的 GT 候選 id（可空=GT 不在候選內，重要負例） |
| `pred_iou` | float | `pred_bbox` 對最佳 GT 的 IoU |
| `correct@0.5` | bool | REC 正確性主標籤（IoU≥0.5 且非 no-target） |
| `best_cand_iou` | float | 全候選對 GT 的最佳 IoU（量「框集合是否含對的」，oracle 上界用） |

#### (D) Belief / uncertainty 訊號（M1 informativeness 篩選的核心）
| 欄位 | 型別 | 成本 | 說明 |
|---|---|---|---|
| `top1_score` | float | 免費 | argmax 絕對分 |
| `margin12` | float | 免費 | top1 − top2 |
| `score_entropy` | float | 免費 | softmax(scores) 的熵 |
| `spatial_dispersion` | float | 免費 | top-m 框中心的幾何離散度（除以 √(w·h) 正規化） |
| `cross_prompt_consistency` | float | 中 | 各 paraphrase row 的 top-1 一致比例 / 平均 IoU（由 paraphrase rows 聚合回 canonical） |
| `identity_stability_entropy` | float | 中 | 跨 prompt「哪個候選被選中」分布的熵 |
| `cross_model_agreement` | float/null | 中高 | 與另一 base top-1 的 IoU / 是否一致（cross-base join 後填） |
| `relation_residual` | float/null | 中 | ΔS(relation discriminative − neutral)；**RefCOCO+ 必為 null** |
| `attribute_residual` | float/null | 中 | ΔS(attribute discriminative − neutral)；RefCOCO+ 主用此項 |

> M1 go/no-go：對 (C) 的 `correct@0.5`（REC）與 `is_no_target`（gRefCOCO）分別算每訊號 AUROC。

#### (E) B intervention ablation 專用
| 欄位 | 型別 | 說明 |
|---|---|---|
| `phenotype` | str/null | `candidate_competition` / `spatial_ambiguity` / `semantic_absence` / `model_disagreement` / `none`（規則見 §4） |
| `disc_prompt` | str/null | 本 row（disc variant）所用 discriminative prompt 原文 |
| `neutral_prompt` | str/null | 對應 neutral prompt 原文 |
| `delta_scores` | list[float]/null | 各候選 ΔS = S(disc) − S(neutral) |
| `reranked_cand_id` | int/null | INTERVENE 後重排 top-1 |
| `reranked_iou` | float/null | 重排後 top-1 對 GT 的 IoU |
| `intervened` | bool | 本 sample 是否實際觸發 INTERVENE |

> ablation：比較 `pred_iou`(argmax) vs `reranked_iou`，在 `intervened=true` 子集上算 intervention lift；並記觸發率與平均額外 forward 次數。

#### (F) Protocol bookkeeping
| 欄位 | 型別 | 說明 |
|---|---|---|
| `calib_fold` | str | `calib_train` / `calib_val` / `test`（§5 切分，固定種子預先指派） |
| `seed` | int | 影響 fold / paraphrase 抽樣的種子 |

### 2.3 Meta（每個 dump 目錄一份 `meta.json`）
記錄：K、paraphrase 數、prompt 模板版本、base checkpoint hash、IoU 對應門檻（0.5）、softmax 溫度、dump_run_id、產生時間（由呼叫端傳入，腳本內不取系統時間以利重現）。

---

## 3. gRefCOCO C-min No-Target Gate — 正式 Baseline Stack

**任務**：判斷「圖中是否存在 referent」。輸出 abstain（no-target）或 grounding。**不能只比 forced-output base**。

### 3.1 Baseline 階梯

| # | Baseline | 規則 | 角色 |
|---|---|---|---|
| 0 | **Forced-output base** | 永遠輸出 argmax，從不 abstain | 下界（證 base 無 abstain 能力） |
| 1 | **Random abstain** | 以基準 abstain 率隨機棄答 | risk-coverage 最低參考線 |
| 2 | **Score-threshold** | `top1_score < τ_s` → no-target | 最常見稻草人 |
| 3 | **Margin-threshold** | `margin12 < τ_m` → no-target | 競爭型訊號對照 |
| 4 | **Entropy-threshold** | `score_entropy > τ_e` → no-target | 分布型訊號對照 |
| 5 | **Learned P(no-target)** | logistic / 小 MLP 吃 (D) 全訊號 → `P(no-target) > τ_p` | **本研究主方法** |
| 6 | **Oracle no-target gate** | 用 GT `is_no_target` 完美決定 | 上界（量 signal gap） |

- 2–4 是**單訊號** baseline；5 是**多訊號 learned** 方法；6 是上界。2–6 的所有 τ 一律在 calib_val 選定（§5），禁止 test 調參。
- **每條都掛在同一組 frozen base 上**（先 CLIP-VG，再 OWL-ViT 驗 base-agnostic），確保差異來自 gate 不是 base。

### 3.2 Related-work positioning（baseline ≠ 只有 threshold）

- **HieA2G (AAAI'25)**：trained Adaptive Grounding Counter，屬「重訓 counting head」路線。列為**對照組（trained upper reference）**，誠實標註它有監督訓練、本研究 #5 無；比的是「post-hoc 近零成本能逼近多少」。
- **VIRO (Park et al., CVPR'26)**：neuro-symbolic verification 的 abstention（verification-aware abstention，report 61.1% balanced acc on target-present+no-target）。列為**對照路線**，標註其 LLM/VLM program 拆解 + per-operator verifier 的推論成本。**注意：VIRO 本身也凍結 base 組件**，所以「動不動 base」**不是**對 VIRO 的區辨點 —— 區辨點是「單一 post-hoc calibrator」vs「程式拆解 + 多個 operator-level 符號驗證器」。若能取得其 no-target 數字則同表並列（其 split 數字 abstract 未列，需查全文），取不到則只在文字 positioning。
- 兩者都**不是**用來「打贏」，而是定位本研究在 trained-architecture ↔ post-hoc 光譜上的位置。

### 3.3 指標
- **No-target AUROC**（gate 分數 vs `is_no_target`）：M1 第二 go/no-go。
- **GREC 官方**：N-acc、T-acc、Pr@(F1=1, IoU≥0.5)。
- **Risk-coverage**：把 abstain 決策當 selective prediction，畫 RC 曲線、AURC。
- 全部報 bootstrap CI（§5）。

---

## 4. B Operator 實作規格 — Belief-conditioned Candidate-Contrastive Re-ranking

> 紅隊修正：INTERVENE **不可**是「低 confidence 時多跑 prompt」（會被打成普通 TTA / prompt ensemble）。必須是 grounding-specific candidate-contrastive operator。

### 4.1 觸發條件（gate 進入 INTERVENE）
僅在 calibrator `p_correct` 落入「中段」且 **margin 小** 時觸發（高 p→ANSWER；全候選低分→ABSTAIN）。觸發後先做 phenotype 診斷。

### 4.2 Ambiguity Phenotype 判斷（四型，互斥，依序判定）

| Phenotype | 判定條件（皆用 calib_val 校過的門檻） | 直覺 |
|---|---|---|
| `semantic_absence` | 全候選 `top1_score` 低 且 `score_entropy` 高 | 圖裡可能根本沒有 → 偏 ABSTAIN/no-target |
| `model_disagreement` | `cross_model_agreement` 低（兩 base top-1 IoU 小） | 兩 base 不同意 → 需 contrastive 裁決 |
| `candidate_competition` | `margin12` 小 且 top-2 候選 IoU 互不重疊 | 兩個不同物在競爭 → attribute/relation 裁決 |
| `spatial_ambiguity` | `margin12` 小 且 top-m 框空間離散 `spatial_dispersion` 高 | 位置/關係詞決勝（**RefCOCO+ 不適用**） |
| `none` | 以上皆否 | 不介入，回 ANSWER |

> 判定順序：absence → disagreement → competition → spatial。落空才往下。phenotype 寫入 dump (E) `phenotype`。

### 4.3 Discriminative prompt 生成
從 expression 解析出**區辨屬性**，對 top-m 候選各生成一條 discriminative prompt：

- **competition（屬性差異）**：抽 expression 中的 attribute token（顏色/材質/類別形容詞），生成
  `"the {attribute} {noun}"`，逐候選打分。
- **spatial_ambiguity（關係）**（**僅 RefCOCO / RefCOCOg**）：抽空間/關係詞（left/on/behind…），生成
  `"the {noun} {relation} {anchor}"`，用候選對 anchor 的幾何關係驗證。
- **RefCOCO+**：資料集**禁止位置詞** → 一律走 **attribute-contrastive**，不得生成關係 prompt（否則違反資料集設定、訊號失真）。phenotype 即使判為 spatial 也降級為 competition 處理。
- **semantic_absence**：不生成 discriminative prompt，直接導向 ABSTAIN / no-target gate。

### 4.4 Neutral prompt 定義
同結構、抽掉區辨資訊的 base prompt，用來扣掉 base 的 per-object score prior（即 BTS 的 score bias 來源）：

- competition / spatial 對應的 neutral = `"the {noun}"`（只留名詞，去掉 attribute / relation）。
- neutral 對 top-m 同樣逐候選打分，得 `S(neutral)`。

### 4.5 Residual normalization（核心，防 score bias 復發）
對每候選 c：

```
ΔS(c) = S_disc(c) − S_neutral(c)
```

- 重排依 `ΔS` 而非絕對分 → 扣除「某候選天生分高」的 prior。
- 若多條 discriminative prompt（多屬性），取各條 ΔS 後 z-score 標準化再加總。
- INTERVENE 輸出 `reranked_cand_id = argmax_c ΔS(c)`，寫入 dump (E)。

### 4.6 退場條件
若 B 在 `intervened` 子集的 lift 不顯著（bootstrap CI 跨 0），依計畫降為 analysis/ablation，A+C 撐主論文。此判定也在 calib_val 上先看趨勢，test 只報最終數字。

---

## 5. Split / Calibration / Threshold Protocol

**鐵律：任何 τ、calibrator 參數、phenotype 門檻一律在 train/val 決定，test 只跑一次出最終數字。**

### 5.1 切分
- **RefCOCO/+/g**：用官方 train / val / testA / testB。
  - `calib_train` = 官方 **train**（訓 calibrator）。
  - `calib_val` = 官方 **val**（選所有 τ 與門檻、early stop、phenotype 校準）。
  - `test` = **testA / testB**（最終報告，分開報）。RefCOCOg 用 val/test。
- **gRefCOCO**：用其官方 train/val/test；no-target gate 的 τ_p 在 gRefCOCO val 選。
- dump 時即把每 row 的 `calib_fold` 依官方 split 固定指派（固定 seed），避免事後洩漏。

### 5.2 Calibrator 訓練
- 模型：logistic regression 或 ≤2 層小 MLP（數十～數百參數，**不碰 base**）。
- 特徵：dump (D) 全訊號；缺值（如 RefCOCO+ 的 relation_residual=null）以 mask + 指示位處理，不可用 0 混淆。
- 標籤：REC 用 `correct@0.5`；no-target gate 用 `is_no_target`。
- 標準化參數（mean/std）只用 `calib_train` 統計，套用到 val/test，**禁止用 test 統計**。
- 多 seed（≥5）重訓，報平均 ± CI。

### 5.3 Threshold 選擇
- 所有 τ（score/margin/entropy/p_correct/p_no_target）在 `calib_val` 上依目標準則選：
  - selective：固定 target risk r*，選滿足 risk≤r* 的最大 coverage 之 τ（Coverage@target-risk）。
  - 或固定 target coverage，回報該點 risk。
- 兩種準則都**先在 val 定 τ，再凍結套到 test**。論文明確寫出準則與 r*/coverage 值。

### 5.4 報告指標
- **Risk–Coverage curve** + **AURC / E-AURC**（招牌圖，selective vs naive vs random vs oracle）。
- **AUROC**（correctness / no-target）。
- **ECE + reliability diagram**（calibrator 校準度）。
- **AUSE**（uncertainty 排序品質）。
- **Selective accuracy @ fixed coverage**、**Coverage @ target risk**。
- **GREC**：N-acc / T-acc / Pr@(F1=1, IoU≥0.5)。
- **Oracle gap**：每張主圖附 oracle 線，量「訊號還差多少」。

### 5.5 統計顯著性
- **Bootstrap CI**：對 test 樣本 resample（≥1000 次）求 AURC / AUROC / accuracy 的 95% CI。
- **多 seed**：calibrator / paraphrase 抽樣的隨機性以 ≥5 seed 平均，報 seed 間 std。
- 跨 base / 跨 baseline 的比較給 CI 重疊判讀；B 的 lift 以 `intervened` 子集 bootstrap CI 是否跨 0 作退場依據（§4.6）。

### 5.6 防洩漏 checklist（寫進論文 appendix）
- [ ] 所有 τ 來自 val，test 單次評估。
- [ ] 標準化統計來自 calib_train。
- [ ] paraphrase / discriminative prompt 模板在看 test 前凍結。
- [ ] phenotype 門檻在 val 校準。
- [ ] cross-base join 不洩漏 GT（cross_model_agreement 只用預測框 IoU，不用 GT）。

---

## 待補清單
- ✅ VIRO 引用已查證並接入（arXiv:2601.12781, CVPR'26, Park et al.）。
- `〔待補〕` VIRO 在 RefCOCO/gRefCOCO 各 split 的 no-target 數字（abstract 只給合併 61.1% balanced acc，需翻全文 PDF；取得後可進 §3.2 同表並列）。
- `〔待補〕` HieA2G no-target 官方數字（若同表並列需查證頁碼）。
- K（候選數）、paraphrase 數、softmax 溫度 → M0 跑通後回填 §2.3 meta。
