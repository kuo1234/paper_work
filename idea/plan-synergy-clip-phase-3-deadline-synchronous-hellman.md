# 碩論主線轉向：Selective / Uncertainty-Aware Referring Grounding

## Context（為什麼做這個轉向）

**問題**：原 BTS 主線（OpenVLA + LIBERO 的 object-attribute binding）已撞上計畫 Phase 3 停損點。可部署的 image evidence module 在 LIBERO 模擬 render 上穩不起來，根因是現成 zero-shot detector（CLIP-crop、OWL-ViT）有**不可校正的 per-object score bias**（owl_errattr.py：零漏檢、錯誤全是 distractor 分數蓋過 target）。繼續往下深入不符成本效益。

**手上真正的資產**：不是某段程式，而是一個論點——**「structured belief 應決定『何時』介入，而非盲目替換 base 模型」**。先前在 LIBERO 已實證：always-on oracle gate 與 geometry gate 都會傷 success，必須 selective（見舊 repo 的 BTS_OPENVLA_INTERVENTION_REPORT）。

**轉向決定（已逐項拍板）**：把這個論點搬到一個**文獻空白、資料乾淨、CLIP 真正能用**的新場域——
> **研究問題：CLIP-family 的 zero-shot referring grounding / target selection，能否被「selective / uncertainty-aware」機制改進？這種改進是否真的提升選對 target 的能力？**

- 評測場域：**RefCOCO / RefCOCO+ / RefCOCOg**（真實影像 referring expression，避開 sim render 的 bias 停損根因）+ **gRefCOCO**（含 no-target / multi-target，提前當核心）。
- 約束：base 模型凍結、低訓練成本（zero-shot / 輕量 calibrator，不大規模 fine-tune）；只用開源資料集（拿不到真 digital twin）。
- 目標：碩論，deadline 2027-04，必須**乾淨正面結論**（不能是「勉強能用、限制一堆」）。

**已查證的關鍵事實**（arXiv 實查，非憑印象）：
- 「Synergy-CLIP」(arXiv:2504.21375) 是 vision+text+**audio** 三模態，與 grounding 無關——是停損計畫代號被誤當錨點，已**棄用**。
- Novelty 空白確認：uncertainty-aware / selective / abstention referring grounding **基本沒人做**（12 篇相關工作 0 篇做 calibration / selective-prediction / abstain）。處理 ambiguity 的走「生成更好描述」或「對話澄清」，無人從 belief 決定何時介入切入。

---

## 核心命題（鎖定正面結論的下限）

不打 SOTA accuracy 戰場（會輸給專門 fine-tune 模型，那不是本研究戰場）。改打**結構性命題**：

> 在固定的 base grounding 模型上，selective intervention 能在**不重訓 base、近零成本**下，把「答錯」轉成「棄答或消歧」，於是在 **risk-coverage 曲線上嚴格 dominate** 非 selective baseline；且在 **gRefCOCO 的 no-target 設定**下，belief gate 直接提供 base 本身沒有的 abstain 能力。

此命題的下限幾乎必為真（只要 uncertainty 訊號比亂猜好，selective prediction 理論保證 risk-coverage 改善）；上限（消歧多有效、跨 base 多通用、generalized REC 多強）是優化空間。

---

## 方法設計

### 問題形式化
給定影像 I、referring expression e。Base 模型輸出候選框 {b_k} 與分數 {s_k}（CLIP-VG / OWL-ViT 直接給）。標準系統回 argmax；**selective 系統**多一個 gate g ∈ {ANSWER, ABSTAIN, INTERVENE}，由 belief 訊號 u 決定。拆三個可獨立驗證的子問題：(1) 哪些 u 真的和「答對機率」相關 (2) u→g 規則 (3) INTERVENE 時做什麼。

### Belief / uncertainty 訊號（base 凍結、免費或低成本）
| 訊號 | 定義 | 成本 |
|---|---|---|
| Top-1 score | 最高分絕對值 | 免費 |
| Top1–Top2 margin | 最佳與次佳差距（小=有競爭=歧義） | 免費 |
| Score entropy | softmax 分布的熵 | 免費 |
| Spatial dispersion | top-m 框的幾何離散度 | 免費 |
| Cross-prompt consistency | 同義改寫 prompt 多次跑，top-1 框是否一致 | 中（多次 forward）|
| Cross-model agreement | CLIP-VG vs OWL-ViT top-1 是否一致 | 中高 |

**設計原則（直接回應 BTS 教訓）**：先用 calibration plot + 各訊號對「答對與否」的 AUROC 做 informativeness 篩選，**再**決定用哪些。此步是 M1 的 go/no-go。

### 三個 variant（由淺到深）
- **Variant A — Selective Abstention（安全網）**：單一最佳訊號 + 學一個閾值 τ，u<τ→ABSTAIN。產出 risk-coverage curve。只要 dominate naive，A 即成立。
- **Variant B — Learned Gate + Disambiguation（主貢獻）**：多訊號餵極輕量 calibrator（logistic / 小 MLP，數十參數，不碰 base）預測 p_correct。三態 gate：高 p→ANSWER；低 p 且 margin 極小→INTERVENE（test-time candidate re-ranking：對 top-m 生成更有區辨力 sub-prompt 重打分，或用 spatial-relation 解析重排）；低 p 且全低分→ABSTAIN。介入只在 gate 觸發時發生（selective 精神，平均推論成本幾乎不增）。
- **Variant C — Belief-aware Generalized Grounding（提前當核心）**：場域切 gRefCOCO。abstain 不再是工程安全閥，而是**任務本身要求的正確輸出**（no-target=該答「沒有」）。gate 輸出結構化 belief：P(no-target)、referent 數量估計 n̂。用 belief 框架統一處理「該答幾個 / 該不該答」——reviewer 最買單的故事。用 GREC 官方指標（N-acc、T-acc、Pr@(F1=1,IoU≥0.5)）。

遞進：A 證訊號有用、B 證介入有用、C 證框架推廣到任務本質。後層失敗不傷前層。

---

## Novelty 定位

> ⚠️ 2026-06-11 GPT 紅隊第二輪後重大修正。**禁用 first/unique/唯一**。VIRO 已 CVPR'26 發表、True/False Verification REC (arXiv:2509.09958) 已做 zero-shot verification+abstention，「新 grounding/verification 方法」這條線已被壓住。完整修正與查證引用見 [[thesis_selective_grounding_spec]] §1.1b/§1.3/§1.5/§1.6。

**Framing**：本研究是 **frozen grounding base 的 post-hoc reliability / calibration study**，不是新 grounding 方法。可守的一句話 claim：

> This thesis studies whether frozen zero-shot grounding models expose reusable, grounding-specific uncertainty signals that can be calibrated post hoc into reliable answer / abstain / re-rank decisions — quantifying reliability, cost, transferability, and oracle gap, rather than competing with trained GREC architectures or neuro-symbolic verification pipelines.

**差異表**（取代舊表；完整含 VIRO/True-False/ReCoVERR/HieA2G 細數見 spec §1.3/§3.4/§3.5）：
| 工作 | 處理歧義方式 | 動 base？ | selective/abstain？ | 與本研究差異 |
|---|---|---|---|---|
| CLIP-VG | curriculum fine-tune | 是 | 否 | 把 base 練更準；我們固定 base 決定何時信它（可當我的 base）|
| OWL-ViT | open-vocab detection | — | 否 | 同上，當第二 base |
| LIHE / gRefCOCO 系 | 顯式建模 0/multi referent | 是 | 部分 | no-target 當分類標籤專訓；我用 post-hoc belief 導出、可掛任意 base、近零訓練 |
| HieA2G (AAAI'25) | trained Adaptive Grounding Counter | 是 | 是 | 全監督 counting head；我 post-hoc 近零成本（trained upper reference）|
| VIRO (CVPR'26) | program 拆解 + per-operator 符號驗證 | 否 | 是 | 我無 program/無符號執行器，單一輕量 calibrator |
| True/False Verification REC (2509.09958) | box-wise VLM True/False | 否 | 是 | 它外掛 general-purpose VLM 取代 grounding、無 calibration；我留 frozen base 校準其自身不確定性 |
| ReCoVERR (ACL'24) | 低信心蒐 evidence 減 over-abstain | — | 是 | VQA 非 grounding；證 selective prediction 已成熟→risk-coverage 本身非 novelty |
| **本研究** | post-hoc belief→selective gate | **否** | **是（核心）** | base-agnostic + post-hoc + risk-calibrated + cross-base transfer + oracle gap |

**防守（更新）**：(1) learned multi-signal gate + 條件式介入，非單一 threshold (2) Variant C 連到任務本質 (3) **cross-base transfer**（train-on-A/test-on-B，非各訓各測）才是真區辨——VIRO/True-False 都 per-pipeline 無 transfer 主張 (4) 量化 oracle gap / cost–risk Pareto，賣 reliability study 不賣 accuracy。單純「比 VIRO 輕」不算貢獻，「輕量+risk-calibrated+cross-base+oracle-gap」整組才算。

**最危險近鄰**：True/False Verification REC (2509.09958)——直接威脅 B（candidate verification 不能再賣新）。B 因此降為 Chapter 5 上限，主論文核心 = A+C+calibration+oracle-gap+cross-base（spec §1.6 四貢獻 C1–C4）。

---

## Baseline Stack
| 層級 | Baseline | 角色 |
|---|---|---|
| 0 下界 | Vanilla CLIP + region proposals argmax | 證 base 會錯、有 selective 空間 |
| 1 主力 base | **CLIP-VG**（開源 github.com/linhuixiao/CLIP-VG）| selective 掛它上面 |
| 2 主力 base | **OWL-ViT** open-vocab | 第二 base，證 base-agnostic |
| 3 第三 base | RefFormer (NeurIPS'24) | 通用性加碼（adapter 復現有風險，列後）|
| 4 naive selective | 固定 threshold（單一 top-1 score）| 稻草人，證 learned gate 勝它 |
| 5 naive selective | Random abstention | risk-coverage 最低參考線 |
| 6 上界 | Oracle gate（用 GT 決定何時棄/選誰）| 理論天花板，量「訊號還差多少」|

---

## 評測指標
- **標準 grounding**（復現+base 比較）：Accuracy@IoU=0.5（RefCOCO/+/g val/testA/testB）。
- **Selective prediction（核心）**：Risk–Coverage curve（招牌圖）、AURC/E-AURC、AUSE、Selective accuracy @ fixed coverage、ECE/reliability diagram、Coverage @ target risk。
- **Generalized REC（Variant C）**：N-acc（no-target）、T-acc、Pr@(F1=1, IoU≥0.5)。
- **介入有效性（B）**：intervention lift（介入子集 accuracy 差）、觸發率 / 平均額外成本。

---

## 里程碑與 go/no-go（對齊 10 個月）

| 期間 | 月 | 工作 | 門檻 |
|---|---|---|---|
| 2026-06中–07中 | M0 | 跑通 CLIP-VG + vanilla CLIP 在 RefCOCO，復現論文數字；建 candidate+score 離線 dump pipeline | 復現成功 |
| 2026-07中–08中 | **M1** | 對 dump 算所有 belief 訊號的 calibration + AUROC；做 Variant A 第一張 risk-coverage | ⭐**GO/NO-GO #1**：至少一訊號 AUROC≥~0.65 → 主題定案（**兩個月內**）|
| 2026-08中–10 | M2 | Variant B：learned gate + disambiguation | **GO/NO-GO #2**：介入有 lift？否則退回 A 為主貢獻 |
| 2026-10–12 | M3 | base-agnostic：同一 gate 掛 OWL-ViT/RefFormer | 跨 base 一致改進主表 |
| 2026-12–2027-02 | **M4** | **Variant C / gRefCOCO（核心章節）**：no-target/multi-target，GREC 指標 | belief gate 的 N-acc vs base（base 無法 abstain，壓倒性對比）|
| 2027-01–02 | M5 | （可選）CROG/OCID manipulation 佐證，真實桌面影像 | 僅進度超前時做 |
| 2027-02–04 | M6 | ablation、oracle gap、failure 分析、寫作、口試 | 完整論文 |

**緩衝**：M5 可丟、M3 可縮。不可妥協是 M0–M2 + M4，2026 年底前完成主體，留近 4 個月寫作。

---

## 如何保證「乾淨正面結論」
- **最低成功線（幾乎保證）**：Variant A 的 risk-coverage 在 RefCOCO 嚴格 dominate naive + random，AURC 顯著較低。M1 GO 即鎖定。
- **理想成功線**：A + B + base-agnostic + gRefCOCO C 章節齊備。
- **核心圖表（第一天就為它收資料）**：(1) 跨 3 base 的 risk-coverage curves（selective vs naive vs random vs oracle）(2) 各訊號 AUROC/AUSE + calibration (3) 介入 lift (4) gRefCOCO N-acc 表 (5) oracle gap + failure 分析。
- **誠實護欄**：不宣稱 SOTA accuracy，明說「我們改的是何時信 base，不是 base 天花板」（寫成 deliberate scope）；主動揭露 oracle gap；所有改進報統計顯著性（多 seed/bootstrap CI）。

---

## 最大風險與對策
**頭號風險：uncertainty 訊號不 informative（與 BTS 的 score bias 同源）。**
- **前置驗證**：M1（第 8 週前）只用 dump 算 AUROC 即可判定，**不需任何訓練**。絕不拖到方法做完才發現訊號沒用。
- **多層 fallback**：(1) 升級到 cross-prompt consistency（擾動一致性訊號通常最強）(2) 換 base（不同 base 校準差很多）(3) 移重心到 gRefCOCO no-target（「圖裡根本沒有→全低分」比「兩候選誰對」更強的訊號）(4) 最終防線：系統性證明「CLIP-family grounding 的 uncertainty 為何不可靠 + oracle 上界」，仍是對社群有用的結果。

其他：「只是 threshold」→ 四點防守；介入無 lift→退回 A；CLIP-VG/RefFormer 復現難→M0 先驗，RefFormer 可砍；gRefCOCO 超時→C 降為概念驗證表；資源→全離線 dump 分析、base forward 只跑一次。

---

## 驗證方式（end-to-end）
1. **M0 復現**：在 RefCOCO val 跑 CLIP-VG，比對論文 accuracy（±合理誤差）→ 確認 pipeline 忠實。
2. **M1 訊號驗證**：dump 上算各訊號 AUROC + 畫 calibration plot → 數字證明訊號 informative（go/no-go 客觀判定）。
3. **Variant A**：畫 risk-coverage curve，與 random/fixed-threshold 比 AURC → 證 selective dominate。
4. **Variant B**：歧義子集上 accuracy(介入) vs argmax → 證介入 lift。
5. **M3**：同 gate 掛 OWL-ViT/RefFormer，重跑 risk-coverage → 證 base-agnostic。
6. **M4 (gRefCOCO)**：GREC 官方指標 N-acc/T-acc/Pr → 證 belief 框架在 generalized REC 的正面結論。
7. 全程多 seed + bootstrap CI 報顯著性。

---

## 銜接與待辦
- 新方向另起 decision-doc（建議 `idea/thesis_selective_grounding_plan.md`），與舊線 `idea/research_direction_options.md`、`idea/innovation_notes.md`（robot play + offline meta-RL）**不混入**。
- BTS 的「先驗證 trigger 訊號是否 informative 再做機制」方法論直接複用到 M1（舊 repo 已抽出為 Desktop/bts-poc 獨立 repo）。
- `idea/paper/VariBAD.md` 提供 belief / uncertainty 理論語彙。
- 第一個動作：M0 環境建置——確認可用 GPU（spark GB10 或 lab A6000）、建 RefCOCO 資料與 CLIP-VG 復現環境（新 venv）。
