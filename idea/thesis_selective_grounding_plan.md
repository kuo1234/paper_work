---
type: decision-doc
aliases:
  - "Selective Grounding thesis plan"
  - "uncertainty-aware referring grounding"
  - "碩論主線 selective grounding"
tags:
  - 研究主線
  - 決策
  - selective-prediction
  - referring-grounding
  - CLIP
  - belief
  - uncertainty
summary: "撞 BTS Phase 3 停損後，碩論主線轉向 selective/uncertainty-aware referring grounding（RefCOCO + gRefCOCO），承接 BTS『belief 決定何時介入』核心 idea。"
---

# 碩論主線轉向：Selective / Uncertainty-Aware Referring Grounding（2026-06-11）

> 狀態：**已定案並通過計畫審核**。本文件為決策紀錄，執行細節見計畫檔
> `~/.claude/plans/plan-synergy-clip-phase-3-deadline-synchronous-hellman.md`。
> Deadline 2027-04。目標：碩論，乾淨正面結論。

---

## 0. 為什麼轉向

原 BTS 主線（OpenVLA + LIBERO object-attribute binding）撞上計畫 Phase 3 停損點：
可部署的 image evidence module 在 LIBERO 模擬 render 上穩不起來，根因是現成 zero-shot
detector（CLIP-crop、OWL-ViT）有**不可校正的 per-object score bias**（owl_errattr.py：
零漏檢、錯誤全是 distractor 分數蓋過 target）。繼續深入邊際價值低。

**手上真正的資產**：不是程式，而是論點——**「structured belief 應決定『何時』介入，
而非盲目替換 base 模型」**。LIBERO 已實證 always-on / geometry gate 都會傷 success，
必須 selective（見舊 repo `bts-poc/experiments/BTS_OPENVLA_INTERVENTION_REPORT.md`）。

把這論點搬到**文獻空白、資料乾淨、CLIP 真正能用**的新場域。

---

## 1. 研究問題

> **CLIP-family 的 zero-shot referring grounding / target selection，能否被
> 「selective / uncertainty-aware」機制改進？這種改進是否真的提升選對 target 的能力？**

- 場域：RefCOCO / RefCOCO+ / RefCOCOg（真實影像 referring expression）+ gRefCOCO（no-target / multi-target，提前當核心）。
- 約束：base 凍結、低訓練成本（zero-shot / 輕量 calibrator）、只用開源資料集。

---

## 2. 已查證的關鍵事實（arXiv 實查）

- **「Synergy-CLIP」(arXiv:2504.21375) 已棄用**：它是 vision+text+**audio** 三模態
  （VGG-sound+，做 zero-shot 分類 / 缺失模態重建），**完全沒碰 grounding**。
  第三模態 audio 在 grounding 任務無對應物。此詞源自停損計畫代號被誤當錨點。
- **Novelty 空白確認**：uncertainty-aware / selective / abstention referring grounding
  基本沒人做（12 篇相關 0 篇做 calibration / selective-prediction / abstain）。
  處理 ambiguity 的走「生成更好描述」或「對話澄清」(SIMMC2.0)、或 zero/multi-referent
  設定 (LIHE on gRefCOCO)，**無人從 belief 決定何時介入切入**。
- 真錨點：CLIP-VG（開源 github.com/linhuixiao/CLIP-VG）、OWL-ViT、RefFormer(NeurIPS'24)。

---

## 3. 核心命題（鎖正面結論下限）

不打 SOTA accuracy 戰場。改打結構性命題：

> base 凍結下，selective intervention 在不重訓、近零成本下把「答錯」轉成「棄答/消歧」，
> 於 risk-coverage 曲線嚴格 dominate 非 selective baseline；gRefCOCO no-target 設定下，
> belief gate 直接提供 base 沒有的 abstain 能力。

下限幾乎必為真（uncertainty 訊號比亂猜好 → selective prediction 理論保證 RC 改善）。

---

## 4. 三個 variant

- **A — Selective Abstention（安全網）**：單一最佳訊號 + 閾值 τ → ABSTAIN。risk-coverage curve。
- **B — Learned Gate + Disambiguation（主貢獻）**：多訊號 → 輕量 calibrator 預測 p_correct；
  三態 gate（ANSWER / INTERVENE 消歧 / ABSTAIN）；介入只在 gate 觸發時發生。
- **C — Belief-aware Generalized Grounding（拍板提前當核心）**：gRefCOCO，abstain 是任務
  本身要求的正確輸出；gate 輸出 P(no-target)、referent 數量 n̂。用 GREC 指標。

belief 訊號（base 免費取得）：top-1 score、top1-top2 margin、score entropy、spatial
dispersion、cross-prompt consistency、cross-model agreement。
**設計原則**：先算各訊號對「答對與否」的 AUROC 做 informativeness 篩選，再決定用哪些。

---

## 5. 關鍵決策紀錄（用戶逐項拍板）

| 決策點 | 結論 |
|---|---|
| 主線方向 | 改錨 CLIP grounding（非 Synergy-CLIP、非外部 detector、非收診斷論文） |
| 評測場域 | RefCOCO referring expression（真實影像，避開 sim render bias） |
| Variant C (gRefCOCO) | **提前當核心章節**（非 stretch） |
| manipulation (CROG/OCID) | **保留為可選延伸**（M5，僅進度超前才做） |
| 主力 base | CLIP-VG + OWL-ViT；RefFormer 第三（復現有風險） |
| 論文層級 | 碩論，要乾淨正面結論 |

---

## 6. 最大風險

**uncertainty 訊號不 informative（與 BTS score bias 同源）。**
前置到 M1（第 8 週前）只用 dump 算 AUROC 即可 go/no-go，**不需訓練**。
fallback：cross-prompt consistency → 換 base → 移重心 gRefCOCO no-target → 最終防線
「系統性證明 CLIP grounding uncertainty 為何不可靠 + oracle 上界」。

---

## 7. 與舊線關係

- 棄：3D-DA/CALVIN、toy varibad、meta-RL+language（[[research_direction_options]]、
  [[innovation_notes]] 的舊主線），與本方向不混入。
- BTS「先驗證 trigger 訊號是否 informative 再做機制」方法論直接複用到 M1。
  舊 repo 已抽出為 Desktop/bts-poc 獨立 repo。
- [[VariBAD]] 提供 belief / uncertainty 理論語彙。
