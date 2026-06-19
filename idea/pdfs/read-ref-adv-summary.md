---
title: "Ref-Adv: Exploring MLLM Visual Reasoning in Referring Expression Tasks 詳細導讀"
author: "整理：kuo"
date: "2026-06-19"
---

# Ref-Adv — Exploring MLLM Visual Reasoning in Referring Expression Tasks（詳細導讀）

## 0. 論文基本資訊

- **論文標題**：Ref-Adv: Exploring MLLM Visual Reasoning in Referring Expression Tasks
- **作者**：Qihua Dong, Kuo Yang, Lin Ju, Handong Zhao, Huimin Zeng, Jianglin Lu, Yitian Zhang, Yizhou Wang, Yun Fu（Northeastern University）
- **發表**：**ICLR 2026**（conference paper）
- **arXiv**：[2602.23898v1](https://arxiv.org/abs/2602.23898)（提交於 2026-02-27）
- **專案**：ref-adv.github.io
- **任務範疇**：抗捷徑（shortcut-suppressing）的現代 REC benchmark

> 註：reading-log 把這篇定為「benchmark 延伸：Ref-Adv (ICLR'26)」。對 CRS 的價值＝**「RefCOCO 已飽和、有水分」這個立論的最強最新實證**，直接支撐 CRS framing 守則「絕不踩準確率戰場」。與 [copsref-summary.md](copsref-summary.md)（controlled distractor 前身）、reading-log 核心問答「92.7% 有水分」互讀。

---

## 1. 核心主張

現代 MLLM 在 RefCOCO(+/g) 已達 **>90% 準確率近飽和**，但作者指出經典 REC benchmark 是**弱化的 reasoning 測試**，有三大缺陷讓模型靠捷徑得分：
1. **表達式太短**：RefCOCO/+ 平均約 3 詞 → 語言/視覺 reasoning 需求都低。
2. **distractor 太少**：多數影像只有 ~1 個同類 distractor → 只需推類別再從小候選集挑。
3. **冗餘描述子 → grounding shortcut**：長描述配少 distractor 時，模型只需匹配**部分**描述子即可定位，繞過對整句的理解（反常地導致長表達式準確率更高）。

→ Ref-Adv 把「語言不平凡的表達式」配上「**只給唯一識別目標所需的最小資訊**」+ **hard distractor**（部分符合但不完全滿足表達式的干擾物），壓掉捷徑。含 negation 等 reasoning facet 標註。

## 2. 資料構造（LLM-authored pipeline）

- 來源：COCO + OpenImages v7（取有 panoptic instance 標註者）。
- 四階段（Fig 3）：(a) 篩影像確保 ≥3 distractor + 標號；(b) GPT-4o 找最相似 pair、抽 group/instance 級 discriminator；(c) 用**最小充分子集** discriminator + optional negation 組表達式（明確避免單步 prompting 產生的 overspecified 冗餘）；(d) 人工驗證表達式正確性 + 確認 hard distractor 存在。
- 釋出 **Ref-Adv-s**：1,142 case 的可重現子集。
- 相較 RefCOCO(+/g)：表達式更長、詞彙更大、distractor 更多、**negation ratio 顯著更高**。

## 3. 關鍵實證（兩個 ablation 證明「真的需要 reasoning」）

1. **Bag-of-Words ablation（Table 3）**：把表達式打散成詞袋隨機排序。先前研究發現 RefCOCOg 打散後效能幾乎不掉（textual reasoning 需求弱）；**Ref-Adv 打散後顯著下降** → 證明它需要真正讀懂語序/語意。
2. **One-descriptor-deletion sufficiency（Table 4）**：隨機刪一個描述子再改寫。若刪了不影響效能 → 該描述子多餘 = 捷徑存在。**Ref-Adv 的 shortcut 遠少於 RefCOCO(+/g)**。

**主結果**：13 個當代 MLLM（閉源+開源）在 RefCOCO(+/g) 強，但**在 Ref-Adv 大幅下滑**，暴露對捷徑的依賴與 visual reasoning/grounding 的缺口。

---

## 4. 對使用者主線（CRS）的對位與借鏡

> CRS 主線見 [[crs-pivot-conformal-referring-set]] / [[crs-litreview-15papers]]。

### 4.1 最強用途：RefCOCO 飽和/有水分的最新權威實證

CRS reading-log 核心問答的守則之一是「**framing 絕不能踩準確率戰場**」，理由是「92.7% 有水分、RefCOCO 飽和有捷徑」。Ref-Adv（ICLR'26）是**這個論點的最強最新背書**：
- 可直接引用：「即使 MLLM 在 RefCOCO 達 >90%，在抗捷徑的 Ref-Adv 上大幅下滑（Dong et al., ICLR 2026），證明經典 REC 的高分有水分、靠 grounding shortcut。」
- → 支撐 CRS：「準確率數字本身不可信、需要的是輸出端的形式化保證」。把 Ref-Adv 放 motivation／related work 的「benchmark 飽和」段，比 COPS-Ref（2020）更新更有力。

### 4.2 與既有 motivation 鏈的合流

構成「準確率 ≠ 可靠性」的多層證據：
- **Ref-Adv**：benchmark 層——RefCOCO 高分靠捷徑，換硬 benchmark 就垮。
- **Grounding-Hallucination**（[read-grounding-hallucination-summary.md](read-grounding-hallucination-summary.md)）：訓練層——加 grounding 目標不減幻覺。
- **GD-1.5**（[read-gd15-summary.md](read-gd15-summary.md)）：架構層——高 recall 必伴高幻覺。
- **CRS**：→ 三層都解不掉，需輸出端有限樣本保證。

### 4.3 借鏡 / 注意

1. **hard distractor 概念**：Ref-Adv「部分符合但不完全滿足」的 hard distractor 是 CRS 評 referring set precision 的好壓力測試——CRS 的 abstention/no-target 在 hard distractor 下是否仍校準，值得補測。
2. **negation 軸**：Ref-Adv 高 negation ratio 是 CRS 沒特別處理的語言難點；若 CRS 要宣稱 robustness，negation query 上的行為可當附錄診斷。
3. **任務差異**：Ref-Adv 是 single-target REC benchmark（評測層），**不含 multi-target/no-target set prediction**（與 gRefCOCO/AgroVG 不同軸）。引用定位＝「RefCOCO 飽和的證據 + hard-distractor 壓力測試場」，不是 CRS 的 generalized-set 競品。⚠️ 注意作者之一也叫 Kuo Yang，與使用者同名但非同人，引用勿混。

**引用優先級**：中高（RefCOCO 飽和最新權威實證，motivation/framing 核心；ICLR'26 新鮮度高）。

---

## 5. 一句話總結

> Ref-Adv 是 **ICLR 2026 的抗捷徑現代 REC benchmark**：針對 RefCOCO(+/g) 的三大缺陷（表達式太短、distractor 太少、冗餘描述子可捷徑），用 LLM pipeline + hard distractor + 最小充分表達式 + negation 構造，並以 bag-of-words 與 descriptor-deletion 兩個 ablation 證明它**真的需要 reasoning**。13 個 MLLM 在 RefCOCO 達 >90% 卻在 Ref-Adv 大幅下滑——這是 CRS framing 守則「絕不踩準確率戰場、RefCOCO 高分有水分」的**最強最新實證**，與 Grounding-Hallucination（訓練層）、GD-1.5（架構層）接成「準確率 ≠ 可靠性」的完整證據鏈。

---

## 6. 參考連結

- arXiv：<https://arxiv.org/abs/2602.23898>
- 專案：<https://ref-adv.github.io/>
- 前身：COPS-Ref（[copsref-summary.md](copsref-summary.md)）；Akula et al. 2020（word-order 必要性測試）
