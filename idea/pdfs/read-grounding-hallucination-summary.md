---
title: "Does Object Grounding Really Reduce Hallucination of LVLMs? 詳細導讀"
author: "整理：kuo"
date: "2026-06-19"
---

# Does Object Grounding Really Reduce Hallucination of Large Vision-Language Models?（詳細導讀）

## 0. 論文基本資訊

- **論文標題**：Does Object Grounding Really Reduce Hallucination of Large Vision-Language Models?
- **作者**：Gregor Geigle, Radu Timofte, Goran Glavaš
- **單位**：University of Würzburg（WüNLP / Computer Vision Lab, CAIDAS）
- **arXiv**：[2406.14492v1](https://arxiv.org/abs/2406.14492)（提交於 2024-06-20；後收錄於 EMNLP 2024 Findings）
- **任務範疇**：實證檢驗「grounding 目標能否減少 LVLM 物件幻覺」這個流行宣稱

> 註：reading-log 把這篇定為「framing/動機」標的。它對 CRS 的價值＝**一篇打臉「grounding 自動帶來可靠性」的實證論文**，正面背書 CRS「光靠更強 grounding 不夠、必須在輸出端加保證」的立論。與 [read-gd15-summary.md](read-gd15-summary.md)（recall-hallucination tradeoff）、[read-dino-x-summary.md](read-dino-x-summary.md) 的 motivation 鏈合流。

---

## 1. 核心主張與動機

### 1.1 被檢驗的宣稱

近期一系列工作（Chen et al. 2023b、You et al.、Pramanick et al.）宣稱：把 **grounding 目標**（如 referring expressions、grounded captioning）加進 LVLM 訓練能**減少物件幻覺**。直覺上合理——region-level 目標要求比 global captioning 更細粒度的影像理解，應該會抑制「無中生有」。

### 1.2 本文的質疑：這個宣稱建立在有瑕疵的評估協議上

作者指出既有「減幻覺」證據有兩大缺陷：
1. **資料污染**：都用 **MSCOCO**——而 MSCOCO 早已大量進入 LVLM 訓練混合，模型對它先天較不幻覺 → **低估真實幻覺**。
2. **評估方式失準**：用 **QA 式**（POPE：「圖中有 X 嗎？」yes/no）衡量，而非**開放式 caption 生成**——QA 是「未經驗證的 proxy」，與真實世界自由生成任務對不上。

### 1.3 本文貢獻：第一個健全協議下的系統分析

- 控制變因：同一架構/訓練流程，只切換 grounding 資料的有無（Base vs +RE vs +GC vs +RE+GC）。
- **三個 LLM backbone**（Vicuna-1.5-7B、Llama-3-8B、Phi-3-mini）增強可轉移性。
- **OOD 評估**：除 MSCOCO 外加 **Objects365**（365 類，含 80 COCO 類）作分布外資料。
- **雙幻覺指標互補**：CHAIR（+ 自提的語意版 **CHAIR-MEN**，用句子編碼器 cosine 取代字串比對，可擴到大類別集）+ FaithScore（LLM 抽 atomic facts → VQA 驗證）。並用 CIDEr/CLIPScore/coverage 監控 caption 資訊量。

---

## 2. 核心發現

> **三個 backbone 一致：在健全協議下，grounding 目標（RE/GC）對開放式 caption 的物件幻覺幾乎無影響（little to no effect），QA 與開放生成皆然。**

細節：
1. **訓練時加 grounding 目標** → CHAIRi、FaithScore、caption 品質（CIDEr/CLIPScore）、資訊量幾乎都沒差（Table 3）。
2. **推論時強制生成 grounded caption**（caption 中插入 box 座標）→ **略減**幻覺，但效果小，且**犧牲 caption 細緻度**（faithfulness ↔ informativeness tradeoff）。
3. **定性檢查**：逼模型為提到的物件生 box，**多數情況仍無法阻止它幻覺內容**（生了 box 還是亂講）。
4. **方法論收穫**：CHAIR-MEN 與 vanilla CHAIR 高度成比例 → 驗證語意匹配可作為大類別集（如 Objects365）的可擴替代。

結論：grounding 目標**未能有意義地降低 LVLM 幻覺**，呼籲新的減幻覺方法。

---

## 3. 對使用者主線（CRS）的對位與借鏡

> CRS 主線見 [[crs-pivot-conformal-referring-set]] / [[crs-litreview-15papers]]。

### 3.1 最強用途：motivation 的「反面背書」

這篇是 CRS framing 的**核心彈藥**：
- 它**實證**「把 grounding/referring 目標塞進訓練**不會**自動帶來可靠性（不減幻覺）」。
- → 直接支撐 CRS 立論：**「可靠性不是靠更強/更多 grounding 訓練就能得到的；必須在輸出端外加分布無關保證（LTT recall+abstention）。」**
- 可寫進 introduction/motivation：「即使是專門設計的 grounding 目標，在健全評估下都無法保證模型不幻覺（Geigle et al. 2024）；這說明 reliability 是一個正交於 grounding 能力的問題，需要 post-hoc 的形式化保證來補。」

### 3.2 與既有閱讀的 motivation 鏈

構成一條完整的「grounding ≠ reliability」論證鏈：
- **GD-1.5**（[read-gd15-summary.md](read-gd15-summary.md)）：偵測器作者自承 early-fusion 高 recall 必伴高幻覺（架構層 tradeoff）。
- **本文**：訓練目標層——加 grounding objective 也不減幻覺（訓練層無效）。
- **CRS**：→ 既然架構與訓練都解不掉，只能在**輸出端**加有限樣本保證。

### 3.3 方法論借鏡

1. **健全評估協議的警示**：本文證明「在訓練見過的資料（MSCOCO）上評估會低估失效」。**CRS 必須注意 RefCOCO 飽和/捷徑問題**（呼應 reading-log 核心問答「92.7% 有水分」），且最好補 OOD / 跨域資料（正好接到 AgroVG 跨域 GREC，見後續 [read-agrovg-summary.md](read-agrovg-summary.md)）。
2. **faithfulness ↔ informativeness tradeoff**：與 CRS 的 recall ↔ set-size / abstention tradeoff 同構——「為了不幻覺而棄答/縮集合」會犧牲資訊量，CRS 的 Pareto 曲線（[[crs-redteam-p0-fixes]] 的 crs_pareto.png）正是量化這個取捨，本文給了同型 tradeoff 的外部佐證。
3. **CHAIR-MEN 的語意匹配思路**：CRS 若需在開放詞彙下判斷「referring set 是否命中正解」，語意 cosine 匹配（取代 exact match）是可借鏡的 nonconformity 設計（但 CRS 主要用 IoU/box，影響有限）。

### 3.4 引用注意

- 本文談的是 **LVLM caption 幻覺**（生成式），不是 REC box 定位的可靠性——**任務不同**。引用時要框成「reliability 與 grounding 能力正交」的**類比/motivation**，不可暗示它直接量測了 REC 的保證問題（否則被審稿人抓 over-claim）。

**引用優先級**：中高（motivation 核心反面背書；framing 用，非方法/競品）。

---

## 4. 一句話總結

> 這篇用三個 LLM backbone + OOD 資料（Objects365）+ 雙互補指標（CHAIR-MEN/FaithScore）的健全協議，**實證打臉「把 grounding/referring 目標加進 LVLM 訓練能減少幻覺」這個流行宣稱**——在開放式 caption 生成下，grounding 目標對物件幻覺幾乎無影響，推論時強制 grounded caption 也只略減且犧牲細緻度。對 CRS 而言這是 motivation 的核心反面背書：**reliability 正交於 grounding 能力，光靠更強/更多 grounding 訓練得不到，必須在輸出端外加分布無關保證**——與 GD-1.5 的架構層 recall-hallucination tradeoff 接成完整論證鏈。

---

## 5. 參考連結

- arXiv：<https://arxiv.org/abs/2406.14492>
- CHAIR：Rohrbach et al. 2018；FaithScore：Jing et al. 2023；POPE：Li et al. 2023b
- 關聯：[read-gd15-summary.md](read-gd15-summary.md)（recall-hallucination tradeoff）
