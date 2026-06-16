# 第 2 章　相關工作

本章把本論文定位在五條相鄰研究脈絡的交會處：(1) 凍結式指稱定位基礎模型、(2) 廣義 REC 與無目標／多目標處理、(3) 視覺—語言的選擇性預測與棄答、(4) verification-based 的零樣本 REC、(5) 保形預測（conformal prediction）與物件偵測的風險控制。本研究與每條脈絡都有重疊，但其交會點——**在凍結指稱定位基礎模型上做事後、風險受控、跨基礎模型可轉移的集合化決策**——尚未被既有工作占據。為避免不可核對的優先權宣稱，本章全程以可被審稿人核對的相對陳述描述差異，不使用「first／unique／novel framework」一類措辞。

## 2.1 凍結式指稱定位基礎模型

REC 的主流做法是訓練一個專用定位器。CLIP-VG 以對比式預訓練的 CLIP 為骨幹，透過 curriculum fine-tune 把 REC 準確率推高，輸出單一框的迴歸結果（`bbox_embed(reg_token).sigmoid()`），既無候選清單也無分數。OWL-ViT 是開放詞彙偵測器，對每個查詢輸出多個帶分數的候選框，分數是單次前向傳遞的免費副產品。RefFormer（NeurIPS'24）以 query-based 機制做 REC。GroundingDINO 則以 query-based 偵測產生高召回的候選框。

這些模型在本論文中扮演的角色不是「比較對象」，而是**被附加信心策略的凍結基礎模型**。我們選用 CLIP-VG（單框迴歸、無分數）與 OWL-ViT（多候選、有分數）兩個結構迥異者，正是為了壓力測試信心訊號是否與基礎模型架構無關（base-agnostic）；在 CRS 章再引入 GroundingDINO 作為高召回的 box selector。關鍵差異在於：上述工作都把重心放在「把基礎模型練得更準」，而本研究**不更新任何基礎模型參數**，只在其決策層做事後校準。

## 2.2 廣義 REC、無目標與多目標

標準 RefCOCO 系列假設每個描述恰有一個目標。gRefCOCO 與廣義 REC（GREC）打破此假設：一個描述可對應零（no-target）、一或多個物件。這需要模型具備棄答與計數能力。

代表性工作以**專門架構 + 標籤監督**達成此能力。LIHE 系在 gRefCOCO 上顯式建模零／多指稱物。HieA2G（AAAI'25, arXiv:2501.01416）引入 Adaptive Grounding Counter，以 trained counting head 搭配階層對齊，在 gRefCOCO val/testA/testB 取得 N-acc 56–60% 的 GREC 成績。

本研究與這條線的關鍵區別是路線而非目標：HieA2G／LIHE 為 no-target／counting 訓練專用模組；本研究的無目標閘（C3）以**事後信心**在**凍結**基礎模型上導出，不重訓、近乎零成本，可掛任意基礎模型。我們把這些 trained 方法當作**文獻定位的對照**（成本—reliability 光譜的重型一端），而非 accuracy 競賽的對手——兩者基礎模型不同，直接比絕對準確率無意義。

## 2.3 視覺—語言的選擇性預測與棄答

選擇性預測（selective prediction）／拒答選項（reject option）是成熟的框架：Geifman & El-Yaniv（2017）已建立深度選擇性分類與目標風險。在多模態上，ReCoVERR（Srinivasan et al., ACL Findings'24, arXiv:2402.15610）把選擇性預測用於 VQA——VLM 低信心時不直接棄答，改由 LLM 提相關問題蒐集高信心證據，在 VQAv2／A-OKVQA 上多答約 20% 而不掉準確率。

這條線對本研究有兩重意義。其一，它證明**風險—覆蓋／選擇性預測本身已是成熟框架**，因此本論文不把風險—覆蓋當作 novelty，而把貢獻放在「凍結定位基礎模型的不確定性結構是否可重用、可轉移」這個更具體、可證偽的問題上。其二，ReCoVERR 是 VQA 而非定位，且依賴外部 LLM 蒐證，與本研究「從凍結基礎模型自身副產品抽訊號 + 輕量校準器」的機制不同，僅作 motivation 與概念光譜的對照。對話式澄清（如 SIMMC2.0）則以多輪互動消解歧義，需使用者回覆；本研究是單輪自動的作答／棄答。

## 2.4 Verification-based 的零樣本 REC（最近鄰）

兩篇近期工作在 framing 上與本研究最接近，必須清楚區辨。

**VIRO**（Park et al., CVPR 2026, arXiv:2601.12781）在 neuro-symbolic REC pipeline 中，把查詢由 LLM/VLM 拆解成可執行 program，並在**每個運算子內嵌輕量 verifier**（Verification-Integrated Reasoning Operators），逐步驗證物件存在性與空間關係，以抑制 cascading error 造成的 high-confidence false positive；以 verification-aware abstention 處理 no-target，報 61.1% balanced accuracy。VIRO 同樣凍結其 LLM/VLM 組件，因此「動不動 base」**不是**區辨點。真正的差異在方法路線：VIRO 走程式拆解 + per-operator 符號驗證；本研究走 frozen base + 單一輕量事後校準器，無程式合成、無運算子級符號執行器，且 base-agnostic、近乎零訓練。

**True/False Verification REC**（Liu & Hu, arXiv:2509.09958）把 REC 重構成 box-wise 視覺—語言驗證：以 COCO-clean 的 generic detector（YOLO-World）產生 proposals，再讓通用 VLM 對**每個 region 獨立答 True/False**，無 REC-specific 訓練、無 fine-tuning，支援棄答與多匹配，且其 controlled study 主結論是「verification 顯著優於 selection-based prompting」。這直接威脅任何以「candidate verification 本身是新的」為賣點的方法。因此本研究的候選對比式介入（Variant B）**不主張 verification novelty**，僅作為上限分析（第 9 章外的補充）。關鍵區辨在於：True/False Verification 沒有 calibration／risk-coverage／belief gate，且把驗證外掛給通用 VLM 重問；本研究保留凍結基礎模型，把其**自身**不確定性校準成風險受控的作答／棄答／集合決策，並量化跨基礎模型可轉移性與 oracle gap——這些是其無法主張的。

## 2.5 保形預測與物件偵測的風險控制

保形預測（conformal prediction）與其推廣 Learn-then-Test（LTT, Angelopoulos et al.）提供分布無關、有限樣本的風險控制：給定校準資料，可構造出滿足指定風險上界的預測集合。在物件偵測上，已有 conformal object detection 工作對框座標或類別做集合化保證。在 VLM／視覺—語言上，亦有把 conformal 套到零樣本分類或檢索的嘗試。

本研究的主結果 CRS 與這條線最相關，但落在一個尚未被占據的交會點：

- 對象是**凍結指稱定位基礎模型的集合化決策**，而非偵測框座標或分類標籤的保形集合；
- 同時校準**三個**與 REC/GREC 語意對應的風險（已作答目標漏檢、無目標誤選、目標棄答），以 LTT + Bonferroni 給聯合有限樣本保證，而非單一風險；
- 以**跨基礎模型組合**（一個基礎模型當 gate、另一個當 box selector）把風險控制的代價因式分解成兩個正交瓶頸，這是把 conformal 與「凍結基礎模型的不確定性結構可重用」這個 C4 主軸縫合起來的結果。

> 「保證」一詞在本論文僅指 LTT 的有限樣本檢定；自助法（bootstrap）信賴區間一律稱為經驗穩定度（empirical stability），兩者全程分開報告、不混用。

## 2.6 差異定位小結

| 工作 | 任務 | 動 base？ | 棄答／無目標 | 機制路線 | 與本研究關係 |
|---|---|---|---|---|---|
| CLIP-VG | REC | 是（curriculum FT） | 否 | 把 base 練更準 | 凍結 base #1 |
| OWL-ViT | open-vocab 偵測 | —（預訓練） | 否 | open-vocab detector | 凍結 base #2，驗 base-agnostic |
| RefFormer (NeurIPS'24) | REC | 是 | 否 | query-based grounding | 可選第三 base |
| LIHE / gRefCOCO 系 | GREC | 是 | 部分（顯式建 0/多） | 專門架構 + 標籤監督 | post-hoc 對照 |
| HieA2G (AAAI'25) | GREC | 是 | 是（Adaptive Counter） | trained counting head | 成本—reliability 重型對照 |
| SIMMC2.0 clarification | 對話式 grounding | — | 類似（多輪澄清） | 對話 loop + 使用者回覆 | 需互動；本研究單輪自動 |
| VIRO (CVPR'26) | REC + no-target | 否（凍結組件） | 是（verification-aware） | LLM/VLM program + operator 內嵌符號 verifier | framing 最近、路線正交 |
| True/False Verification (2509.09958) | REC | 否（zero-shot） | 是（棄答＋多匹配） | YOLO-World proposals + 通用 VLM 逐 box 真假 | 最近鄰、威脅 B；無 calibration/gate |
| ReCoVERR (ACL Findings'24) | VQA（非 grounding） | — | 是（減少過度棄答） | 低信心時 LLM 提問蒐證 | 證 selective prediction 已成熟 |
| Conformal OD / VLM conformal | 偵測／分類 | —（多為後處理） | 集合化 | 保形集合保證 | CRS 的方法淵源，但對象與三風險不同 |
| **本研究（C1–C4 + CRS）** | REC + GREC | **否** | **是（核心）** | 凍結 base + 事後風險校準 belief policy；CRS 跨 base 保形組合 | — |
