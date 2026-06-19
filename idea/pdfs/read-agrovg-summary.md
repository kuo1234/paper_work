---
title: "AgroVG: Multi-Source Benchmark for Agricultural Visual Grounding 詳細導讀"
author: "整理：kuo"
date: "2026-06-19"
---

# AgroVG — A Large-Scale Multi-Source Benchmark for Agricultural Visual Grounding（詳細導讀）

## 0. 論文基本資訊

- **論文標題**：AgroVG: A Large-Scale Multi-Source Benchmark for Agricultural Visual Grounding
- **作者**：Haocheng Li, Juepeng Zheng, ... Jianxi Huang（China Agricultural University / Sun Yat-sen / Tsinghua 等）
- **arXiv**：[2605.22034v1](https://arxiv.org/abs/2605.22034)（提交於 2026-05-21，**Preprint**）
- **資料/程式**：anonymous.4open.science/r/AgroVG-5172/
- **任務範疇**：跨域（農業）generalized visual grounding benchmark（set prediction：single/multi/target-absent）

> 註：reading-log 把這篇定為「benchmark 延伸：跨域 GREC」。對 CRS 的價值＝**現成的第三/跨域資料集**，可在農業影像上驗證 CRS 的 cross-base 與 abstention 保證是否 domain-agnostic。與 [grec-summary.md](grec-summary.md)（gRefCOCO/GREC 主場）、[copsref-summary.md](copsref-summary.md)（controlled distractor）互讀。

---

## 1. 一句話定位

AgroVG 把**農業視覺 grounding 形式化為 generalized set prediction**（與 gRefCOCO/GREC 同款）：給影像 + referring expression，模型須回傳**所有**符合的 target instance，**或在無目標時棄答**。10,071 個 annotation-grounded image-query pair，來自 **10 個來源資料集**、橫跨 **6 個 target family**（crop/weed、fruit、wheat head、pest、plant disease、tree canopy）。

## 2. 為何農業 grounding 與一般域不同（benchmark 設計動機）

三個特性使它區別於 RefCOCO 系列：
1. **密集重複實例**：一幀可有數十個視覺相似的麥穗/果實/交錯作物雜草 → referring expression **不能假設唯一目標**（天生 multi-target）。
2. **小、遮擋、不規則形狀**：害蟲、病斑、葉片常小且非凸 → box 不足以忠實評估 → 需 **instance-mask grounding**。
3. **target-absent expression**：田間指令常指不在畫面的物件 → 需**棄答**而非幻覺。

## 3. Benchmark 結構

- **兩任務**：T1 = bounding-box grounding（6 family 全覆蓋，9 來源）；T2 = instance-mask grounding（5 mask-capable 來源，3 family）。
- **三 regime × 兩任務 = 6 診斷 cell**：single（K=1）/ multi（K>1）/ target-absent（K=0）。
- **annotation-grounded query construction**：每個 positive query 可追溯到驗證過的 target set 與物件 identity；每個 target-absent query 都驗證過在 benchmark 標註下確為空集。
- **never synthesize masks from box geometry alone**（mask 須有像素級證據）。

## 4. 評估協議（對 CRS 最相關）

- **T1 box-set matching**：對每個 IoU 門檻 τ∈{0.50,0.75}，在 GT 框集與預測框集間建二分圖（IoU≥τ 連邊），求 **maximum-cardinality matching**（最大化滿足門檻的配對數，非 IoU 總和）。matched=TP、未配對預測=FP、未配對 GT=FN → query-level precision/recall/**Set-F1**。主指標＝macro Set-F1@{.5,.75}。
- **T2 query-level mask coverage**：GT mask 由 query 的 target instance ID 聯集成 query-level binary mask，預測也二值化後比對。
- 報告含 single（S-Acc）/ multi（M-F1）/ empty（E-Acc）診斷。

## 5. 主要發現（zero-shot，26 個模型配置）

> AgroVG 遠未飽和：**最佳 multi-target Set-F1 僅 0.35**；最佳 positive-query mask success rate @IoU0.75 **<0.17**。
- 涵蓋 closed-source MLLM、open-source VLM、specialized grounding system 三類。
- **棄答校準差**：對 target-absent query 不是幻覺出框/mask，就是對 positive query 過度棄答——**沒有任何模型把 existence-aware abstention 做好**。

---

## 6. 對使用者主線（CRS）的對位與借鏡

> CRS 主線見 [[crs-pivot-conformal-referring-set]] / [[crs-litreview-15papers]]；現有錨＝RefCOCO/gRefCOCO。

### 6.1 最強用途：跨域第三資料集 + abstention 痛點外部佐證

1. **cross-domain 驗證 base-agnostic**：CRS 目前錨在 RefCOCO/gRefCOCO（自然影像）。AgroVG 提供**現成的跨域 GREC-style 資料**（農業、小目標、密集 multi-target），可作 robustness 附錄：「CRS 的 LTT recall+abstention 保證在 domain shift 下是否仍成立」。這比 reading-log 早先考慮的 COPS-Ref 更新、更貼 generalized 設定。
2. **「棄答校準差」是 CRS 的正中靶心**：AgroVG headline 之一就是「沒有模型把 existence-aware abstention 做好（幻覺 or 過度棄答）」——這正是 CRS 用 LTT 給 abstention 保證要解的問題，可直接引為「abstention 是 open 痛點、且跨域更嚴重」的證據（呼應 [[crs-pivot-conformal-selective-grounding]] 的 forced-output N-acc 0.18→0.94 結構性 gap）。
3. **Set-F1 協議與 CRS metric 對齊**：AgroVG 的 box-set matching（二分圖 max-cardinality matching → Set-F1）與 CRS 的 referring-set recall/precision 量法**高度一致**——可借鏡它的 maximum-cardinality matching 作為 CRS multi-target set 評分的標準化協議，也讓 CRS 結果能與 AgroVG 並列比較。

### 6.2 借鏡 / 注意

1. **annotation-grounded + 驗證空集**的 query 構造：CRS 在自建/篩選 no-target 樣本時，AgroVG「每個 target-absent query 都驗證過確為空集」是好的品質標準（避免假 no-target 污染 abstention 校準）。
2. **mask 不從 box geometry 合成**：若 CRS 未來擴到 RES（referring segmentation），這條原則值得遵守。
3. **任務差異**：AgroVG 是 benchmark（評測層），非方法/競品。引用定位＝「跨域 robustness 測試場 + abstention 痛點佐證」，不是要 beat 它的數字。

**引用優先級**：中（跨域 robustness 資料集 + abstention 痛點外部佐證；benchmark 非競品）。

---

## 7. 一句話總結

> AgroVG 是 **2026 年把農業視覺 grounding 形式化為 generalized set prediction（single/multi/target-absent）的跨域 benchmark**：10,071 pair、6 target family、box(T1)+mask(T2) 雙協議，用二分圖 max-cardinality matching 算 Set-F1。zero-shot 26 配置遠未飽和（最佳 multi-target Set-F1 僅 0.35），且**所有模型的 existence-aware abstention 都做不好**——這恰是 CRS 用 LTT 給 abstention 保證的正中靶心。對 CRS 而言它是現成的**跨域 robustness 測試場** + **abstention 是 open 痛點**的外部佐證，且 Set-F1 box-set matching 協議與 CRS referring-set 量法高度一致可直接對齊。

---

## 8. 參考連結

- arXiv：<https://arxiv.org/abs/2605.22034>
- 資料/程式：<https://anonymous.4open.science/r/AgroVG-5172/>
- 前身概念：gRefCOCO/GREC（[grec-summary.md](grec-summary.md)）；gRef-CW（單來源農業 grounding）
