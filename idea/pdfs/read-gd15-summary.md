---
title: "Grounding DINO 1.5 詳細導讀"
author: "整理：kuo"
date: "2026-06-19"
---

# Grounding DINO 1.5 — Advance the "Edge" of Open-Set Object Detection（詳細導讀）

## 0. 論文基本資訊

- **論文標題**：Grounding DINO 1.5: Advance the "Edge" of Open-Set Object Detection
- **作者**：Tianhe Ren, Qing Jiang, Shilong Liu, ... Lei Zhang（IDEA Research Team）
- **單位**：International Digital Economy Academy (IDEA), IDEA Research
- **arXiv**：[2405.10300v2](https://arxiv.org/abs/2405.10300)（v2 提交於 2024-06-01，**Technical report**）
- **API/Repo**：<https://github.com/IDEA-Research/Grounding-DINO-1.5-API>（API-gated）
- **任務範疇**：open-set object detection（text-prompt → box）

> 註：reading-log 把這篇定為「新 base 實驗標的**首選**」（取代被排除的 VL-SAM-v3）。本篇任務＝評估它當 CRS 第四個 frozen base 的可行性，而非方法精讀。結論見 §4。與 [read-vlsamv3-summary.md](read-vlsamv3-summary.md)（排除原因）、後繼 [read-dino-x-summary.md](read-dino-x-summary.md)（次選）互讀。

---

## 1. 一句話定位

GD-1.5 是 Grounding DINO 的放大續作：把 vision backbone 換成 **ViT-L**、訓練資料擴到 **20M+ grounding 標註影像（Grounding-20M）**，分 **Pro**（高效能泛化）與 **Edge**（邊緣部署）兩款。核心架構沿用 GD 的 dual-encoder-single-decoder + GLIP 式「detection 當 phrase grounding」+ deep early fusion + box contrastive loss。

## 2. 兩個模型

### 2.1 GD 1.5 Pro
- ViT-L backbone + deep early fusion（decode 前就 cross-attention 融合語言/影像）。
- **early vs late fusion 的關鍵觀察（對 CRS 最有用）**：
  - early fusion → **higher detection recall + 更準的 box**，但 **更容易幻覺**（預測不存在的物件）。
  - late fusion → robust 抗幻覺，但 recall 較低（跨模態對齊較難）。
  - 折衷：保留 early fusion，但**訓練時提高負樣本比例**來平衡。

### 2.2 GD 1.5 Edge
- EfficientViT-L1 backbone + 新 efficient feature enhancer（只融合 P5 高階特徵、vanilla self-attention 取代 deformable、cross-scale fusion 補 P3/P4）。
- TensorRT 下 75.2 FPS；NVIDIA Orin NX 上 >10 FPS @640²。

## 3. 主要結果（皆 detection benchmark，非 REC）

| Benchmark | GD 1.5 Pro | 備註 |
|---|---|---|
| COCO zero-shot | **54.3 AP** | +1.8 over GD Swin-L |
| LVIS-minival zero-shot | **55.7 AP** | +6.9 over DetCLIPv3 |
| LVIS-val zero-shot | **47.6 AP** | +6.2 over 前 SOTA |
| ODinW35 / ODinW13 | 30.2 / 58.7 AP | open detection in the wild |
| GD 1.5 Edge LVIS-minival | 36.2 AP | 勝所有即時開集偵測器 |

⚠️ **全文無 RefCOCO/RefCOCO+/gRefCOCO/REC 評測**——它評的是 category/phrase 偵測（COCO/LVIS/ODinW），不是 referring expression comprehension。

---

## 4. 對使用者主線（CRS）的可行性裁定

> CRS 主線見 [[crs-pivot-conformal-referring-set]] / [[crs-litreview-15papers]]；現有 base＝OWL-ViT gate + GroundingDINO box。

### 4.1 當第四個 frozen base：**有條件可行，但比 reading-log 原本想像的更受限**

| 維度 | 裁定 |
|---|---|
| **介面相容** | ✅ 吃自然語言 text prompt、出 box，與現有 GD box base **同家族同介面**（GD→GD-1.5 是直系升級），整合風險低——這是 reading-log 選它為首選的主因，成立。 |
| **frozen 可用** | ⚠️ **部分**。論文是 technical report，**Pro 僅 API-gated（無公開權重）**；要當「凍結本地 base」跑 calibration/大量推論，得靠 API（成本/速率/可重現性風險）。Edge 也未明確釋出 checkpoint。對比現有 GroundingDINO（開源權重）這是退步。 |
| **分數異質性** | ✅ box contrastive score 與現有 GD 同源 → 但這也意味**與現有 GD box base 分數同質**，當「第四個異質 base」強化 cross-base 的價值有限（不如 DINO-X 換 CLIP text encoder 來得異質）。 |
| **REC 適配** | ⚠️ 無 RefCOCO 證據；GD 家族本身可做 REC，但要自行在 gRefCOCO 上驗證 no-target/multi-target 行為。 |

### 4.2 最有價值的 take-away（不只可行性）

**early-fusion → 高 recall 但高幻覺；late-fusion → 抗幻覺但低 recall** 這個 tradeoff 是 CRS motivation 的**外部佐證彈藥**：
- 它白紙黑字說明「更強的 grounding base（high recall）天生伴隨更多幻覺（FP）」——正好呼應 CRS「base 越強越需要在輸出端加 recall + abstention 雙保證」的核心論點。
- 可寫進 motivation：連 SOTA 偵測器作者都承認 recall/hallucination 是結構性 tradeoff，靠訓練負樣本只能「平衡」不能「保證」——這正是 CRS 用 LTT 給有限樣本保證要補的洞。

### 4.3 建議

1. **若要加第四 base 強化 cross-base 異質性 → 優先 DINO-X（CLIP text encoder，分數更異質）而非 GD-1.5**（GD-1.5 與現有 GD box 太同源）。
2. **GD-1.5 的最佳用途是 motivation 引用**（early/late fusion 的 recall-hallucination tradeoff），而非當 base。
3. 若仍要當 base，務必先確認**權重可得性**（API vs 本地），否則 frozen-base 的可重現性故事會被審稿人質疑。

**引用優先級**：中（motivation 引用 recall-hallucination tradeoff；當 base 退居次選，異質性不足 + API-gated）。

---

## 5. 一句話總結

> GD-1.5 是 Grounding DINO 的 ViT-L + 20M 資料放大續作（Pro/Edge 兩款），在 COCO/LVIS/ODinW 偵測 benchmark 刷新開集偵測 SOTA——但它是 **API-gated 的 technical report、無 RefCOCO/REC 評測、box score 與現有 GD base 同源**，當 CRS「第四個異質 frozen base」性價比不如 DINO-X。它真正對 CRS 有價值的是 **early-fusion 高 recall 必伴高幻覺** 的 tradeoff 自白，可當 CRS「base 越強越需 recall+abstention 雙保證」的 motivation 外部佐證。

---

## 6. 參考連結

- arXiv：<https://arxiv.org/abs/2405.10300>
- API/Repo：<https://github.com/IDEA-Research/Grounding-DINO-1.5-API>
- 前作 Grounding DINO：arXiv 2303.05499
- 後繼 DINO-X：[read-dino-x-summary.md](read-dino-x-summary.md)（arXiv 2411.14347）
