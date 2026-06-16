# 第 4 章　實驗設定

本章界定全篇實驗的共同基礎：資料集、凍結基礎模型、資料切分與校準／閾值協定、評測指標，以及防止資訊洩漏的 checklist。所有結果章（C1–C4、M4、CRS）皆遵循本章協定；章節內若有額外設定，會明確標註並仍服從此處的鐵律。

## 4.1 資料集

- **RefCOCO / RefCOCO+ / RefCOCOg**：真實影像的指稱語表達理解資料集，每個描述對應單一目標。RefCOCO 與 RefCOCO+ 提供官方 train / val / testA / testB；RefCOCOg 提供 val / test。RefCOCO+ 禁用空間／位置詞，因此關係殘差訊號（relation residual）在 RefCOCO+ 上一律為空，改以屬性殘差（attribute residual）為主。
- **gRefCOCO**：廣義 REC 資料集，每個描述可對應零（no-target）、一或多個物件，提供官方 train / val / testA / testB。無目標閘（C3）與完整 GREC 壓力測試（M4）、CRS 主結果均在此評測。

選擇 RefCOCO 系列與 gRefCOCO，是因為它們是真實影像（避開模擬 render 的偏差）、有官方標準切分、且 no-target／multi-target 設定能直接驗證棄答與集合化能力。

## 4.2 凍結基礎模型

我們以結構迥異的凍結基礎模型為實例，壓力測試信心訊號是否與基礎模型架構無關：

- **CLIP-VG**（單框迴歸）：以 CLIP 為骨幹，輸出恰一個框，**無候選清單、無分數**。這迫使信心只能來自**擾動訊號**（同義改寫下的預測變化），而非分數統計——此特性反而有利於跨基礎模型轉移（見第 10 章 C4）。
- **OWL-ViT**（開放詞彙偵測器）：單次前向傳遞輸出多個帶分數候選框，分數為免費副產品，支援**分數型信心**與**原生棄答**（最大分數低 ⇒ 傾向無目標）。
- **GroundingDINO**（query-based 偵測器，僅 CRS 章引入）：高召回的候選框產生器，在 CRS 中作為 box selector，與 OWL-ViT 的 gate 角色組合。

所有基礎模型參數**全程凍結**，never updated。我們只在其決策層附加輕量校準器（logistic regression 或 ≤2 層小 MLP，數十至數百參數）。

## 4.3 離線 dump 與粒度

實驗以離線 dump 基礎模型副產品為基礎，一次 dump 同時支撐 correctness AUROC、no-target AUROC、risk-coverage、cross-base transfer 與 CRS 校準。每一筆 row 是一個 **(sample, base_model, prompt_variant)** 三元組：

- `prompt_variant = canonical` 為主 row，承載 correctness／no-target／risk-coverage 主分析；
- 同義改寫（paraphrase）、判別性（discriminative）、中性（neutral）提示各為附 row，共享 `sample_uid`，用於還原跨提示一致性與殘差訊號；
- 跨基礎模型以 `sample_uid = {dataset}:{split}:{image_id}:{ref_id}` 為 join key。

每筆 row 記錄：識別與環境欄位、基礎模型輸出的候選框與分數、ground truth 與 correctness（`is_no_target`、`num_referents`、`gt_bboxes`、`pred_iou`、`correct@0.5`、`best_cand_iou`）、信心訊號（見 §4.5）、以及 protocol bookkeeping 的 `calib_fold`。dump 存為 JSONL，分區目錄 `dump/{dataset}/{base_model}/{split}.jsonl`，大張量外掛 `.npz`。

## 4.4 資料切分與校準協定

> **鐵律：任何閾值 τ、校準器參數、phenotype 門檻一律在 train/val 決定，test 只跑一次出最終數字。**

- **切分**：RefCOCO 系列用官方 train / val / testA / testB。`calib_train` = 官方 train（訓校準器）；`calib_val` = 官方 val（選所有 τ 與門檻、early stop、phenotype 校準）；`test` = testA / testB（最終報告，分開報）。RefCOCOg 用 val/test。gRefCOCO 用其官方 train/val/test，無目標閘的 τ_p 在 gRefCOCO val 選。dump 時即依官方 split 固定指派每 row 的 `calib_fold`（固定 seed），避免事後洩漏。

- **校準器訓練**：特徵為 dump 的全訊號；缺值（如 RefCOCO+ 的 relation_residual）以 mask + 指示位處理，不以 0 混淆。標準化統計（mean/std）只用 `calib_train`，套用到 val/test，**禁止用 test 統計**。多 seed（≥5）重訓，報平均 ± CI。

- **特徵分兩類**（跨基礎模型轉移的關鍵設計）：
  - **base-specific**（跨基礎模型不穩）：raw `top1_score`、raw `margin12`、raw `score_entropy`——絕對分數尺度因基礎模型而異。
  - **base-normalized／grounding-structural**（跨基礎模型可重用）：rank-normalized margin、identity stability entropy（提示擾動）、spatial dispersion、candidate-set entropy、cross-model agreement、candidate recall upper bound、關係／屬性殘差的 rank。
  - 跨基礎模型轉移主實驗**只用 base-normalized 特徵**；base-specific 特徵僅在 within-base 設定使用，並單獨報「加了它們 transfer 掉多少」做消融。

- **閾值選擇**：所有 τ 在 `calib_val` 上依目標準則選——選擇性預測固定 target risk r*，取滿足 risk≤r* 的最大 coverage 之 τ；或固定 target coverage，回報該點 risk。兩種準則皆先在 val 定 τ 再凍結套到 test，論文明確寫出準則與 r*／coverage 值。

## 4.5 信心訊號

所有訊號皆為事後（post-hoc），不更新基礎模型，依成本與可轉移性分兩族：

**(A) base-agnostic 擾動訊號**（成本 K× 前向，K 個改寫）：
- `cross_prompt_consistency`：K 個改寫提示下預測框的平均成對 IoU。一致性低 ⇒ 指稱身分在良性改寫下不穩 ⇒ 錯誤風險高。這是與定位相關的訊號，量測指稱穩定度而非泛用信心。
- `prompt_box_dispersion` / `spatial_dispersion`：改寫提示下預測框中心的離散度（除以 √area 正規化）。
- `identity_stability_entropy`：跨提示「哪個候選被選中」分布的熵。

**(B) base-specific 分數訊號**（成本 1× 前向，免費副產品；OWL-ViT）：
- `top1_score`、`margin12`（= top1 − top2）、`score_entropy`（softmax 候選分數的熵）、`score_mean_topk`。

**(C) 殘差訊號**（中成本）：
- `relation_residual` = ΔS(關係判別 − 中性)，僅適用 RefCOCO/RefCOCOg（RefCOCO+ 為空）；
- `attribute_residual` = ΔS(屬性判別 − 中性)，RefCOCO+ 主用。
- `cross_model_agreement`：與另一基礎模型 top-1 框的 IoU／一致性（cross-base join 後填，只用預測框、不用 GT）。

## 4.6 評測指標

- **Risk–Coverage 曲線**與 **AURC / E-AURC**（招牌圖：selective vs naive vs random vs oracle）。
- **AUROC**（correctness／no-target）。
- **ECE 與 reliability diagram**（校準器校準度）。
- **AUSE**（不確定性排序品質）。
- **Selective accuracy @ fixed coverage**、**Coverage @ target risk**。
- **GREC**：N-acc / T-acc / Pr@(F1=1, IoU≥0.5)，採官方忠實的 greedy-IoU 匹配。
- **CRS 三風險**：R1 已作答目標漏檢率（answered-target FNR，條件風險，不稱 recall guarantee）、R2 無目標誤選率、R3 目標棄答率；集合大小（set size）。
- **Oracle gap**：每張主圖附 oracle 線，量「訊號還差多少」。

## 4.7 統計顯著性與「保證」的界定

- **Bootstrap CI**：對 test 樣本 resample（≥1000 次）求 AURC／AUROC／accuracy／set size 的 95% 信賴區間；校準器與 config 固定。
- **多 seed**：校準器／改寫抽樣的隨機性以 ≥5 seed 平均，報 seed 間 std。
- **保證 vs CI（CRS 專用，必須分清）**：CRS 的有限樣本風險控制保證來自 **LTT p-value + Bonferroni** 檢定；固定 config 上的 bootstrap CI 只是**經驗穩定度**，**不是**保證。兩者分開報告，never conflated。CRS 的閾值網格只建構於**校準 covariate**（無任何 evaluation covariate 或 label 進入網格建構、風險檢定或操作點選擇）。

## 4.8 防洩漏 checklist（隨論文附錄）

- [ ] 所有 τ 來自 val，test 單次評估。
- [ ] 標準化統計來自 calib_train。
- [ ] paraphrase／discriminative 提示模板在看 test 前凍結。
- [ ] phenotype 門檻在 val 校準。
- [ ] cross-base join 不洩漏 GT（cross_model_agreement 只用預測框 IoU）。
- [ ] 跨基礎模型轉移的校準器只用 base-normalized 特徵；target base 的 test 統計不回流到 source base 訓練。
- [ ] CRS 閾值網格只用校準 covariate。

## 4.9 計算環境

所有 dump 與校準在單機 GPU（NVIDIA GB10）上完成，PyTorch 2.12 / CUDA 13.0。推論成本以「每查詢平均前向傳遞次數」加實測 throughput（query/s）量化，不以形容詞描述（詳見第 11 章成本—風險 Pareto）。
