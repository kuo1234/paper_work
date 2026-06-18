# InstanceVG 深讀筆記與 CRS 定位分析

## 一、書目資訊

- **標題**：Improving Generalized Visual Grounding with Instance-aware Joint Learning
  （系統/開源庫名 **InstanceVG**；論文標題本身不含 "InstanceVG:" 前綴，標題即上述句子）
- **作者**：Ming Dai（戴明，東南大學自動化學院第一作者）、Wenxuan Cheng、Jiang-Jiang Liu（百度）、Lingfeng Yang（南京理工）、Zhenhua Feng（江南大學）、Wankou Yang（通訊，東南大學）、Jingdong Wang（百度，通訊群）。
- **出處**：arXiv 2509.13747（v1），標示投往 TPAMI（IEEE Trans. PAMI），2025。
- **程式碼**：https://github.com/Dmmm1997/InstanceVG （宣稱將開源 code、model 與重處理後的資料集）。
- **關鍵字**：Visual Grounding、Multimodal Transformer、Instance Awareness、Multi-Task Learning。

> 註：作者群同時是 SimVG（NeurIPS'24）、PropVG、DeRIS 的作者，InstanceVG 是這條「BEiT-3 fusion encoder 系列」的延伸與集大成之作。

---

## 二、問題定義與動機

### 2.1 任務背景
- **經典 visual grounding**：REC（Referring Expression Comprehension，輸出單一 bounding box）與 RES（Referring Expression Segmentation，輸出像素級 mask），核心假設是「一句話對一個目標」（one-to-one）。
- **Generalized visual grounding（廣義）**：將經典範式擴展到 **multi-target（多目標）** 與 **non-target（無目標）** 兩種情境：
  - **GREC**（Generalized REC）：粗粒度，輸出所有被指涉物件的 bbox。
  - **GRES**（Generalized RES）：細粒度，輸出像素級 mask。
- 應用場景：embodied AI、自駕。

### 2.2 論文點出的三個缺口（motivation）
1. **GREC 與 GRES 被獨立處理**：現有方法分別訓練偵測與分割兩個模型，忽略了聯合訓練帶來的「多粒度一致性」與流程精簡。
2. **GRES 被當成語意分割**：現有 GRES（如 ReLA、CoHD）把所有實例 mask 合併成「單一全域 mask」監督，**喪失了 instance-aware（實例感知）能力**——多目標時無法區分、定位個別實體。
3. **缺乏 box–mask 一致性機制**：沒有顯式機制保證「同一個實例的預測框與遮罩對應一致」。

### 2.3 兩個核心研究問題
1. 如何在廣義情境下有效進行 **multi-task 聯合訓練**，精簡流程同時得到互補且實例一致的預測？
2. 如何設計一個賦予廣義 grounding **instance-aware 能力**的原則性框架，用細粒度監督強化指涉理解？

---

## 三、方法詳解

### 3.1 整體架構（Fig. 2）
InstanceVG = **共享 multi-modality encoder + 兩條分支**。

- **Multi-Modality Encoder = BEiT-3**（ViT-B / ViT-L）。影像切 patch、文字 tokenize，兩模態特徵串接後丟進 BEiT-3 聯合編碼（處理細節沿用作者自家的 **SimVG**）。輸出再經影像/文字線性層投影到低維 C，得 `F_i`（影像）與 `F_t`（文字）。
- **分支 (c) Global Segmentation Branch（全域分割分支）**：用 **SimFPN** 把 ViT 單層輸出擴成多尺度，接一個簡單 **U-Net decoder** 產生全域語意分割 `S_global`（解析度 H/4 × W/4）。此分支身兼四職：(1) 全域語意分割預測；(2) **判斷是否存在目標（non-target 判別）**；(3) 提供多尺度影像特徵給 APD；(4) 與解碼後 query `Q_d` 互動產生實例語意 query `Q_s`。
- **分支 (a)+(b) Instance Perception Branch（實例感知分支）**：預測 instance box 與 instance mask，由 APD 與 PIPH 兩模組構成。

### 3.2 APD — Attention-based Point-prior Decoder（注意力點先驗解碼器）
目的：自適應地產生「先驗參考點」並注入 instance query。

**(1) IQG — Instance Query Generator（Fig. 3）**
- **Text Filter（Algorithm 1）**：從 N_t 個文字 token 中，先去掉 padding，再用 **L2 norm 分數** 取 top-N_q 個高響應 token 作為 `F_filter`（不足則 pad）。預設 **N_q = 10**（因為指涉任務目標數很少超過 10）。
- 以 `F_filter` 為 Q、影像特徵 `F_i` 為 K/V 做 cross-attention 得 attention map `M_attn`，沿 query 維平均得空間分數圖 `M_s`。
- **Dynamic Point Selector（Algorithm 2，貪婪演算法）**：在 `M_s` 上 sigmoid 後，先取最高分點，之後每步以 `S = M_s + W_dist × D`（D 為與已選點的最小距離，W_dist=0.003）選點，**兼顧高響應與彼此分散**，確保覆蓋所有潛在實例。產出 N_q 個先驗點 `P_r` 與對應 query。
- 作者自訂 **CoverAcc = (1/N)Σ TP/(TP+FN)** 衡量點對目標的覆蓋率（Fig. 4），證明 dynamic selector 比 TopK 更能覆蓋實例。
- 把選到的 query 與 `F_q` 串接過 MLP 得 instance query `Q_instance`。

**(2) Point-prior Multi-scale Deformable Decoder**
- 基於 Deformable DETR。關鍵改動：用 IQG 產生的先驗點 `p_r` 取代 grid 上預設的密集參考點，object query 從隨機 `z_q` 換成 `Q_instance`。
- 好處：query 數從數百降到 ~10（降算力）＋ 每個 query 拿到目標中心附近的強位置先驗（提效能）。輸出解碼後 query `Q_d`。

### 3.3 PIPH — Point-guided Instance-aware Perception Head（點導引實例感知頭）
目的：顯式建立 **點 → 框 → 遮罩** 的對應與一致性。

- **實例語意 query**：`Q_s = Q_d (elementwise) S_global`（解碼 query 與全域語意特徵點積），讓 query 抽取全域語意以導引 instance-level 分割。
- **(1) Point-guided Object Matcher（點導引匹配，§3.3.1）**：在 DETR 的匈牙利匹配 cost 上**新增點 cost**：
  - `C_ij = λ_cls·CE + λ_box·L1(box) + λ_giou·GIoU(box) + λ_point·L1(point, center)`
  - λ_cls=1.0, λ_box=5.0, λ_giou=2.0（沿用 DETR），新引入 **λ_point=2.0**。讓參考點盡量靠近目標中心，強化先驗導引。
- **(2) Query-Mask Aligner（§3.3.2）**：利用 box↔mask 的一對一對應，把「query→box」的匹配關係 `M_q2b` 傳播成「query→mask」`M_q2m`，靠 (i) 資料層 gt box 與 gt mask 一對一、(ii) 預測層 `Q_d` 與 `Q_s` 順序對齊 兩個前提，保證點/框/遮罩一致。

### 3.4 訓練目標（§3.4）
四部分損失加權：
`L_total = λ_detr·L_detr + λ_seg·L_seg + λ_instance·L_ins-seg + λ_exist·L_exist`
- **L_detr**：偵測（L1 + CE + GIoU，DETR 式）。
- **L_seg**：全域語意分割（BCE + Dice）。
- **L_ins-seg**：實例級分割（BCE + Dice），對正樣本算 loss，並用 **λ_neg=0.2** 加權負樣本 mask 抑制：`L_ins-seg = (1/N_pos)ΣL_pos + (λ_neg/N_neg)ΣL_neg`。
- **L_exist**：**non-target 判別 = 二元分類 BCE loss**，判斷指涉物是否存在。
- 權重：λ_detr=0.1, λ_seg=1.0, λ_instance=1.0, λ_exist=0.2。

### 3.5 後處理（§3.5，Fig. 5）— 與本論文 CRS 最相關的一節
1. **query 分數 × non-target 分數**（dot product）合併，降低無目標場景的 false positive。
2. 用閾值 **thr_q** 篩出有效 query（index）→ 偵測分支據此輸出目標。
3. 分割分支：全域 mask 與 instance mask 用閾值 **thr_m** 二值化後，依 index 篩選並做 **logical OR** 合併，補全不完整實例。

> **關鍵觀察（給 CRS）**：N-acc（無目標正確率）來自 (a) 一個訓練好的二元 **L_exist BCE 分支** 的 point estimate，外加 (b) 後處理中 non-target 分數與 thr_q 的**手動調參融合**。**完全沒有任何 calibration / coverage（覆蓋率）/ distribution-free 保證**。thr_q 是在 val 集上掃出來的（見 Table XIII：thr_q 從 0.7→0.9，F1 由 71.43 升到 74.38），屬於經驗性閾值選擇，不是統計保證的棄答機制。

---

## 四、完整實驗數字

設定：雙 RTX 4090；SOTA 實驗輸入 320×320；GREC/GRES 訓 10 epoch；REC/RES 訓 20 epoch。

### 4.1 GREC（gRefCOCO，Table VI，**所有方法 score threshold = 0.7**）
指標：Pr@(F1=1, IoU≥0.5)（簡稱 F1score）與 N-acc.

| Method | val F1 | val N-acc | testA F1 | testA N-acc | testB F1 | testB N-acc |
|---|---|---|---|---|---|---|
| MCN | 28.0 | 30.6 | 32.3 | 32.0 | 26.8 | 30.3 |
| VLT | 36.6 | 35.2 | 40.2 | 34.1 | 30.2 | 32.5 |
| MDETR | 42.7 | 36.3 | 50.0 | 34.5 | 36.5 | 31.0 |
| UNINEXT | 58.2 | 50.6 | 46.4 | 49.3 | 42.9 | 48.2 |
| SimVG | 62.1 | 54.7 | 64.6 | 57.2 | 54.8 | 57.2 |
| PropVG | 72.2 | 72.8 | 68.8 | 69.9 | 59.0 | 65.0 |
| **InstanceVG** | **73.5** | **72.8** | **70.2** | **71.1** | **60.8** | **65.2** |

**這就是使用者記憶中「gRefCOCO GREC val F1 73.5 / N-acc 72.8」的來源，已逐項核實無誤。** InstanceVG 比前一代 SimVG 在 val/testA/testB F1 各 +11.4 / +5.6 / +6.0。

### 4.2 GRES（gRefCOCO，Table III，ViT-B）
指標：gIoU / cIoU / N-acc

| Method | Backbone | val gIoU | val cIoU | val N-acc | testA gIoU | testA cIoU | testA N-acc | testB gIoU | testB cIoU | testB N-acc |
|---|---|---|---|---|---|---|---|---|---|---|
| ReLA | Swin-B | 63.60 | 62.42 | 56.37 | 70.03 | 69.26 | 59.02 | 61.02 | 59.88 | 58.40 |
| CoHD | Swin-B | 68.42 | 65.17 | 63.68 | 72.67 | 71.85 | 64.00 | 63.60 | 62.63 | 60.37 |
| PropVG | BEiT3-B | 73.29 | 69.23 | 72.83 | 74.43 | 74.20 | 69.87 | 65.87 | 64.76 | 64.97 |
| DeRIS | Swin-S+BEiT3-B | 74.10 | 68.06 | **77.03** | 73.72 | 71.99 | **75.98** | 65.63 | 64.65 | 63.44 |
| **InstanceVG** | BEiT3-B | **73.36** | **69.22** | 72.84 | **75.21** | **74.51** | 71.09 | **66.74** | **65.67** | **65.18** |

對 CoHD：val/testA/testB gIoU 各 +4.9 / +2.5 / +3.1。注意 **N-acc 在 GRES 上 InstanceVG（72.84）其實略輸 DeRIS（77.03）**，論文宣稱「all metrics SOTA」需小心解讀——它的優勢主要在 gIoU/cIoU。

### 4.3 其他任務（佐證「10 資料集 4 任務 SOTA」主張）
- **REC（RefCOCO/+/g，Table I，Prec@0.5）**：ViT-L 版 val/testA/testB = RefCOCO 94.42/96.04/92.39、RefCOCO+ 90.12/92.89/85.94、RefCOCOg 89.58/90.62（**zero fine-tune，僅 28K pre-train**），勝 OneRef-L、SimVG-L。
- **RES（RefCOCO/+/g，Table II，mIoU）**：ViT-L 版 RefCOCO 86.27/87.12/85.30 等，勝 OneRef-L、DeRIS-L。
- **Ref-ZOM（Table IV）**：oIoU 71.52 / mIoU 71.12 / Acc 97.42，勝 CoHD、GSVA-7B。
- **R-RefCOCO/+/g（Table V）**：rIoU 62.41/59.13/54.36，與 PropVG 互有勝負（rIoU 對 CoHD +8.8/+10.0/+12.2）。

### 4.4 重要 ablation（解釋方法有效性）
- **核心模組（Table VII，224×224）**：baseline 65.98 → +Multi-Task 67.13 → +APD 69.17 → +PIPH **71.43** F1（gIoU 65.03→72.41）。
- **不同 backbone（Table XVII）**：CLIP-ViT-B、ViLT-B、BEiT3-B 加上 APD+PIPH 模組都一致提升（如 BEiT3-B 67.13→71.43），證明模組是 backbone-agnostic。
- **後處理閾值 thr_q（Table XIII）**：0.70→0.90，F1 71.43→74.38（**val 集調出來的，非保證**）。
- **參數量（Table XVI）**：APD+PIPH 僅 ~3M（佔總參數 1.6%），算力佔 7.6%，總模型 ~182.7M params。

### 4.5 論文自承的限制（Conclusion）
1. **目標存在性判別準確度不足**（即 N-acc 還有很大空間）。
2. 小物件感知能力有限（受輸入解析度與早期 16× 降採樣所限）。
3. 細粒度分割不足（ViT patch embedding 壓縮造成資訊損失）。

---

## 五、與 CRS 的逐項 diff 表

| 面向 | **InstanceVG**（本論文，現任 GREC SOTA） | **CRS**（使用者碩論：Cross-Base Conformal Composition） |
|---|---|---|
| **訓練範式** | Trained end-to-end，整個 BEiT-3 + APD + PIPH 一起 fine-tune（10 epoch，雙 4090） | **Frozen post-hoc**：OWL-ViT gate + GroundingDINO box 全程凍結，零重訓 |
| **base 數量** | 單一統一模型（single model） | **Cross-base composition**：兩個異質 frozen base 組合（OWL gate 控棄答、GD box 控緊緻集合） |
| **輸出形式** | 點估計：一組 box/mask + 二元 existence 旗標 | **Conformal referring SET**（符合保證的指涉集合） |
| **No-target / 棄答機制** | 訓練好的 BCE existence 分支（L_exist）+ 後處理 thr_q 手動掃出的閾值融合 | **LTT（Learn-then-Test）校準**的 abstention，具 distribution-free 棄答保證 |
| **N-acc 性質** | **Point estimate**，無覆蓋保證；論文自承「準確度不足」 | 不只報 N-acc，而是給 **同時的 recall + abstention 雙保證**（R1/R2 CI 上界 < 0.3） |
| **統計保證** | **無任何 calibration / coverage / CI 保證**；thr_q 在 val 上經驗選擇 | **LTT 聯合校準 + Bonferroni 多風險控制**，size CI 與 baseline 分離 |
| **可靠性視角** | 追求絕對分數（F1/gIoU/cIoU 最大化） | 追求 **可靠性 / 風險可控**（在保證下輸出最小集合） |
| **計算成本** | 需 GPU 重訓全模型 | 僅校準階段（grid search + LTT），base 推論可離線快取 |
| **指標** | Pr@(F1=1, IoU≥0.5)、N-acc、gIoU/cIoU | 同樣用 GREC 指標 Pr@(F1=1, IoU≥0.5)/N-acc/T-acc，外加 set size、coverage CI |

**一句話結論**：InstanceVG 與 CRS 在**同一個 benchmark（gRefCOCO + GREC metrics）**競技，但屬於**兩條完全不同的賽道**——前者是「訓練式單模型把絕對分數推到最高」，後者是「凍結式跨 base 組合，在分布無關保證下輸出符合集合」。**兩者不應在絕對分數上正面對撞。**

---

## 六、論文該怎麼定位（給 CRS 寫作的具體建議）

### 6.1 核心定位語：「不比絕對分數，比可靠性」
- 在 related work / SOTA 對照表中，**明確把 InstanceVG 標為「method-layer ceiling（方法層天花板）」**，並用一句話框定：CRS 是 **reliability/calibration study on frozen bases**，與 InstanceVG 的 trained instance-aware end-to-end model 是 orthogonal（正交）的兩條 track。
- CRS 的賣點不是「F1 比 73.5 高」，而是「**在 distribution-free 保證下**輸出一個可信賴的指涉集合」——這正好補上 InstanceVG 自承的限制（「target existence determination 不足、無保證」）。

### 6.2 SOTA 對照表怎麼擺
建議用**雙欄/雙區塊**呈現，避免誤導性的同欄直接比分數：
1. **「Trained / method-layer」區塊**：列 InstanceVG（73.5/72.8）、PropVG、SimVG、DeRIS——標註 *requires full retraining, no guarantees*。
2. **「Frozen post-hoc / reliability」區塊**：列 CRS（含 OWL-only、GD-only baseline 與 COMPOSE）——標註 *zero retraining, LTT recall+abstention guarantee, set output*。
3. 額外加一欄 **「Guarantee?」**：InstanceVG = 無、CRS = 有（recall + abstention，distribution-free）。這一欄是 CRS 最強的差異化。
- 誠實揭露：CRS 全量 val/testA/testB 在 α=β=0.3 下輸出 3.21/2.02/3.48 框，Pr@F1=1 約 0.26–0.61（記憶中 M4 數字），**會明顯低於 InstanceVG 的 73.5**。要在表格 caption 與正文把這點講清楚是「保證 vs 點估計」的本質取捨，不是 CRS 弱。

### 6.3 要不要把 InstanceVG 當成 CRS 的 box base 之一？（可行性評估）
**短期：不建議納入主結果；長期：可作為 future work / robustness 附錄。**
- **可行性正面**：InstanceVG 會開源 code + model（GitHub Dmmm1997/InstanceVG）；它本身就輸出多框（GREC），且其 query score 可當作 CRS 校準所需的 nonconformity 分數來源——架構上可插入 CRS 的 box-base 槽。
- **可行性風險與成本**：
  1. **凍結純度爭議**：CRS 的賣點是「frozen base + 不重訓」。InstanceVG 已在 gRefCOCO 上 end-to-end 訓練過，把它當 base 會讓「凍結 + 跨 base transfer」的故事變模糊（它不是 RefCOCO 泛用 grounding base，而是 in-domain trained model）——這會稀釋 C4（cross-base transfer）的論述。
  2. **依賴未發布權重**：截稿前模型是否真的開源、推論環境（雙 4090 / GB10 相容性）都是風險，呼應記憶中 BTS 在 GB10 上遇過的 detector 數值問題。
  3. **邊際效益**：CRS 已用 OWL-ViT + GroundingDINO 證成 factorization（OWL gate + GD box），再加一個 trained SOTA base 主要是「展示框架可吃更強 base」，屬 nice-to-have 而非核心。
- **建議做法**：把 InstanceVG 當成 **(a) SOTA 對照表的方法層天花板**（必做）、**(b) 一個 future-work 的「更強 box base 候選」**（提一句即可）。若時間允許，可做一個小型 robustness 附錄：「若以 InstanceVG 的輸出框替換 GD box base，CRS 的保證是否仍成立 / set size 是否縮小」——這能直接展示「CRS 是 base-agnostic 的可靠性外殼」，把 SOTA 方法反過來「包」進 CRS，論述上非常漂亮（CRS 不與 SOTA 競爭，而是讓任何 SOTA 都能戴上保證）。

---

## 七、個人評價

### 優點
- **工程整合度高**：第一個把 GREC + GRES 在單一 query-based 架構聯合訓練、且引入 instance-aware 的框架；APD（點先驗）+ PIPH（點-框-遮罩一致性）設計巧妙，僅 +3M 參數換來跨 4 任務 10 資料集的全面 SOTA。
- **point-prior 取代 dense grid query** 是務實洞見（指涉目標 ≤10），同時降算力又給強位置先驗。
- backbone-agnostic 的 ablation（Table XVII）做得紮實，可信度高。
- 數據覆蓋全面、消融細緻（光後處理就拆了 thr_q、mask merge、NT score、NMS 四張表）。

### 缺點 / 它沒做的
1. **完全沒有 calibration / abstention guarantee**：N-acc 與 existence 判別純靠一個 BCE 分支的點估計 + val 上手動掃的 thr_q。論文自己在 Conclusion 第一條就承認「target existence determination 準確度不足」——這正是 **CRS 的整個切入點**。
2. **thr_q 在 val 集上選**：0.7→0.9 把 F1 從 71.43 推到 74.38，有 val-set overfitting 的味道，且沒有任何在 test 上的覆蓋率保證或信賴區間。
3. **「all metrics SOTA」略誇**：GRES 的 N-acc（72.84）實際輸給 DeRIS（77.03），論文行文未充分強調這點。
4. **小物件 / 細粒度分割受限**（自承），且 GREC 用固定 threshold=0.7 比較，未探討不同 operating point 下的 precision-recall / cost trade-off——而這恰好是 CRS 用 Pareto 與 LTT 風險曲線能補的。
5. **可靠性盲區**：它給「一個答案」，但不告訴你「這個答案多可信、何時該棄答」。在 embodied AI / 自駕（它自己舉的應用）這種 safety-critical 場景，沒有棄答保證是真實短板——CRS 的 distribution-free recall+abstention 雙保證正是針對這個缺口。

### 對 CRS 的最終 take-away
InstanceVG 是 CRS 必須「站在其下方並標清楚賽道」的方法層天花板，但它的存在**強化而非威脅** CRS 的故事：一個訓練到極致、且作者自承「存在性判別不足、無保證」的 SOTA，正好證明「絕對分數的競賽已接近飽和，真正缺的是可靠性層」。CRS 應把自己定位成「**任何 grounding base（含 InstanceVG 這種 SOTA）都能戴上的 distribution-free 保證外殼**」，而非又一個刷分方法。
