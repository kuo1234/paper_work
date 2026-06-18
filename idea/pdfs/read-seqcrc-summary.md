# 論文精讀：Conformal Object Detection by Sequential Risk Control (SeqCRC)

> 本文是針對 SeqCRC 的深讀筆記，重點放在「方法如何運作」與「它與本論文主線 CRS (Cross-Base Conformal Composition) 的逐項差異」。SeqCRC 是 CRS 在結構上最近的鄰居（同樣是 conformal OD、同樣是兩個參數依序校準），審稿人極可能拿它來質疑 CRS 的新穎性，因此本筆記特別強化「如何切割、如何引用」這一段。

---

## 1. 書目

| 欄位 | 內容 |
|---|---|
| 標題 | Conformal Object Detection by Sequential Risk Control |
| 作者 | Léo Andéol、Luca Mossina、Adrien Mazoyer、Sébastien Gerchinovitz |
| 單位 | Univ Toulouse (Institut de Mathématiques de Toulouse)、SNCF、IRT Saint Exupéry、DEEL / ANITI |
| 形式 | 期刊投稿稿（footnote 註明 "submitted to the IEEE for possible publication"），v2 |
| 年份 | 2025 |
| arXiv | 2505.24038 (v2) |
| 程式碼 | https://github.com/leoandeol/cods （COD Toolkit，PyTorch，CPU/GPU） |

這是一個明確走「安全關鍵工業認證」路線的團隊（鐵路號誌、跑道偵測、航太 DEEL/IRT Saint Exupéry 背景），其前作鏈是 SAFECOMP 2022 [de Grancey et al.]、COPA 2023 鐵路號誌、AI & Ethics 2024，本文是這條線的「集大成統一框架」。

---

## 2. 問題定義與動機

### 2.1 動機
物件偵測 (OD) 已被工業採用，但要部署到 **安全關鍵系統**（自駕、醫療、航太認證）就需要對預測「給統計保證」。神經網路本身不可靠、OD 模型結構又複雜（一張圖未知數量的物件 × 同時要定位+分類），因此作者轉向 **Conformal Prediction (CP)**：post-hoc、分布無關 (distribution-free)、有限樣本有效 (finite-sample valid)、與模型/資料分布無關。

### 2.2 既有工作的缺口（作者自己定位的 gap）
作者主張：現有 conformal OD 文獻 **各只處理 OD pipeline 的一個子步驟或單一模型家族**：
- 多數只做 **localization**（給 bounding box 加 margin），如 de Grancey 2022、Andéol 2023/2024、Copula-based [Mukama]。
- 兩個例外：Timans et al. (ECCV 2024) 處理 classification 且做 class-conditional 的 localization 修正；Li et al. (PAC multi-object) 針對 Faster R-CNN 建「全包式」集合但**綁死特定模型**。
- 沒有人提出一個 **holistic 框架**：對「幾乎所有 OD 模型家族」都成立、且同時對「預測物件數量 + 定位 + 分類正確率」給統計保證。

### 2.3 本文要解決的核心技術障礙
OD 的 conformal 化天然需要 **多個依序相依的參數**，但經典 CRC (Conformal Risk Control) 只能處理**單一純量參數** λ。這就是本文的技術靶心。

---

## 3. 方法詳解：Sequential Conformal Risk Control (SeqCRC)

### 3.1 三個參數的語意（這是理解全文的鑰匙）
SeqCRC 對 OD 的後處理拆成 **「1 + 2」三個純量參數**：

1. **λ^cnf（confidence，信心門檻）** — 定義一個門檻 `o(x) ≥ 1 − λ^cnf`，低於門檻的預測框被丟掉。控制「保留多少框」。
2. **λ^loc（localization，定位 margin）** — 對保留下來的每個框，加上 margin（加法式 Eq.12，或乘以框寬高的乘法式 Eq.13）把框「撐大」，使其能涵蓋 ground-truth。
3. **λ^cls（classification，分類集合）** — 對每個框輸出一個「類別集合」（LAC 門檻式 Eq.15，或 APS 累積機率式 Eq.16），保證以高機率含真類別。

關鍵相依結構：**loc 與 cls 都依賴 cnf 的結果**（因為先 threshold 才有框可定位/分類），但 loc 與 cls 彼此**平行獨立**。所以是「先 1（cnf），後 2（loc、cls 並行）」的 **sequential** 結構 → 名稱由此而來。

### 3.2 背景：經典 CRC（單參數）
給校準集 D_cal = (X_i,Y_i)，預測集 Γ_λ(X)，損失 ℓ(Y,Γ_λ(X))（要求對 λ **非遞增、右連續、值域 [0,B]**，且在 λ̄=maxΛ 時損失為 0）。CRC 選：

```
λ̂ = inf{ λ : (1/(n+1))·Σ L_i(λ) + B/(n+1) ≤ α }
```

保證 (Theorem 1)：`α − 2B/(n+1) ≤ E[L_test(λ̂)] ≤ α`。注意這是 **對期望值（marginal over 校準+測試集）** 的保證，不是 per-instance、也不是 PAC 式高機率保證。

### 3.3 SeqCRC 的核心機制（本文真正的新貢獻）
直接把 CRC 串兩次會出問題：第二步的 λ^loc/λ^cls 是用「第一步選出的 λ^cnf」算出來的，這個相依性破壞了 exchangeability，使 finite-sample 保證失效。SeqCRC 的解法有兩個技巧：

**技巧 A — 一個 confidence step 產生「兩個」估計量（不是一個）**

```
λ^cnf_+ = inf{ λ : n·R̃_n^cnf(λ)/(n+1) + B̃^cnf/(n+1) ≤ α^cnf }   (Eq.4，保守 / 加 B 項)
λ^cnf_- = inf{ λ : n·R̃_n^cnf(λ)/(n+1) + 0     ≤ α^cnf }   (Eq.5，樂觀 / 不加 B 項)
```

- `λ^cnf_+`（保守估計量）→ **實際用在測試影像** 做 thresholding。
- `λ^cnf_-`（樂觀估計量）→ **只在第二步 Eq.6 內部使用**，作為對稱性 (symmetry) 證明的關鍵。這是讓有限樣本保證成立的核心數學手法。

**技巧 B — 把第二步「錨」在樂觀估計量 λ^cnf_- 上**

```
λ^∙_+ = inf{ λ^∙ : n·R_n^∙(λ^cnf_-, λ^∙)/(n+1) + B^∙/(n+1) ≤ α^∙ },  ∙ in {loc, cls}   (Eq.6)
```

**技巧 C — 保守化的 confidence 風險 R̃^cnf（Eq.3）**：第一步的風險不只看 cnf loss，而是取三者最大值
`R̃_n^cnf(λ) = max{ R_n^cnf(λ), R_n^loc(λ, λ̄^loc), R_n^cls(λ, λ̄^cls) }`
目的是**確保第二步 Eq.6 的解集合非空**（feasibility）。

### 3.4 保證形式（Theorem 2 + Corollary 1）
在 Assumption 1 (i.i.d. + 確定性預測器) 與 Assumption 3 (兩步版的單調/右連續/值域有界/在 λ̄ 損失為 0) 下，若 `α^∙ ≥ α^cnf + B^∙/(n+1)`：

- **定位 / 分類保證 (Eq.7)**：`E[ L_test^∙(λ^cnf_+, λ^∙_+) ] ≤ α^∙`
- **信心保證 (Eq.8，需額外假設)**：`E[ L_test^cnf(λ^cnf_+) ] ≤ α^cnf`
- **聯合保證 (Corollary 1, Eq.9)**：把 α 在任務間「分帳」
  `E[ max( L_test^loc, L_test^cls ) ] ≤ α^tot`，其中 `α^tot = α^loc + α^cls`（用 max{a,b} ≤ a+b 推得；若要把 cnf 也納入則 α^tot = α^cnf+α^loc+α^cls）。

**保證性質要點**：
- 是 **期望值（marginal/平均）保證**，非 PAC 式 (1−δ) 高機率保證、非 per-instance。
- 重要差異化點（作者自己強調）：與並行獨立工作 Xu et al. 2024 (Two-stage CRC for ranked retrieval, arXiv 2404.17769) 相比，**SeqCRC 只需「單一資料切分 (single data split)」就能拿到有限樣本保證**；Xu 等人的方法要嘛是 asymptotic、要嘛要「兩次資料切分」。這是 SeqCRC 數學上最硬的賣點。
- 與 LTT (Learn-Then-Test, Angelopoulos et al.) 對比：LTT 用 multiple testing 處理多參數，本文則用「樂觀/保守雙估計量 + 對稱性」走 CRC 路線（不是 LTT 路線）。

### 3.5 OD 特有的建模細節（Section IV）
這些是讓抽象 SeqCRC 在 OD 上「真正能跑」的工程選擇：

- **保證層級 (level of guarantee)**：object-level（每個標註物件為一個 instance）vs image-level（每張圖為一個 instance）。本文核心採 **image-level**，理由是更彈性、可表達物件間互動。
- **Matching（預測框 ↔ ground-truth 配對）**：用「距離」d 把每個真框配到最近預測框（可非單射；也可用 Hungarian 做單射）。距離選擇：
  - `d_haus`：非對稱有號 Hausdorff（要把預測框撐多大才能完全蓋住真框的最小 margin）— 定位用。
  - `d_LAC`：1 − 真類別的 softmax 分數 — 分類用。
  - `d_mix = τ·d_LAC + (1−τ)·d_haus`（Eq.10，實驗 τ=0.25）— 綜合，整體表現最好。
  - `d_GIoU`：Generalized IoU — 雖然最「OD 直覺」，但**實驗中全面表現最差**（因為與後續 loss 脫節）。
- **損失函數（新舊都有）**：
  - Confidence：`box_count_threshold`（嚴格：框數 < 真框數就罰 1）、`box_count_recall`（鬆弛，缺框比例）。
  - Localization：`L_thr`（門檻式覆蓋率 ≥ τ）、`L_box`（box-wise recall，Eq.14）、`L_pix`（pixel-wise，按面積覆蓋比例，Eq.最鬆、框最小）。
  - Classification：對每個真物件檢查類別是否落在預測集合內，取平均（Eq.17）。
- **單調化技巧 (monotonization trick)**：loc/cls loss 對 λ^cnf **不必然單調**（因為 λ^cnf 變大→框變多→matching 改變），破壞 Assumption 3。作者用 `sup_{λ'≥λ^cnf} L(λ',·)` 的最小單調上界即時取代原 loss（沿用 Angelopoulos CRC 的做法），保住保證但代價是更大的集合。
- 兩個演算法：Algorithm 1（校準，含 on-the-fly 單調化）、Algorithm 2（推論，只用 λ_+ 值，計算成本相對一次前向傳播可忽略）。

---

## 4. 實驗

### 4.1 設定
- **資料集**：MS-COCO，用 validation split，再對切成 calibration / inference 各 **n = 2500**。
- **模型**：YOLOv8x (68M，單階段) 與 DETR-101 (60M，transformer)，強調 **model-agnostic**。
- **風險層級**：α^tot = 0.1（α^cnf=0.02, α^loc=0.05, α^cls=0.05）與 α^tot = 0.2。NMS IoU 門檻 0.5。額外過濾 conf < 1e-3（資料無關，不破壞保證）以減緩單調化造成的爆框。
- **指標**：依 CP 慣例報 **Empirical Risk**（= 1 − coverage，對 binary loss）與 **Set Size**（三種：confidence=平均框數；localization=conformal 框面積/原框面積開根號的 stretch；classification=平均類別集合大小）。**沒有用 GREC 那套 Pr@(F1=1)/N-acc/T-acc**。

### 4.2 主要結果
- **風險全面被控制**：幾乎所有設定下 empirical risk ≤ 目標 α（少數略超出歸因於抽樣隨機性），實證驗證 Theorem 2，且對 DETR/YOLO 都成立 → 印證 model-agnosticism。
- **Table I（DETR, α^tot=0.1）關鍵數字**：pixelwise loc loss 給最小 stretch（1.043），box_count_recall 比 threshold 框數少（17.8 vs 25.6）。鬆弛 loss → 集合更小但保證更弱。
- **Table II（matching × α）**：Hausdorff 最小化定位 size、LAC 最小化分類 size、Mix 給最佳折衷；**GIoU 全面爆掉**（loc size 28、cls size 44）。α 對 set size 的影響**不平滑**，小幅改 α 可能 size 大跳。
- **Table III（DETR vs YOLO, α^tot=0.1）**：兩模型風險都受控，但 set size 取決於底層模型品質；YOLOv8 傾向「選較少框但定位修正更大」（同一組超參，trade-off 不同）。
- **heavy tail 警告 (Fig.7)**：跨所有 loss×set×matching 組合，set size 分布是重尾 —— 很多組合會產生「整張圖那麼大的框」或「幾十個類別的集合」，**完全不可用**。亦即 loss/組合選錯會毀掉實用性。
- **失敗案例 (Fig.8, 兩隻斑馬)**：除了兩隻斑馬被正確偵測，還多出一個把兩隻都框住的大框（無 ground-truth 對應）。作者點出兩個根本問題：
  1. **本方法只保證 recall 夠高，只能「經驗性」限制 false positive**（precision 是其參數的非單調函數，硬單調化會讓風險過大而無解）。
  2. MS-COCO 標註本身不一致（個別物件旁又有一個大包圍框、漏標），模型學了就會不可預測地產生 group box。

### 4.3 Toolkit
開源 COD Toolkit（PyTorch，內含 YOLO/DETR、多種 loss、視覺化），定位為「Conformal OD 的首個大規模 benchmark + 可重現基礎設施」（號稱數百次 run）。

---

## 5. 與 CRS (本論文主線) 的逐項 Diff 表

> CRS = Cross-Base Conformal Composition：**OWL-ViT 當 abstention / no-target GATE + GroundingDINO 出 BOX**，用 LTT 聯合校準，對 **recall 與 abstention（棄答/no-target）** 同時給保證，Bonferroni over grid 處理多風險；在 gRefCOCO/RefCOCO 上輸出 conformal **referring set**，以 GREC 指標 (Pr@(F1=1,IoU≥0.5)、N-acc、T-acc) 評估；frozen base、post-hoc、不重訓。

### 5.1 相同點（這些一定會被審稿人抓來說「很像」）
| 面向 | 兩者皆是 |
|---|---|
| 範式 | post-hoc、frozen 模型、不重訓、distribution-free、有限樣本保證 |
| 家族 | 都屬 Conformal Risk Control / risk-control 系（CRC ↔ LTT 同源思想） |
| 結構 | 都有「**兩個依序校準的參數/階段**」（SeqCRC: cnf→loc/cls；CRS: gate→box）|
| 輸出 | 都輸出「一組框」而非單框，並對該集合給統計保證 |
| 多風險 | 都要處理「同時控多個風險」（SeqCRC 用 α 分帳+max；CRS 用 Bonferroni over grid）|
| 評估 | 都報 risk + set size 類指標 |

### 5.2 不同點（差異化主軸，逐項拆）
| 維度 | SeqCRC | CRS（本論文） |
|---|---|---|
| **任務本質** | 通用 OD：定位 + 分類「閉集 K 類 softmax」 | **語言指涉 grounding (REC/GREC)**：輸入是自然語言 query，輸出是「query 指到的框集合」；沒有閉集類別、核心是 query↔region 的語意對齊 |
| **偵測器配置** | **單一偵測器**（DETR 或 YOLO，同一個 f 同時出 box+class+conf）| **跨 base 異質組合 (cross-base composition)**：OWL-ViT 與 GroundingDINO 是**兩個不同模型**，一個負責 gate/abstention、一個負責 box；組合本身是貢獻 |
| **兩參數的語意** | cnf（信心門檻）+ loc（框 margin）/ cls（類別集合）— 都是**同一模型內部**的後處理旋鈕 | gate 門檻（控 abstention / no-target / recall）+ box 選取（控 compact referring set）— **跨兩個模型的功能分工 (factorization)** |
| **雙風險的內容** | loc 風險 + cls 風險（兩個任務的覆蓋）| **recall + abstention(no-target) 雙風險**：不只「框要蓋到」，還要「該棄答時棄答」（gRefCOCO 的 no-target 樣本）。abstention 是 SeqCRC **完全沒有**的軸 |
| **棄答 / no-target** | 無 abstention 機制；confidence threshold 只是 filter，不是「整張圖判定無目標」 | **核心貢獻就是 no-target 判定**（GREC 的 N-acc），這是 grounding 特有、SeqCRC 結構上不存在的問題 |
| **集合語意** | generic box set（每個框各自加 margin）+ per-box 類別集合 | **referring set**：所有框共同回答「哪些 region 符合這句話」，集合層級語意 |
| **評估指標** | Empirical Risk、Stretch、類別集合大小（CP 慣例）| **GREC 官方指標** Pr@(F1=1, IoU≥0.5)、N-acc、T-acc（grounding 社群指標，與下游使用直接對齊）|
| **多風險聯合手法** | α 分帳 + Corollary 1 的 max 上界（兩任務）| **LTT + Bonferroni over a grid**（multiple testing 路線）|
| **保證數學手法** | 樂觀/保守雙估計量 (λ^cnf_±) + 對稱性 → **單切分**有限樣本 | LTT 的 multiple-hypothesis testing → 高機率 (1−δ) 風險控制（PAC 式，與 SeqCRC 的「期望值」保證型態不同）|
| **應用定位** | 工業/安全關鍵認證（鐵路、自駕、航太）| 語言指涉 grounding 的 reliability / selective prediction 研究 |

### 5.3 一句話的「結構同 vs 本質異」
SeqCRC 與 CRS **在「兩個依序校準的 conformal 參數」這個骨架上同形**，但 (1) SeqCRC 的兩參數是**單一偵測器內部**的 conf/margin，CRS 的兩階段是**兩個異質模型**的 gate/box 功能分工；(2) SeqCRC 控的是 loc+cls 覆蓋，CRS 控的是 **recall + abstention** 雙風險，其中 abstention/no-target 是 grounding 獨有、SeqCRC 結構上不存在的維度；(3) 任務一個是 generic 閉集 OD，一個是開放語言指涉 grounding。**「結構撞名、語意不同軸」是 CRS 的防守線。**

---

## 6. 論文要怎麼引用 / 切割這篇（Related Work 可直接改寫的一段）

> **建議 related work 段落（中文草稿，可翻英）：**
>
> 在 conformal object detection 方向，最接近本文結構的是 Andéol et al. 的 Sequential Conformal Risk Control (SeqCRC) [arXiv:2505.24038]。SeqCRC 將 Conformal Risk Control 推廣到「1+2 個依序相依的純量參數」（信心門檻、定位 margin、分類集合），對單一通用偵測器 (DETR/YOLO) 在 MS-COCO 上同時控制定位與分類風險，並以樂觀/保守雙估計量取得「單一資料切分」的有限樣本保證。**我們的 CRS 在三個層面與之根本不同：**(i) **任務軸不同** —— SeqCRC 處理閉集通用 OD 的定位/分類，本文處理開放詞彙的語言指涉 grounding (GREC)，核心是 query↔region 對齊與 no-target 判定；(ii) **組合形態不同** —— SeqCRC 的兩參數是「單一偵測器內部」的後處理旋鈕，CRS 則是「兩個異質凍結模型」(OWL-ViT 作 abstention/no-target gate、GroundingDINO 出 box) 的功能分工式組合 (factorization)，並以 ablation 證實此分工非單純 ensemble；(iii) **風險軸不同** —— SeqCRC 控定位+分類覆蓋，CRS 則同時對 **recall 與 abstention (no-target)** 給保證，而 abstention 是 grounding 特有、SeqCRC 結構上不涉及的維度。此外，SeqCRC 採 CRC 的期望值保證並以 α 分帳處理多任務，CRS 則採 LTT + Bonferroni-over-grid 取得高機率 (PAC 式) 多風險控制，並以 GREC 官方指標 (Pr@(F1=1), N-acc, T-acc) 評估，與下游使用直接對齊。

**切割要點（口袋清單，被審稿人 challenge 時用）：**
1. **「你不就是 SeqCRC 換個任務？」** → 不是。SeqCRC 沒有 abstention/no-target 軸，而那是 grounding 的核心；SeqCRC 是單偵測器內部旋鈕，CRS 是跨異質 base 的功能 factorization（有 2×2 ablation 證成）。
2. **「兩個依序參數的數學是不是一樣？」** → 同屬 risk-control 家族但路線不同：SeqCRC 走 CRC 的雙估計量+對稱性（期望值保證、單切分）；CRS 走 LTT+Bonferroni（PAC 式高機率、grid multiple testing）。可主動承認結構同源，把差異收斂到「保證型態 + 風險語意」。
3. **指標不可比**：SeqCRC 報 stretch / 類別集合大小；CRS 報 GREC 指標。可在實驗章節說明「為何 grounding 必須用 GREC 而非 generic CP set-size」。
4. **可引為「同期獨立、結構最近鄰」**，承認其貢獻（首個 holistic conformal OD 框架 + 大規模 benchmark + 開源 toolkit），藉此凸顯 CRS 把同一骨架搬到「語言指涉 + 跨 base + 棄答」這個更難且 SeqCRC 未觸及的設定。

---

## 7. 個人評價

### 優點
- **問題定位漂亮**：把「conformal OD 只做子步驟」這個 gap 講得很清楚，holistic 框架 + 三參數拆解確實補了洞。
- **數學賣點硬**：相對 Xu et al. (ranked retrieval) 的「兩次切分 or asymptotic」，SeqCRC 用樂觀/保守雙估計量做到**單切分有限樣本**，這是真正可被引用的技術貢獻。
- **工程誠實**：單調化技巧、GIoU 失效、heavy-tail set size、斑馬 false positive、COCO 標註不一致 —— 把 conformal OD 的實務陷阱攤開講，對後續研究很有價值。
- **可重現性強**：開源 toolkit + 數百 run benchmark，社群基礎設施貢獻明確。

### 缺點 / 侷限
- **只保證 recall，不保證 precision**：作者自己承認 precision 非單調、無法在框架內控制。對安全關鍵應用（要少誤報）其實是大坑。
- **期望值保證偏弱**：是 marginal 期望值，不是 per-instance、不是 PAC 高機率。實務上「平均達標但某些圖爆掉」的風險仍在（heavy tail 正是這個徵兆）。
- **聯合保證過保守**：Corollary 1 的 max ≤ a+b 分帳，作者自承 global risk「常常不必要地保守」。
- **set size 對 α 不平滑**：小改 α 可能讓集合大跳，調參體驗差，實務難用。
- **依賴 ground-truth 品質**：若標註比模型還差，conformal 會被迫加超大修正→集合不可用。這在 grounding（標註更主觀）只會更嚴重，對 CRS 反而是要警惕的共通風險。
- **任務範圍窄於它的雄心**：號稱 holistic，但實際只在 COCO + 兩個偵測器、閉集分類上驗證；沒碰開放詞彙、沒碰 abstention —— 正好是 CRS 的主場。

### 對本論文 (CRS) 的戰術啟示
- SeqCRC 是**最該主動引用且主動切割**的對象，不要迴避。承認結構同源能降低審稿人「你在規避」的疑慮。
- 把 CRS 的差異化**收斂到三個不可被吸收的軸**：abstention/no-target、cross-base factorization、PAC-式雙風險（recall+abstention）。這三個 SeqCRC 結構上都沒有。
- SeqCRC 的 heavy-tail 與 precision 缺口，可在 CRS 的 discussion 借力：說明 CRS 用 LTT 高機率保證 + abstention 軸，正是在補「只控 recall、只給期望值」的不足。
