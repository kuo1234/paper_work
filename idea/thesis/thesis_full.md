# 選擇性指稱定位：凍結基礎模型的事後可靠性與風險受控指稱集合

> 碩士論文中文初稿（合併稿，由 ch00–ch12 串接生成；單章可見同目錄各檔）
> 生成日期：2026-06-22　主線：Selective Grounding / CRS（三貢獻整合版）

---

## 目錄

- 摘要
- 第 1 章　緒論
- 第 2 章　相關工作
- 第 3 章　方法框架
- 第 4 章　實驗設定
- 第 5 章　C1：與定位相關的不確定性訊號審計
- 第 6 章　C2：選擇性風險控制
- 第 7 章　C3：事後無目標閘
- 第 8 章　M4：完整 GREC 的 exact-match 之牆
- 第 9 章　CRS：跨基礎模型保形組合指稱集合（base claim）
- 第 9b 章　Decomp CRS：候選池維度（第二主結果）
- 第 9c 章　WB-Gate CRS：閘分數維度（第三主結果）
- 第 9d 章　統一框架：三正交維度與 frozen detector frontier
- 第 10 章　C4：定位不確定性的 base-invariant 結構
- 第 11 章　Cost–Risk Pareto
- 第 12 章　結論與限制

---


---

# 摘要

指稱語表達理解（Referring Expression Comprehension, REC）要求模型依據一段自然語言描述，在影像中定位出被指稱的物件。近年凍結式（frozen）的視覺—語言基礎模型（如 CLIP-VG、OWL-ViT）已能在零樣本或輕量微調下達到可用的定位準確率，但它們**始終輸出一個點預測**：對每個查詢都吐出一個（或經 argmax 選出的一個）框，既無法在「描述其實沒有對應物件」（no-target）時棄答，也無法在多個候選同樣合理時表達不確定。當這類模型被放進需要可靠性保證的下游系統時，這種「永遠給答案、且不標示信心」的行為構成一道實際的可靠性缺口。

本論文不從「把基礎模型訓練得更準」切入，而把基礎模型視為**凍結的黑盒**，研究一個正交的問題：**凍結式指稱定位基礎模型，是否在其單次前向傳遞的廉價副產品中，暴露出可重用、且與定位任務本身相關的不確定性結構，使我們能在不更新任何基礎模型參數、近乎零訓練成本下，事後（post-hoc）校準出可靠的「作答／棄答／集合化」決策？** 我們以一個輕量的信心策略（belief policy）`π` 觀測基礎模型的副產品（候選幾何、改寫提示下的指稱身分穩定度、關係／屬性殘差、跨模型一致性），並量化其可靠性、成本、跨基礎模型的可轉移性，以及與理想預言（oracle）之間的差距。

我們在 RefCOCO 系列與 gRefCOCO 上，以兩個結構迥異的凍結基礎模型（CLIP-VG 的單框迴歸、OWL-ViT 的開放詞彙偵測）為實例，建立四項貢獻：

- **C1（不確定性訊號審計）**：系統性量測哪些信心訊號對「答對與否」與「無目標」具有資訊量、哪些沒有。跨提示一致性（cross-prompt consistency）作為與定位相關的擾動訊號，在三個 split 上對答對與否的 AUROC 達約 0.72；在 OWL-ViT 上，分數熵（score entropy）對答對與否的 AUROC 達約 0.68，是該基礎模型最具資訊量的分數型訊號。

- **C2（選擇性風險控制）**：以校準後的信心做選擇性預測，其風險—覆蓋（risk–coverage）曲線優於隨機、單一分數／邊際／熵閾值等天真基線，並以 oracle 線量化尚未被利用的訊號上限。

- **C3（事後無目標閘）**：在 gRefCOCO 上，凍結基礎模型恆吐 argmax 框時無目標準確率為零；輕量閘輸出 P(no-target) 後，無目標 AUROC 達 0.74–0.82（三個 split），相對於強迫輸出的基線提供基礎模型原本不具備的棄答能力。我們以 HieA2G、VIRO、True/False Verification 等近期工作為文獻定位。

- **C4（跨基礎模型轉移與成本分析）**：僅用基礎模型正規化（base-normalized）特徵訓練的策略，可在不重新擬合下從一個基礎模型套到另一個，其風險—覆蓋表現接近原生（native），顯示定位失敗具有跨基礎模型可重用的結構；並以成本—風險 Pareto 報告推論開銷。

更進一步，我們指出**點預測在完整 GREC（generalized REC）的 exact-match 評估下存在原理性的天花板**：在官方 Pr@(F1=1, IoU≥0.5) 下，逐樣本的 oracle-τ 上界僅 0.19–0.24，瓶頸是多目標的計數／集合預測問題，超出單一信心閾值的能力範圍——這不是方法失敗，而是對「事後校準在何處停止有效」的明確界定。

由此引出本論文的主結果——**跨基礎模型保形組合指稱集合（Cross-Base Conformal Referring Set, CRS）**：讓 `π` 不再輸出點預測，而輸出一個基數可為零、一或多的**風險受控指稱集合**。我們以 Learn-then-Test（LTT，搭配 Bonferroni 校正）在校準 split 上**聯合**校準三個有界風險——已作答目標漏檢率 R1≤α、無目標誤選率 R2≤β、目標棄答率 R3≤γ——並給出有限樣本保證。CRS 把風險控制的代價**因式分解**為「該不該答」（gate）與「答得準」（box）兩個正交瓶頸，以一個凍結基礎模型（OWL-ViT）當 gate、另一個（GroundingDINO）當 box selector。在 α=β=0.3 下，組合（COMPOSE）於 val/testA/testB 三個 split 分別輸出 **3.24 / 2.02 / 3.50** 個框，接近真實基數，且 R1/R2 全數守住、目標棄答率全數低於 γ；其集合大小的自助法（bootstrap）信賴區間與純 OWL-ViT 基線完全分離（公平比較下縮小 2.0–4.5 倍）。2×2 消融顯示，唯有「OWL-ViT 當 gate、GroundingDINO 當 box」的對角線組合能同時守住三風險並產生緊緻集合，證明此因式分解並非單純的集成（ensemble）。

我們亦誠實標註兩項限制：影像層級分群評估在 testA/testB 上樣本量不足而留空；長指稱語句下 R2（無目標誤選）會退化，反映 OWL-ViT 閘對長句的弱點。

在 CRS 確立「凍結基礎模型 + 事後 LTT = 有保證的指稱集合」這個基礎主張之後，我們進一步指出此系統有**三個正交、可獨立替換的可改善維度**——候選池（candidate pool）、閘分數（gate score）、憑證（certificate）——並各給出實例化。**Decomp CRS（候選池維度）** 以凍結 VLM router 把目標語句拆成子部件、各跑 GroundingDINO 後取聯集候選池；同一 LTT 協議下只換候選池，多目標漏檢率 R1 在三個 split 全部 ≤ frozen（0.165/0.244/0.163 vs 0.191/0.260/0.168）而 R2／集合大小持平，且贏過所有 full-pool 整理（NMS／top-K），是純賺 R1 而不付代價的第二主結果。**WB-Gate CRS（閘分數維度）** 以 21 維凍結特徵上的可解釋白箱閘（EBM）取代單一分數閘，在同分布下把原本 infeasible 的設定變 feasible（rule INFEASIBLE → 可行集 #feas=115）並大幅改善 R2/R3，與 Decomp 互補疊加（集合大小 3.68→2.37）；**憑證維度** 則以 Hoeffding→Hoeffding-Bentkus 免費把可行區擴大三倍（集合大小 5.07→3.72）。三維度正交可組合，共同決定凍結偵測器給定下可達的 risk–cost frontier，我們測繪此 frontier 並釘出天花板。

我們對這兩個額外主結果也誠實標註三項新限制：(1) WB-Gate 的 learned 閘邊界**不跨資料集轉移**——固定門檻下目標通過率在 testA/testB 崩塌（0.89→0.35），而無訓練的規則閘反而穩定（0.88→0.81），顯示「學一個更強的可靠性訊號 ≠ 更可靠的跨分布轉移」，這是 selective prediction 的誠實教訓；(2) Decomp 的 over-decomposition——單目標語句的退步約半數源於 router 過度拆解（自身錯）；(3) Decomp 雖 training-free 但**非 compute-free**，每個目標語句多一次 VLM router 呼叫。

**關鍵詞**：指稱語表達理解、廣義 REC、選擇性預測、棄答、保形預測、Learn-then-Test、風險控制、凍結基礎模型、不確定性校準、跨模型轉移。

---

# 第 1 章　緒論

## 1.1 研究背景

指稱語表達理解（Referring Expression Comprehension, REC）是視覺—語言研究的核心任務之一：給定一張影像 `I` 與一段自然語言描述 `e`（例如「左邊穿紅衣服的女孩」），模型須在影像中定位出 `e` 所指稱的物件，輸出其邊界框（bounding box）。REC 串接了語言理解、視覺辨識與兩者的對齊，是機器人指令跟隨、影像編輯、視覺問答等下游應用的基礎能力。

近年，凍結式（frozen）的視覺—語言基礎模型大幅推進了 REC 的可用性。以對比式預訓練的 CLIP 為骨幹的 CLIP-VG、開放詞彙偵測器 OWL-ViT、以及 query-based 的 GroundingDINO 等，能在零樣本或輕量微調下對真實影像達到可用的定位準確率。這類模型的吸引力在於「即插即用」：不需為每個新場域重新訓練一個專用定位器，就能套用其廣泛的視覺—語言先驗。

然而，這類凍結基礎模型在標準 REC 設定下有一個共同的、結構性的行為特徵：**它們永遠輸出一個點預測**。模型對每個查詢都吐出一個框（或從候選中以 argmax 選出一個框），並且**不附帶可校準的信心**。標準 REC 的評測協定也默認「每個描述恰有一個對應物件」，於是模型從不需要、也沒有機制去表達兩種同樣重要的情形：

1. **無目標（no-target）**：描述其實在影像中沒有對應物件。凍結基礎模型仍會強行吐出一個 argmax 框，造成必然的誤報。
2. **指稱不確定**：多個候選同樣合理，或同義改寫描述就會讓模型改選另一個物件。點預測無法表達這種模稜兩可。

廣義 REC（Generalized REC, GREC）與 gRefCOCO 資料集把上述情形顯式納入：一個描述可對應到零個（no-target）、一個或多個（multi-target）物件。這暴露出凍結基礎模型的可靠性缺口——在需要「知道自己何時不該答」的下游系統中，一個永遠給答案、且不標示信心的定位器是不安全的。

## 1.2 問題陳述

本論文研究的核心問題是：

> **凍結式指稱定位基礎模型，是否在其單次（或少數幾次）前向傳遞的廉價副產品中，暴露出可重用、且與定位任務本身相關的不確定性結構，使我們能在不更新任何基礎模型參數、近乎零訓練成本下，事後（post-hoc）校準出可靠的「作答／棄答／集合化」決策？並量化這種校準的可靠性、成本、跨基礎模型可轉移性，以及與理想預言（oracle）的差距？**

這個提法刻意與兩條既有路線區隔：

- **不是**把基礎模型訓練得更準（與 trained GREC 架構如 LIHE、HieA2G 比 accuracy）；
- **不是**用外部大型模型逐物件重新驗證以取代定位（如 True/False Verification）、或把查詢拆成可執行程式逐運算子做符號驗證（如 VIRO）。

我們把基礎模型視為**凍結黑盒**，只在其決策層（decision layer）附加一個輕量的信心策略（belief policy）`π`。`π` 觀測基礎模型免費或低成本的副產品——候選框的幾何分布、同義改寫提示下「哪個候選被選中」的身分穩定度、關係／屬性的判別性殘差、與另一個基礎模型的一致性——並據此決定每個查詢該**作答（answer）**、**棄答（abstain）**、或輸出一個**風險受控的集合（risk-controlled set）**。本論文量化的，是 `π` 能從基礎模型副產品中可靠校準到何種程度、代價多少、能否跨基礎模型轉移、以及在何處原理性地失效。

## 1.3 研究動機與切入角度

本研究的切入角度承襲一個方法論信念：**結構化的信心應決定「何時」介入，而非盲目地替換基礎模型。** 這個觀點源自先前在機器人視覺—語言操作（OpenVLA + LIBERO）上的實證——always-on 或基於幾何規則的介入都會傷害成功率，唯有選擇性（selective）的介入才有正面價值。把這個論點搬到文獻空白、資料乾淨、基礎模型真正能用的指稱定位場域，便構成本論文。

之所以鎖定**事後可靠性研究（post-hoc reliability study）**而非提出新的定位方法，有三個理由：

1. **下限穩固**。選擇性預測的理論保證告訴我們：只要不確定性訊號比亂猜好，風險—覆蓋曲線必有改善。因此即使最壞情況，本研究仍有一個非平凡的正面結論。
2. **文獻區辨清楚**。verification-based 的新定位方法路線已被 VIRO（CVPR 2026）與 True/False Verification（arXiv:2509.09958）占據；硬碰會落入無法區辨的近鄰。但「凍結基礎模型自身副產品的不確定性結構是否可重用、可跨基礎模型轉移」是這些 per-pipeline / per-VLM 方法**無法**主張的科學問題。
3. **務實可行**。整個研究只需離線 dump 基礎模型的副產品 + 輕量校準器，不需重訓任何大型模型，計算與資料成本低，適合在有限算力下完成。

## 1.4 貢獻

本論文的貢獻分為兩層。第一層是四項以凍結基礎模型副產品為基礎的**可靠性量測**（C1–C4），構成全篇的測量地基；第二層是把框架從「量測可靠度」推進到「建構有分布無關保證的集合」的**主結果 CRS**。

**量測地基（C1–C4）**

- **C1　與定位相關的不確定性訊號審計。** 系統性量測哪些信心訊號對「答對與否」（REC correctness）與「無目標」（gRefCOCO no-target）具有資訊量。主要發現：跨提示一致性是與定位相關的擾動訊號（量測指稱身分在良性改寫下的穩定度，而非泛用信心），對答對與否的 AUROC 約 0.72；在 OWL-ViT 上分數熵對答對與否的 AUROC 約 0.68，為其最具資訊量的分數型訊號。

- **C2　選擇性風險控制。** 以校準後的信心做選擇性預測，其風險—覆蓋曲線在 RefCOCO 上優於隨機、單一分數／邊際／熵閾值等天真基線，並以 oracle 線量化尚未被利用的訊號上限（oracle gap）。

- **C3　事後無目標閘。** 在 gRefCOCO 上，恆吐 argmax 框的凍結基礎模型其無目標準確率為零；輕量閘以基礎模型副產品輸出 P(no-target)，三個 split 的無目標 AUROC 達 0.74–0.82，提供基礎模型原本不具備的棄答能力。以 HieA2G / VIRO / True-False Verification 做文獻定位。

- **C4　跨基礎模型轉移與成本分析。** 僅用基礎模型正規化特徵訓練的策略，可在不重新擬合下從一個基礎模型轉移到另一個，其風險—覆蓋表現接近原生，顯示定位失敗具有**跨基礎模型可重用的結構**——這是 per-pipeline / per-VLM 驗證方法無法做出的主張。並以成本—風險 Pareto 報告推論開銷。

**邊界與主結果（M4 → CRS）**

- **M4　完整 GREC 的壓力測試（邊界分析，非方法貢獻）。** 我們把點預測策略推到完整 GREC 的 exact-match 評估（官方 Pr@(F1=1, IoU≥0.5) / N-acc / T-acc），定位出事後校準停止有效之處：逐樣本的 oracle-τ 上界僅 0.19–0.24，瓶頸是多目標的計數／集合預測，超出單一信心閾值的能力。這是對方法邊界的明確界定。

- **CRS　跨基礎模型保形組合指稱集合（主結果之一）。** 讓 `π` 輸出基數可為 0／1／多的風險受控集合。以 Learn-then-Test（搭配 Bonferroni 校正）在校準 split 上**聯合**校準三個有界風險（R1 已作答目標漏檢率、R2 無目標誤選率、R3 目標棄答率），給出有限樣本保證；並把代價因式分解為 gate（OWL-ViT）與 box（GroundingDINO）兩個正交瓶頸。在 α=β=0.3 下，組合於三個 split 輸出 3.24 / 2.02 / 3.50 個框，三風險全守、集合大小信賴區間與純 OWL-ViT 分離；2×2 消融證明此因式分解並非單純集成。

**三正交維度（CRS 之上的兩個額外主結果 + 統一框架）**

CRS 確立 base claim 之後，我們進一步發現：在 frozen detector 給定下，CRS 系統有三個**正交、可獨立替換**的可改善維度——候選池（candidate pool）、閘分數（gate score）、憑證（certificate）。兩個額外主結果分別實例化前兩個維度：

- **main 2（Decomp CRS，候選池維度）。** 在不碰 detector、不訓練的前提下，用 frozen VLM router 把 target-present expression 拆成子部件、各跑 GroundingDINO 後取 union 候選池。同一 LTT 協議下只換候選池，R1（多目標漏檢）三個 split 全部 ≤ frozen（0.165/0.244/0.163 vs 0.191/0.260/0.168），R2 與集合大小持平——**純賺 R1**。並以三道硬防線（robustness、strong full-pool baseline、frontier）堵住 matched 假象。

- **main 3（WB-Gate CRS，閘分數維度）。** 把單一 OWL top1 gate 換成 21 維 frozen 特徵上的可解釋 white-box gate（EBM）。同分布下把單一分數 infeasible 的設定變 feasible（rule INFEASIBLE → WB #feas=115），大幅改善 R2/R3，並與 Decomp 互補疊加（set size 3.68→2.37）。但 learned gate boundary **cross-dataset 不轉移**（rule pass 0.88→0.81 穩 vs WB 0.89→0.35 崩），構成一個有研究價值的 negative finding。**第三維度（certificate）** 由 Hoeffding→Hoeffding-Bentkus 免費實例化（feasible 區 ×3，set size 5.07→3.72）。

- **統一框架。** 三維度正交可組合；在 frozen detector 給定下，(pool, gate, bound) 三軸共同決定可達的 risk-cost frontier。我們測繪此 frontier 並釘出天花板（frozen OWL+GDINO 的 scoring 上限）。

## 1.5 論文結構

- **第 2 章（相關工作）**：定位本研究於 conformal 物件偵測、zero-shot VLM conformal、trained GREC 架構、VLM 選擇性預測、以及 verification-based REC（VIRO、True/False Verification）之間的座標。
- **第 3 章（方法框架）**：定義以凍結基礎模型 + 輕量信心策略 `π` 為核心的統一框架；所有實驗（C1–C4、M4、CRS）皆為其實例。
- **第 4 章（實驗設定）**：資料集、基礎模型、split／校準／閾值協定、指標、防洩漏 checklist。
- **第 5–7 章（C1–C3）**：不確定性訊號審計、選擇性風險控制、事後無目標閘的結果章。
- **第 8 章（M4）**：完整 GREC exact-match 的邊界分析。
- **第 9 章（CRS）**：主方法與主結果——跨基礎模型保形組合指稱集合（base claim）。
- **第 9b 章（Decomp CRS）**：候選池維度——第二主結果，純賺 R1。
- **第 9c 章（WB-Gate CRS）**：閘分數維度——第三主結果，feasibility/R2/R3 改善 + cross-dataset 不轉移的 negative finding。
- **第 9d 章（統一框架）**：三正交維度收斂、互補性論證、frozen detector frontier 與天花板。
- **第 10–11 章**：跨基礎模型轉移（C4）、成本—風險 Pareto。
- **第 12 章（結論）**：六貢獻收束、限制全景、未來工作。

> 章序邏輯：C1→C2→C3 是難度遞增的測量地基，M4 證明點預測撞牆，CRS 是把框架推到「有保證的集合預測」的解法；9b/9c/9d 沿三正交維度把可達 frontier 推開並收斂成統一框架，C4／cost 是橫向分析，最後收束於結論。

---

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

---

# 第 3 章　方法框架

本章定義全篇統一的方法框架：以**凍結基礎模型 + 輕量信心策略**為核心，所有實驗（C1–C4、M4、CRS）皆為其實例。框架的主角不是基礎模型 `g`，而是附加在其決策層的信心策略 `π`——所有貢獻都是「`π` 能從 `g` 的副產品學到多少可靠決策」。本章先給出問題形式化（§3.1）、凍結基礎模型與信心訊號（§3.2–3.3）、選擇性預測與跨基礎模型轉移（§3.4）、無目標閘與廣義 REC 邊界（§3.5），再宣告把框架從「測量可靠度」推進到「建構有保證的集合」的關鍵一步——風險受控指稱集合（§3.5b），最後鎖定評測協定（§3.6）。

## 3.1 問題形式化

我們研究**凍結基礎模型下的選擇性指稱定位**。一個指稱定位模型 `g` 把影像 `I` 與描述 `e` 映射到一或多個邊界框。在標準 REC 設定下 `g` 回傳單一框；在廣義設定（GREC）下，它須回傳一個基數可為零（no-target）、一或多（multi-target）的框集合。

我們把 `g` 視為**凍結黑盒**：其參數從不更新。在 `g` 之上，我們附加一個輕量的**信心策略** `π`，它觀測 `g` 單次（或少數幾次）前向傳遞的廉價副產品，並對每個查詢決定下列動作之一：

- **作答（answer）**——輸出基礎模型的預測；
- **棄答（abstain）**——宣告無目標／拒絕作答；
- **（可選）重排（re-rank）**——施加一個與定位相關的候選操作；
- **建構風險受控集合（construct a risk-controlled set）**——不輸出點預測，而回傳一個經校準的候選框集合（可為空），其風險以有限樣本保證受控（即第 9 章 CRS）。

本論文量化的，是跨基礎模型與資料集，`π` 能從 `g` 的副產品中**多可靠地**被校準、**代價多少**、**跨基礎模型轉移得多好**、**點預測在何處原理性地失效**（oracle gap／邊界，第 8 章 M4），以及**當點預測失效時，集合化的 `π` 能提供什麼保證**（CRS）。

## 3.2 凍結基礎模型

我們在兩個結構迥異的凍結基礎模型上實例化框架，以壓力測試信心訊號是否與基礎模型架構無關：

- **CLIP-VG**（單框迴歸）：經 `bbox_embed(reg_token).sigmoid()` 恰輸出一個框，**無候選清單、無分數**。這迫使信心只能來自**擾動**訊號，而非分數統計——此特性反而有利於跨基礎模型轉移（§3.4）。
- **OWL-ViT**（開放詞彙偵測器）：單次前向傳遞輸出多個帶分數候選框，分數為免費副產品，支援分數型信心與**原生棄答**（最大分數低 ⇒ 傾向無目標）。

兩個基礎模型的訊號家族完全不同（擾動 vs 分數），卻都 informative，且其中一類可跨基礎模型轉移——這是 C4 的科學主張（第 10 章）。在 CRS 章再引入第三個凍結基礎模型 **GroundingDINO** 作為高召回的 box selector。

## 3.3 信心訊號

所有訊號皆為事後（post-hoc，不更新基礎模型），依成本與可轉移性分兩族：

**(A) base-agnostic 擾動訊號**（成本 K× 前向，K 個改寫）：
- `cross_prompt_consistency`：K 個改寫提示下預測框的平均成對 IoU。一致性低 ⇒ 指稱身分在良性改寫下不穩 ⇒ 錯誤風險高。這是**與定位相關**的——量測指稱穩定度，而非泛用信心。
- `prompt_box_dispersion`：改寫提示下預測框中心的離散度（除以 √area 正規化）。

**(B) base-specific 分數訊號**（成本 1× 前向，免費副產品；OWL-ViT）：
- `top1_score`、`margin12`（= top1 − top2）、`score_entropy`（softmax 候選分數的熵）、`score_mean_topk`。

(A) 貴但可轉移、(B) 免費但 base-specific——這個「成本↔可轉移性」的二分本身就是貢獻，直接撐起成本—風險 Pareto（第 11 章）與 C4（第 10 章）。校準器是標準化訊號上的邏輯迴歸，於 held-out 校準 split 擬合；其預測機率即作為排序選擇性預測的信心／棄答分數。我們刻意把校準器保持最小——主張是關於**訊號資訊量**，而非校準器容量。

## 3.4 選擇性預測與跨基礎模型轉移

**選擇性預測（C2）。** 給定 `π` 的信心 `c(x)`，回答信心最高的 coverage 比例查詢、棄答其餘。我們報告**風險—覆蓋曲線**及其面積（AURC，越低越好），對照隨機、單訊號閾值，以及以真實 correctness 排序的 oracle。

**跨基礎模型轉移（C4）。** 這是核心的科學測試：僅用**base-normalized** 訊號在一個 *source* 基礎模型上擬合 `π`，再**不重新擬合**地套用到一個 *target* 基礎模型。若策略可轉移，則定位失敗具有*可重用、基礎模型共享的結構*——這是任何 per-pipeline 驗證方法（VIRO、True/False Verification）無法做出的主張。轉移以「被轉移策略在 target 基礎模型上的 AURC」對照「target 原生策略」與「隨機」量測。

C4 的乾淨版主張是：raw consistency 零參數、無 refit，從 CLIP-VG 套到 OWL-ViT，AURC 接近原生——亦即 grounding 不確定性具有 base-invariant 結構。這比 C1/C2 的審計更像「發現」，也是對抗「只是 calibration／threshold」批評的護城河。

## 3.5 無目標閘與廣義 REC

**無目標閘（C3）。** 一個恆吐 argmax 框的凍結基礎模型，依其建構**毫無棄答能力**（無目標查詢的 N-acc = 0）。在 gRefCOCO 上，我們校準 `π` 以基礎模型副產品輸出 `P(no-target)`，再二分為棄答決策。指標：無目標 AUROC、balanced accuracy、N-acc——全部 leak-safe（閾值固定於 calib）。

**廣義 REC 邊界（M4）。** 把 `π` 推到完整 GREC 需要輸出*正確的框集合*，以官方 **Pr@(F1=1, IoU≥0.5) / N-acc / T-acc**（忠實 greedy-IoU 匹配）評測。這是刻意的**壓力測試**，而非方法主張：它定位出事後校準停止有效之處（多目標 exact-match 是超出單一信心閾值的計數／集合預測問題）。我們以逐樣本的 oracle-τ 上界與 bootstrap CI 量化此邊界。M4 在方法框架中先聲明為 boundary analysis，避免在結果章被讀成失敗——「刻意把方法推到斷裂點並量化斷在哪」是成熟度，不是減分。

## 3.5b 風險受控指稱集合（CRS）：從點預測到有保證的集合

M4 的牆（多目標 exact-match 超出單一信心閾值的動作空間）促成框架的最後一步：讓 `π` 不再輸出**點預測**，而輸出一個 **風險受控指稱集合** `S(I,e) ⊆ {候選框}`，基數可為 0（棄答／無目標）、1 或多。我們以 **Learn-then-Test（LTT）** 在校準 split 上**聯合**校準三個有界風險並給出有限樣本保證：

- **R1 已作答目標漏檢率（answered-target FNR）** ≤ α：在未棄答的 target-present 查詢上的漏檢率（**條件**風險，不稱 recall guarantee）；
- **R2 無目標誤選率（no-target false selection）** ≤ β：no-target 樣本吐出任何框的比例；
- **R3 目標棄答率（target deferral）** ≤ γ：target-present 樣本被吐空集的比例。

進一步地，CRS 把風險控制的代價**因式分解**為 gate（該不該答）與 box（答得準）兩個正交瓶頸，並以 **cross-base conformal composition**（一個凍結基礎模型當 gate、另一個當 box selector）在三保證下達到接近真實基數的緊緻集合。完整方法、可行域與結果見第 9 章。

CRS 把 C1–C4 的 measurement 與 M4 的 boundary 縫進同一條線：訊號可測（C1–C4）→ 點預測撞牆（M4）→ 換成有保證的集合預測（CRS）。本章不放 CRS 數字（數字在第 9 章），避免重複。CRS 不宣稱解決完整 GREC 的 exact-match，而是把問題從 exact-match 點預測 pivot 成 risk-controlled 集合建構——這個 metric pivot 在第 8、9 章都會明寫。

## 3.5c 三正交維度：CRS 系統的可改善軸

CRS（§3.5b）確立 base claim 後，本論文進一步把 CRS 系統拆成三個**正交、可獨立替換**的維度。
關鍵觀察：在 frozen detector 給定下，一個 risk-controlled referring set 的產生鏈是

```
候選池 (pool) → 閘分數 (gate) → LTT 憑證 (certificate) → referring set
```

三個環節各自可在**不碰 detector 權重、不改其他兩環**的前提下替換，因此互相正交，可獨立疊加：

- **維度 1 候選池（pool）**：CRS 用 frozen base 的原生候選池。可換成 decomposition-union pool
  （凍結 VLM router 拆子部件 → 各跑 GDINO → union），只改 box 候選來源，gate 與 certificate 不變。
  主攻 R1（多目標漏檢）。完整方法與結果見第 9b 章。
- **維度 2 閘分數（gate）**：CRS 用單一 OWL top1 分數當棄答 gate。可換成 21 維 frozen 特徵上的
  可解釋 learned gate（EBM），只改 gate decision 層，pool 與 certificate 不變。主攻 R2/R3 與可行性。
  完整方法與結果見第 9c 章。
- **維度 3 憑證（certificate）**：CRS 的 LTT 用 Hoeffding bound 算 p-value。可換成更緊的
  Hoeffding–Bentkus（HB），同切分／grid／Bonferroni、**只換 p-value**，免費擴大可行域。見第 9c 章。

**統一協議的意義**：因為三維度共用同一套 LTT 風險定義（R1/R2/R3）、同一套 calib/test 切分與
防洩漏規則，任意 (pool, gate, bound) 組合都套同一協議，四個版本（Rule / Decomp / WB-Gate /
WB+Decomp）可在同一張表逐欄比較。三維度共同決定 frozen detector 給定下可達的 risk–cost frontier
（統一論述見第 9d 章）。

## 3.6 評測協定

- **切分與洩漏**：所有閾值／標準化／校準器擬合都來自校準 split；test 只評估一次。跨基礎模型轉移不讓 target-test 統計回流到 source 訓練。
- **成本**：推論成本以「每查詢前向傳遞次數」加實測 throughput（GB10）報告，不以形容詞描述（第 11 章 Pareto）。
- **不確定性**：所有 headline 指標附 **bootstrap 95% CI**（≥1000 次 resample test，校準器固定）。
- **保證 vs CI（CRS 專用，必須分清）**：對 CRS，**LTT p-value + Bonferroni** 檢定是*有限樣本風險控制保證*；固定 selected config 上的 bootstrap CI 只是*經驗穩定度*，**不是**保證。兩者分開報、never conflated。CRS 閾值網格只建構於**校準 covariate**（無 evaluation covariate 或 label 進入網格建構、風險檢定或操作點選擇）。
- **Oracle gap**：每張選擇性圖都附 oracle 線，界定還有多少訊號未被利用。

這六條（洩漏、成本、顯著性、保證≠CI、網格 calib-only、oracle 上界）是審查最常攻的點；方法章先把協定講死，結果章便能省去反覆辯護。

---

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

---

# 第 5 章　C1：與定位相關的不確定性訊號審計

本章回答四貢獻的第一個、也是最基礎的問題：**凍結指稱定位基礎模型的廉價副產品中，哪些信心訊號對「答對與否」具有資訊量，哪些沒有？** 這是後續所有選擇性決策（C2 風險控制、C3 無目標閘、CRS 集合化）的地基——若訊號本身不 informative，後面一切都無從談起。我們刻意把校準器保持最小（標準化後的單訊號排序，或邏輯迴歸），因為本章的主張是關於**訊號的資訊量**，而非校準器的容量。

## 5.1 審計設定

我們以 AUROC（訊號排序「答對與否」的能力）與 AURC（以該訊號做選擇性預測的風險—覆蓋曲線下面積，越低越好）兩個互補指標量測每個訊號。AUROC 是門檻無關的排序品質；AURC 則直接對應下游選擇性預測的代價。所有指標附自助法（bootstrap，≥1000 次 resample test 樣本，校準器固定）95% 信賴區間。

兩個凍結基礎模型暴露的訊號家族不同，因此分開審計：

- **CLIP-VG**（單框迴歸，無候選分數）：只能用**擾動訊號**——跨提示一致性（cross-prompt consistency）與提示框離散度（prompt box dispersion）。
- **OWL-ViT**（多候選帶分數）：可用**分數型訊號**——分數熵（score entropy）、top1−top2 邊際（margin12）、top1 絕對分（top1 score）——以及同樣可算的跨提示一致性。

`correct@0.5`（IoU≥0.5 且非無目標）為正確性標籤。RefCOCO 系列三個 split（val/testA/testB）分開報告。

## 5.2 CLIP-VG：擾動訊號的資訊量

CLIP-VG 作為凍結 REC 基礎模型本身已相當準（三個 split 的 base accuracy 為 val 84.28%、testA 87.75%、testB 78.43%），因此這裡的審計問的是「在一個已經不錯的基礎模型上，擾動訊號能否進一步排出哪些查詢可能答錯」。

**表 5.1　CLIP-VG 擾動訊號對答對與否的 AUROC / AURC（括號為 95% bootstrap CI）**

| Split | n | base acc | 訊號 | AUROC | AURC | random AURC |
|---|---|---|---|---|---|---|
| val | 10834 | 0.843 | cross-prompt consistency | **0.722** (0.710–0.735) | **0.075** (0.070–0.080) | 0.157 |
| val | | | prompt box dispersion | 0.654 (0.642–0.667) | 0.089 (0.083–0.095) | 0.157 |
| testA | 5657 | 0.878 | cross-prompt consistency | **0.706** (0.686–0.726) | **0.062** (0.055–0.070) | 0.123 |
| testA | | | prompt box dispersion | 0.638 (0.618–0.659) | 0.075 (0.066–0.083) | 0.123 |
| testB | 5095 | 0.784 | cross-prompt consistency | **0.723** (0.707–0.738) | **0.105** (0.096–0.114) | 0.216 |
| testB | | | prompt box dispersion | 0.680 (0.663–0.694) | 0.114 (0.105–0.123) | 0.216 |

**解讀。** 跨提示一致性在三個 split 上對答對與否的 AUROC 穩定落在 0.71–0.72，明顯優於隨機（0.5），且其信賴區間在三個 split 上互不與 0.5 相交，也與提示框離散度的區間分離——它是 CLIP-VG 上最具資訊量的擾動訊號。其 AURC 在三個 split 上分別為 0.075 / 0.062 / 0.105，相對於隨機基線（等於 base error，0.157 / 0.123 / 0.216）約**降低一半**，顯示用它做選擇性預測能把風險壓到隨機棄答的一半左右。

這個訊號的意義不是泛用信心，而是**指稱身分的穩定度**：當良性的同義改寫就會讓 CLIP-VG 改選另一個框，這個查詢的指稱本身是不穩的，答錯風險高。這正是「與定位相關」（grounding-specific）的不確定性，而非一般分類信心的搬移。

## 5.3 OWL-ViT：分數型訊號的資訊量

OWL-ViT 作為凍結 REC 基礎模型則弱得多（以 argmax 候選為預測時 base accuracy 僅 val 41.94%、testA 46.74%、testB 38.45%）——它本是開放詞彙偵測器，並非為單目標 REC 優化。因此這裡審計的問題是「在一個較弱、但有候選分數的基礎模型上，哪一族分數訊號最能排出答錯」。

**表 5.2　OWL-ViT 各訊號對答對與否的 AUROC（括號為 95% bootstrap CI）**

| 訊號 | val | testA | testB |
|---|---|---|---|
| **score entropy** | **0.680** (0.670–0.690) | **0.691** (0.678–0.704) | **0.680** (0.666–0.695) |
| margin12 | 0.650 (0.640–0.661) | 0.657 (0.643–0.670) | 0.652 (0.637–0.668) |
| cross-prompt consistency | 0.641 (0.630–0.651) | 0.660 (0.644–0.673) | 0.614 (0.596–0.627) |
| top1 score | 0.622 (0.611–0.632) | 0.613 (0.598–0.628) | 0.630 (0.615–0.645) |

**解讀。** 在 OWL-ViT 上，**分數熵**是最具資訊量的訊號（三個 split AUROC 0.68–0.69），優於邊際、跨提示一致性與 top1 絕對分；其信賴區間在三個 split 上與 top1 score 分離。值得注意的是 **top1 絕對分是最弱的訊號**（AUROC 0.61–0.63）——這與先前在偵測器上觀察到的「per-object 絕對分數有不可校正偏差」一致：絕對分數的尺度不可靠，但分數的**分布形狀**（熵）仍攜帶資訊。這也預示了第 10 章的設計選擇：跨基礎模型轉移要用 base-normalized 的訊號，而非絕對分數。

## 5.4 跨基礎模型的對照觀察

把兩個基礎模型並排，浮現一個對全篇重要的結構性觀察：**最具資訊量的訊號家族隨基礎模型而異，但「不確定性可被廉價副產品捕捉」這件事在兩個基礎模型上都成立。**

- CLIP-VG（無分數）：靠擾動訊號（跨提示一致性 AUROC ≈0.72）。
- OWL-ViT（有分數）：靠分數分布訊號（分數熵 AUROC ≈0.68）。

兩者都顯著優於隨機，且都不依賴基礎模型的絕對分數尺度。跨提示一致性在兩個基礎模型上都可算、且都 informative（CLIP-VG 0.72、OWL-ViT 0.64），是少數**跨基礎模型共通**的訊號——這是第 10 章 C4 把它選為轉移主訊號的依據。

## 5.5 小結

C1 確立了地基結論：凍結指稱定位基礎模型的不確定性**確實可由前向傳遞的廉價副產品捕捉**，且訊號是與定位相關的（指稱身分穩定度、分數分布形狀），而非泛用信心的搬移。最具資訊量的訊號家族依基礎模型架構而異（CLIP-VG 用擾動、OWL-ViT 用分數熵），但跨提示一致性在兩個基礎模型上都 informative。這為 C2 的選擇性風險控制、C3 的無目標閘提供了可校準的原料；同時，「絕對分數最弱、分布／擾動訊號較強」這個觀察，預示了跨基礎模型轉移必須避開絕對分數尺度（第 10 章）。

---

# 第 6 章　C2：選擇性風險控制

C1 證明凍結基礎模型的副產品攜帶與定位相關的不確定性訊號。本章把這些訊號**用起來**：以校準後的信心做**選擇性預測**（selective prediction）——回答信心最高的一部分查詢、對其餘棄答——並證明其風險—覆蓋（risk–coverage）曲線優於天真基線，同時以 oracle 線量化尚有多少訊號未被利用。

## 6.1 選擇性預測與評測指標

給定信心分數 `c(x)`，選擇性預測回答 `c(x)` 最高的 coverage 比例查詢、棄答其餘。掃過所有 coverage 值即得**風險—覆蓋曲線**：橫軸覆蓋率、縱軸已作答子集上的錯誤率。其曲線下面積即 **AURC**（area under risk–coverage，越低越好）。

我們以四條線解讀每個訊號：

- **random**：隨機排序棄答，其 AURC 等於 base error（覆蓋率不影響期望風險）。
- **single-signal threshold**：以 C1 審計出的各單訊號排序。
- **oracle**：以真實 correctness 排序（先答對的、最後才答錯的），給出任何信心訊號可達的下界。
- 兩者之間的差距即 **oracle gap**——量化「訊號還差理想多少」。

所有 AURC 附自助法 95% 信賴區間（≥1000 次 resample test，校準器固定）。

## 6.2 CLIP-VG 上的選擇性風險控制

在 CLIP-VG 上，跨提示一致性的選擇性預測把風險壓得相當低（數字承第 5 章表 5.1）：

**表 6.1　CLIP-VG 跨提示一致性的 AURC vs 隨機（括號為 95% CI）**

| Split | AURC（consistency） | random AURC（= base error） | 相對降幅 |
|---|---|---|---|
| val | **0.075** (0.070–0.080) | 0.157 | ≈ 52% |
| testA | **0.062** (0.055–0.070) | 0.123 | ≈ 50% |
| testB | **0.105** (0.096–0.114) | 0.216 | ≈ 51% |

三個 split 上，以跨提示一致性做選擇性預測的 AURC 都約為隨機棄答的**一半**，且信賴區間與隨機基線完全分離。換言之，在不更新 CLIP-VG 任何參數的前提下，僅靠改寫提示下的指稱穩定度，就能把「該優先棄答哪些查詢」排得遠比亂棄好。

## 6.3 OWL-ViT 上的選擇性風險控制

OWL-ViT 是較弱的 REC 基礎模型（base error 0.58–0.62），但選擇性預測同樣有效。分數熵（C1 審計出的最強分數訊號）給出最低 AURC：

**表 6.2　OWL-ViT 各訊號的 AURC vs 隨機（括號為 95% CI）**

| 訊號 | val | testA | testB |
|---|---|---|---|
| **score entropy** | **0.450** (0.436–0.464) | **0.397** (0.380–0.417) | **0.482** (0.463–0.500) |
| margin12 | 0.452 (0.440–0.465) | 0.396 (0.379–0.413) | 0.487 (0.469–0.505) |
| top1 score | 0.481 (0.468–0.494) | 0.431 (0.414–0.449) | 0.512 (0.492–0.531) |
| cross-prompt consistency | 0.494 (0.482–0.508) | 0.433 (0.416–0.451) | 0.549 (0.530–0.569) |
| random（= base error） | 0.581 | 0.533 | 0.616 |

**解讀。** 分數熵與邊際在三個 split 上把 AURC 從隨機的 0.53–0.62 降到 0.40–0.49，相對降幅約 20–26%。降幅小於 CLIP-VG，但這是預期的：OWL-ViT 的 base error 高得多（它本非單目標 REC 模型），可被排序救回的空間結構性地較小。重點是**即使在一個弱基礎模型上，選擇性預測仍穩定優於隨機**，且最強訊號（分數熵）的信賴區間與隨機基線分離。

值得注意 top1 絕對分在三個 split 都是較弱的選擇性訊號（與 C1 的 AUROC 觀察一致），再次印證絕對分數尺度不可靠，而分布形狀（熵）／相對量（邊際）較可信。

## 6.4 Oracle gap

每條風險—覆蓋曲線都附 oracle 線（以真實 correctness 排序）。oracle gap 的意義是雙重的：它既證明**現有訊號尚未榨乾**（gap > 0，仍有改善空間，例如更強的校準器或更多訊號），也界定**任何單一信心訊號的天花板**（oracle 本身），避免過度宣稱。在 CLIP-VG 上 gap 較小（訊號已接近把可救回的查詢排到前面）；在 OWL-ViT 上 gap 較大，呼應其基礎模型本身較弱、不確定性結構較難用單訊號完全捕捉。oracle gap 的存在也是後續章節的伏筆：當點預測的選擇性閾值逼近其天花板，要進一步提升可靠性就須改變動作空間——這正是 M4 邊界與 CRS 集合化的動機。

## 6.5 小結

C2 確立：以 C1 審計出的訊號做選擇性預測，在兩個結構迥異的凍結基礎模型上都使風險—覆蓋曲線優於隨機與弱訊號基線——CLIP-VG 上 AURC 約降至隨機的一半，OWL-ViT 上降約 20–26%。oracle gap 同時量化了未用盡的訊號與單訊號天花板。這證明凍結基礎模型的不確定性不只是「可測量」（C1），更是「可用於風險控制」（C2）。但 oracle gap 與基礎模型本身的高 base error 也提示：單一信心閾值的點預測有其上限，後續章節將以無目標棄答（C3）與有保證的集合預測（CRS）突破之。

---

# 第 7 章　C3：事後無目標閘

C1、C2 在標準 REC 設定下量測並利用了「答對與否」的不確定性。本章轉到**廣義 REC（GREC）**的一個更基本的能力缺口：**當描述在影像中根本沒有對應物件（no-target）時，凍結基礎模型該如何棄答？** 標準基礎模型恆吐 argmax 框，對無目標查詢的準確率**結構性地為零**——它沒有「不答」這個動作。本章證明，一個輕量的事後閘能以基礎模型副產品輸出 P(no-target)，把這個從無到有的棄答能力加到凍結基礎模型上。

## 7.1 任務與基線

gRefCOCO 把無目標顯式納入：一個描述可對應零、一或多個物件。我們在此聚焦無目標子問題——閘需判斷「該不該答」。

**天真基線（forced-output base）**：凍結基礎模型恆輸出 argmax 框。對任何無目標查詢，它必然吐出一個錯框，因此其**無目標準確率 N-acc = 0**，這是結構性的下界，不是調參能改善的。任何非零的無目標處理能力都來自外加的棄答機制。

**事後無目標閘（learned gate）**：以基礎模型副產品為特徵的輕量校準器輸出 P(no-target)，再以在 gRefCOCO val 上選定的閾值 τ 二分為「棄答 / 作答」。閾值與校準器全部在 calib 上決定，test 單次評估（防洩漏協定見第 4 章）。

評測指標：無目標 AUROC（門檻無關的排序品質）、balanced accuracy（在 τ 下對無目標 / 有目標的平衡準確率）、以及基底的無目標比率（no-target rate）。

## 7.2 主結果

**表 7.1　gRefCOCO 事後無目標閘（learned gate）三個 split 結果（括號為 95% bootstrap CI）**

| Split | n_test | no-target rate | τ | no-target AUROC | balanced acc |
|---|---|---|---|---|---|
| val | 6974 | 0.608 | 0.589 | **0.824** (0.814–0.833) | 0.759 (0.749–0.768) |
| testA | 9510 | 0.233 | 0.545 | **0.773** (0.761–0.784) | 0.704 (0.693–0.715) |
| testB | 7973 | 0.294 | 0.591 | **0.741** (0.731–0.754) | 0.689 (0.678–0.700) |

**解讀。** 事後閘在三個 split 上的無目標 AUROC 為 0.741–0.824，全數顯著高於隨機（0.5），信賴區間都不與 0.5 相交。對照 forced-output 基線的 N-acc = 0，這是**從無到有**：凍結基礎模型原本完全沒有棄答能力，輕量閘以其自身副產品就能把無目標排序到 AUROC 0.74 以上。balanced accuracy 0.69–0.76 進一步顯示，在 val 選定的單一閾值下，閘能同時合理地處理無目標與有目標兩類，而非靠偏向一邊作弊。

三個 split 的難度有別：val（無目標比率 0.61）最高（AUROC 0.824），testA / testB（無目標比率 0.23 / 0.29）較低（0.773 / 0.741）。testA 以人物為主、testB 以物件為主，後者的開放詞彙比對較難，無目標判斷也較難，這與 AUROC 由 testA 到 testB 的下滑一致。

## 7.3 文獻定位

無目標 / 廣義 REC 本身不是新設定，已有專門架構處理。把本章的事後閘放進文獻座標，可凸顯其定位（而非宣稱更強）：

- **HieA2G**（AAAI'25, arXiv:2501.01416）以 trained Adaptive Grounding Counter 在 gRefCOCO val/testA/testB 達 N-acc 56–60%。它是**全監督專訓的計數頭**。
- **VIRO**（CVPR'26, arXiv:2601.12781）以 neuro-symbolic program + per-operator verifier 做 verification-aware abstention，報 61.1% balanced accuracy（有目標 + 無目標合併）。
- **True/False Verification**（arXiv:2509.09958）以通用 VLM 對每個 box 答真假，支援棄答。

本章的事後閘與這些工作的**路線**不同，因此不做絕對 accuracy 競賽（基礎模型、評測口徑、訓練成本都不同，直接比較無意義）。本章的定位是成本—reliability 光譜的**輕量一端**：不重訓任何基礎模型、近乎零訓練成本，就能把棄答能力從零加到凍結基礎模型上，並達到 AUROC 0.74–0.82。它與 HieA2G / VIRO 的關係是「在多低的成本下能吃到多少棄答 benefit」的對照（完整成本—風險 Pareto 見第 11 章），而非取代。

## 7.4 從棄答到集合：通往 M4 與 CRS

C3 證明事後閘能加上「該不該答」的二元能力。但廣義 REC 的完整要求不止於此——它要求輸出**正確的框集合**（基數 0／1／多）。把事後閘推到完整 GREC 的 exact-match 評估，會撞上一道原理性的牆：多目標的計數與集合預測超出單一信心閾值的動作空間。這正是第 8 章（M4 邊界分析）的主題，並直接引出第 9 章 CRS 的集合化解法——其中無目標棄答（本章的能力）會以 R2（無目標誤選率）與 R3（目標棄答率）兩個受控風險的形式，被納入聯合校準的風險控制框架。

## 7.5 小結

C3 確立：在 gRefCOCO 上，一個以凍結基礎模型副產品為輸入的輕量事後閘，能把 forced-output 基線結構性為零的棄答能力提升到無目標 AUROC 0.741–0.824（三個 split）、balanced accuracy 0.69–0.76。這是基礎模型原本不具備的能力，且以近乎零訓練成本達成。本章亦以 HieA2G / VIRO / True-False Verification 把此能力定位在成本—reliability 光譜的輕量端，而非與其比絕對準確率。棄答能力的確立，為 CRS 把「該不該答」納入風險受控集合鋪好了路。

---

# 第 8 章　M4：完整 GREC 的 exact-match 之牆

本章把事後點預測策略推到完整 GREC，定位它在何處、為何撞牆。這**不是方法貢獻，而是一個邊界／壓力測試**——但它在全篇敘事中是**轉折點**：exact-match 之牆證明點預測不足，正是下一章 CRS（風險受控集合預測）的存在理由。同一批數據的敘事方向因此不是「我們做不到完整 GREC」，而是「完整 GREC exact-match 在凍結基礎模型上有一道可量化的牆，這道牆正是 CRS 的動機」。

## 8.1 動機與設定

C3 已證明凍結基礎模型的輸出訊號可被事後校準出**無目標棄答**能力（基礎模型本身結構上沒有此能力）。一個自然的壓力測試是：把同一套事後策略推到**完整 GREC**——不只判斷「該不該答」，還要在 target-present 時輸出**正確的框集合**（multi-target），並以官方 **Pr@(F1=1, IoU≥0.5) / N-acc / T-acc** 評測，與 trained 架構（HieA2G、VIRO）在同一指標座標上定位。

**關鍵資料事實**：gRefCOCO 的 target-present 樣本**幾乎全是 multi-target**（val：1499 no-target／0 single／2738 multi，多數 2 框）。因此完整 GREC 在此資料集上本質是**集合預測／計數**問題，而非單框定位。

**GREC 官方 Pr@(F1=1) 協定**（忠實移植自 gRefCOCO `mdetr/datasets/refexp.py`）：以信心閾值 τ 過濾預測框 → 與 GT 框做貪婪 IoU 匹配（非 Hungarian，IoU≥0.5）→ 計 TP/FP/FN → `F1 = 2TP/(2TP+FP+FN)`；一個樣本須 **F1=1**（所有 GT 命中且無多餘框）才算正確；no-target 樣本須過濾後 **0 框**才算正確。OWL-ViT 原始候選分數偏低（multi-target top1 中位數 0.13），官方預設 τ=0.7 會濾掉所有框，故 τ 在 calib（ref_id 奇偶切分）重新標定，test 單次評估。

## 8.2 策略階梯與結果

格式為 **Pr@(F1=1) / N-acc / T-acc**（test split，95% bootstrap CI 見 §8.3）。

**表 8.1　完整 GREC 策略階梯（三個 split）**

| 策略 | val | testA | testB |
|---|---|---|---|
| forced-output base（下界，永不棄答） | 0.164 / 0.267 / 0.003 | 0.043 / 0.177 / 0.002 | 0.085 / 0.279 / 0.004 |
| conf-threshold（單一全域 τ） | 0.607 / 0.996 / 0.003 | 0.258 / 0.939 / 0.052 | 0.301 / 0.938 / 0.035 |
| **P(no-target)+conf（本研究事後策略）** | 0.607 / 0.996 / 0.003 | 0.258 / 0.939 / 0.052 | 0.301 / 0.939 / 0.035 |
| oracle（GT 完美棄答） | 0.623 / 1.0 / 0.037 | 0.278 / 1.0 / 0.060 | 0.329 / 1.0 / 0.050 |
| per-sample oracle τ（天花板） | 0.682 / 1.0 / **0.189** | 0.409 / 1.0 / **0.230** | 0.460 / 1.0 / **0.235** |

對照線（trained，**基礎模型不同，僅供座標定位，不主張可直接比較**）：HieA2G 完整 GREC Pr@(F1=1) val/testA/testB = 67.8 / 66.0 / 56.5（ResNet101 全監督，含 gRefCOCO 標籤訓練 counting head）。

**三點觀察：**

1. **Pr@(F1=1) 的提升幾乎全部來自棄答，而非多目標命中。** forced-output → 策略的提升（val 3.7×、testA 6×、testB 3.5×）對應的是 N-acc 從 0.18–0.28 升到 0.94–0.996，而 T-acc 幾乎不動。事後策略在「該不該答」維度有效，在「答幾個框」維度無效。

2. **「本研究 ≈ conf-threshold」——誠實承認，不挑數字。** 三個 split 的兩階段 P(no-target)+conf 策略與單一全域 conf-threshold 幾乎沒有差距（testA/testB 僅 N-acc 第三位小數差異）。原因：gRefCOCO 無目標比率高，加上 Pr@(F1=1) 這個 exact-match 指標，使得全域信心閾值已吸收絕大部分可得的 gain；learned no-target gate 在此指標下沒有額外拉開差距。這說明完整 GREC 的主要瓶頸**不在無目標偵測，而在 target-present 的多目標集合預測**。

3. **T-acc 天花板極低且非雜訊。** 即使給每個樣本完美的 per-sample τ（per-sample oracle，等同擁有 AUROC=1.0 的完美計數訊號），T-acc 天花板也只有 0.19–0.24。亦即：在這個凍結基礎模型上，多目標 exact-match 的失敗**不是因為缺乏好的信心訊號**，而是動作空間受限——「單一信心閾值」這個動作無法表達「該圖留 2 框、那圖留 5 框」的 per-sample 計數。

## 8.3 為什麼失效：召回足夠，瓶頸在 FP

對 val target-present 樣本的診斷：

- **GT-coverage recall = 0.94**：九成樣本中，每個 GT 框都有某個預測框 IoU≥0.5 命中。**召回不是瓶頸。**
- **best achievable F1（per-sample 最佳 τ）平均僅 0.62，僅 18.8% 樣本能達 F1=1**：問題出在 **FP（多餘框）**——OWL-ViT 輸出一堆候選框，單一全域 τ 無法在每個樣本上恰好留下「正確數量」的框。

因此完整 GREC exact-match 本質是**計數／集合預測**問題：要破此須 per-sample 的框數決策（如 HieA2G 的 trained Adaptive Grounding Counter），而這已超出「凍結基礎模型 + 單一輕量事後校準器」的範圍。

**Bootstrap 95% CI**（leak-safe：calib 固定、只 resample test ≥1000 次）佐證此邊界主張為統計穩固，非抽樣雜訊：

**表 8.2　T-acc 的 bootstrap CI**

| split | T-acc point | 95% CI |
|---|---|---|
| val | 0.003 | [0.001, 0.006] |
| testA | 0.052 | [0.046, 0.057] |
| testB | 0.035 | [0.031, 0.040] |

T-acc 的 CI 全部貼近 0（最高上界僅 0.057），「凍結零樣本偵測器的多目標 exact-match 超出事後校準能力」是一個有信賴區間支撐的結論。

## 8.4 結論：通往 CRS 的跳板

> **事後可靠性校準可以補出無目標棄答，但不能取代 trained counting head 或集合預測模組；完整 GREC exact-match 暴露了凍結基礎模型事後點預測的動作空間限制。**

這道牆不是論文的終點，而是轉折點。它劃出兩條路：(a) 補一個 trained counting／集合預測模組（HieA2G 路線，但這放棄了「凍結基礎模型 + 輕量事後」的賣點）；(b) **不再強迫 exact single/multi-box 點預測，改輸出有分布無關保證的框集合**——這正是第 9 章 CRS 的解法。

M4 的兩個診斷數字直接成為 CRS 的設計依據：

1. **per-sample oracle τ 的 T-acc 天花板僅 0.19–0.24** → 「單一信心閾值」這個動作表達不了 per-sample 計數。CRS 的回應：放棄點估計，輸出集合，把「該留幾個框」的不確定性吸收進**集合大小**與**棄答**（CRS 的 R3 目標棄答率）。
2. **recall 0.94 足夠、瓶頸在 FP** → raw score 把 TP/FP 在分數軸交織，點閾值切不乾淨。CRS 的回應：用 LTT 對「已作答目標漏檢率」給有限樣本保證，而非追逐 F1=1 的 0/1 事件。

因此 M4 與 CRS 是**同一條 framing 的兩端**：M4 證明凍結基礎模型的點預測撞牆（量化「缺的是動作空間，不是訊號」）；CRS 證明換成風險受控集合預測後，同一批凍結基礎模型能在三風險保證下輸出接近真實基數的緊緻集合。

需明寫的誠實邊界：M4 的結論本身不被 CRS 推翻——完整 GREC 的 **exact-match（F1=1）** 仍超出凍結事後能力。CRS 不宣稱「解了完整 GREC」，而是**改變問題**——從 exact-match 點預測換成風險受控集合建構。這個 metric pivot 在兩章都寫白，避免被讀成「CRS 解了 M4 解不了的同一個問題」。

對照定位（trained ↔ post-hoc 光譜）：HieA2G（全監督專訓 counting head）／ VIRO（凍結基礎模型 + 重型 neuro-symbolic per-operator verifier）／ 本研究（凍結基礎模型 + 單一輕量事後校準器）。三者在「訓練成本 ↓、推論成本 ↓」光譜上，本研究佔最輕量端；M4 誠實標明在此端，多目標 exact-match 是能力邊界，而非可由校準跨越的目標。

---

# 第 9 章　CRS：跨基礎模型保形組合指稱集合（主方法與主結果）

本章是全篇的台柱。C1–C4 回答「凍結基礎模型的不確定性訊號**有多 informative**」，M4 證明點預測在完整 GREC exact-match 撞牆。CRS 把貢獻從「**量測**可靠度」升級為「**建構有分布無關保證的可靠集合**」：讓信心策略 `π` 輸出一個基數可為 0／1／多的**風險受控指稱集合**，以 Learn-then-Test 聯合校準三個有界風險並給出有限樣本保證，再以跨基礎模型組合把風險控制的代價因式分解為兩個正交瓶頸。

## 9.1 動機：從「測量」到「保證」

CRS 的轉折來自 M4 的診斷數字本身。M4 顯示多目標 exact-match 的 per-sample oracle 天花板僅 0.19–0.24，召回 recall 0.94、瓶頸是 FP 多餘框、單一全域 τ 無法每樣本恰好選對框數。

- **舊解讀**：事後校準做不到（死路）。
- **新解讀**：raw score 把 TP/FP 框在分數軸上交織，任何**點估計**閾值都切不乾淨——但我們不需要點估計，可以輸出一個**有分布無關保證的集合**。問題於是從「能不能精確命中」變成「**能不能在凍結基礎模型上給出有有限樣本保證的框集合，並刻畫保證的可行邊界**」。

這一步把題目從「我們做不到 SOTA exact-match」改寫成「我們給 distribution-free 保證 + 可行域」。審查打不掉「保證」，也打不掉「我們證明了什麼可達、什麼不可達」——這是 measurement 給不了的護城河。

## 9.2 問題形式化：風險受控指稱集合

凍結基礎模型 `g` 對查詢 `(I, e)` 吐出候選框集合 `{(b_i, s_i)}`（OWL-ViT／GroundingDINO 原生有；CLIP-VG 無候選，不適用本章，留作轉移討論）。一個**指稱集合策略** `π_λ` 以閾值／參數 `λ` 從候選中選出集合 `S_λ(I,e) ⊆ {b_i}`，基數可為 0（棄答／無目標）、1 或多。

我們控制的不是準度，而是**風險**。對 target-present 樣本定義**漏檢風險（FNR）**：

```
L_FNR(S, G) = 1 − |{g ∈ G : ∃ b ∈ S, IoU(b,g) ≥ 0.5}| / |G|
```

（G = GT 框集合）。對 no-target 樣本定義**誤選風險** `L_NT(S) = 1[|S| > 0]`。對 target-present 樣本另定義**棄答風險** `L_DEF(S) = 1[|S| = 0]`（真有 target 卻吐空集）。

**目標**：在校準 split 上選 `λ̂`，使得在 unseen test 上同時控制三個有界風險：

- **(R1) 已作答目標漏檢率（answered-target FNR）**：在**未棄答**的 target-present 樣本上 `E[L_FNR] ≤ α`；
- **(R2) 無目標誤選率（no-target false selection）**：no-target 上 `E[L_NT] ≤ β`；
- **(R3) 目標棄答率（target deferral）**：target-present 上 `E[L_DEF] ≤ γ`。

三者皆 distribution-free、finite-sample。α、β、γ 是使用者旋鈕（safety budget）。

**誠實命名（必須寫白）**：R1 是**條件**風險——只在系統選擇回答的 target-present 樣本上計算 FNR，因此**不可**稱為 target recall／coverage guarantee；正式名稱一律 **answered-target FNR**（完整：未棄答 target-present 查詢上的條件 target-set FNR）。R3 的存在正是為了防守「你只是把難的 target 棄答掉，所以 R1 好看」——棄答是被明碼控制的第三風險，不是藏起來的成本。R2 與 R3 語義不同：no-target 吐空集是**正確**決策，target-present 吐空集才是**代價**，兩者分開計、分開控。

**指標精確定義（必寫在 caption）**：

- **answered TP set size**（主表的 "set size"）：平均集合大小只在**已作答 target-present 查詢**（target-present 且最終非空輸出）上計算，不是所有查詢的平均。
- **R3 target deferral**：target-present 最終輸出空集的比例（gate 棄答**或** box 門檻濾光皆計入）。
- **R1 answered-target FNR**：僅在 target-present 且最終非空的查詢上計算。

**校準協定澄清（避免誤解為標準 benchmark held-out test）**：本研究對每個官方 split（val/testA/testB）**內部**以 `ref_id` parity 切成 calibration / evaluation 兩半；閾值網格、LTT 檢定、操作點選擇**只用 calibration 半**，回報的風險與 bootstrap CI 在 held-out parity 半上計算。這是 split 內的風險控制評估，而非標準 train-val-test；跨 split transfer 若報，僅為經驗穩定度，不在 distribution-free LTT 保證範圍內。

這正是 GREC 官方 Pr@(F1=1) 的「機率化、有保證」版本：F1=1 要求零漏檢且零多選，我們不追逐那個 0/1 事件，而是給「answered-target FNR ≤ α」的連續、有保證旋鈕。

## 9.3 單風險 CRS：Conformal Risk Control 與不可化約下限

先只控 R1。`L_FNR` 對「選更多框」單調非增 ⇒ 對閾值 `λ` 單調非減。用 **Conformal Risk Control**（Angelopoulos et al. 2023）在 calib 的 n 個 target-present 樣本上選最大的 λ（最小集合）使

```
(n · R̂_FNR(λ) + B) / (n + 1) ≤ α,    B = 1（loss 上界）
```

**實證**（OWL-ViT gRefCOCO 三個 split，ref_id 奇偶分 calib/test）：nominal α 與 empirical FNR 近乎完美對角線追蹤（val：α 0.2→0.189 / 0.3→0.279 / 0.4→0.374 / 0.5→0.471；testA/B 同）。唯一例外在 α=0.05（testB 連 0.10）——並非 conformal 失效，而是**候選池物理召回上限**：全選候選後殘餘 FNR 仍有 val 0.06 / testA 0.10 / testB 0.14（即 GT-coverage recall ≈ 0.94 的硬上限）。

conformal 框架**精確暴露**了凍結基礎模型的不可化約下限：任何閾值都打不破候選池沒召回到的 GT。這是一個可量測、可報告、distribution-free 的「基礎模型能力上限」——CRS 不只給保證，還診斷出保證在哪裡撞到基礎模型的物理極限。

## 9.4 多風險 CRS：耦合風險與 Learn-then-Test

同時控 R1 + R2 + R3 是本章的技術核心，也是與既有 conformal-OD 文獻（只控單一 box coverage）的關鍵區隔。

**為何單一全域閾值不行**：R1 要「選多」、R2 要「選少」，方向相反；更糟的是兩者透過**重疊的分數分布**耦合——OWL-ViT zero-shot 弱，target 與 no-target 樣本的 top-score 分布高度重疊。

**天真序貫校準失敗（誠實記錄）**：先用 no-target calib 標一個棄答 gate `τ_A`（壓 R2），再在通過 gate 的 target 樣本上標 λ（壓 R1）。實證 R2 守住但 R1 爆到 0.63–0.76——因為 `τ_A` 為壓 no-target 誤選被推高，連帶擋掉大量真 target（它們分數也低），被擋的真 target 吐空集 ⇒ FNR=1。**這個失敗是發現**：兩風險不是獨立可加，序貫校準破壞聯合有效性。

**正解 = Learn-then-Test（LTT, Angelopoulos et al. 2021）**：把 `(τ_A, λ)` 二維網格的每一格當成一個假設，對每個 risk 做 finite-sample 檢定（Hoeffding bound 的 p-value），用 Bonferroni 控制 family-wise error ≤ δ：

```
p_R(config) = exp(−2 n_R (target_R − R̂)²)   若 R̂ < target_R 否則 1
config 有效 ⟺ p_R1 ≤ δ/(3|grid|) 且 p_R2 ≤ δ/(3|grid|) 且 p_R3 ≤ δ/(3|grid|)
```

只保留三 risk 都通過的 config ⇒ **聯合、distribution-free 保證**。Bonferroni 分母是 **3·|grid|**（三風險 × 整個 config 網格），FWER ≤ δ 涵蓋整個搜尋空間。所有通過的 config 構成**可行域（feasible region）**；在可行域內挑「**最小 calibration 集合大小**」的點作為操作點。此 selection 仍 post-hoc 合法：因為 FWER 已涵蓋整個 grid，可行域內任意挑點都不破壞保證（這正是 LTT 的設計目的，直接回答「min-size 選點是否挑數字」的質疑）。

**洩漏防守（必須寫白）**：閾值網格的 quantile **只用 calibration covariates** 建構；evaluation 樣本的 covariates 與 labels 從不參與網格建構、風險檢定或操作點選擇。

**實證**（單一 base OWL-ViT，δ=0.1，30×30 calib-only grid，三風險）：三個 split 聯合保證**全部成立**，**但操作點退化**：純 OWL-ViT 即使取可行域最小集合，仍需約 8–9 框（val 8.23 / testA 9.25 / testB 7.12），GroundingDINO 單 base 則因 gate 弱、需靠高棄答才守住 R2/R3。這個退化正是下一節 cross-base composition 的動機。

在弱零樣本單 base 上，要同時拿 distribution-free 的三風險保證，代價是大集合或高棄答。這**不是方法失敗**，是一個嚴格的數學事實：保證的代價由基礎模型的分數可分性決定。CRS 把這個代價**明碼標出來**（三風險 + 集合大小），而非藏在平均準度後面。這比 M4 的經驗觀察強一個量級——它是有限樣本、有保證、可畫成可行域圖的根本權衡刻畫。

## 9.5 推可行域：nonconformity 設計與更強基礎模型

可行域退化的根因是 nonconformity 太弱（raw score）。兩個正交槓桿把可行域往**有用方向**（小集合、低棄答）推：

**(a) Per-box consistency belief（方法內元件）**：以改寫下的 per-box 出現穩定度重定義 nonconformity：`belief(b) = s(b) · (1 + agreement(b))`，agreement = 該 box 在改寫下被高分框呼應的程度。實證（OWL-ViT gRefCOCO val）：同保證下集合縮小 α 0.1→1.7% / 0.2→3.7% / 0.3→6.2% / 0.4→10.0%，validity 不破。誠實標註：高召回區增益偏弱，當「nonconformity 設計」一節如實報，非台柱。

**(b) 更強凍結基礎模型（GroundingDINO）**：在新 framing 下，基礎模型**不是被比較的對象**，而是「**框架隨基礎模型能力擴張可行域**」的一個軸。同一套 LTT 保證程序，基礎模型的分數可分性越好，可行域裡越早出現有用操作點（低棄答 + 小集合）。

**實證診斷（決定性）**：對 target-present 子問題，達到同一 FNR≤α 保證所需集合大小，GroundingDINO 比 OWL-ViT 小約 **5×**（val，α=0.2：3.3 vs 18.6 框；α=0.3：2.4 vs 12.1），且**不可化約召回下限 0.007 vs 0.062**（val；GroundingDINO 候選池幾乎涵蓋所有 GT；testA/testB floor：GD 0.012/0.017 vs OWL 0.104/0.142）。target 平均約 2 GT 框，GroundingDINO 在 α=0.3 只需 2.4 框 ≈「幾乎恰好選對」。

## 9.6 核心方法：Cross-Base Conformal Composition

**問題的乾淨因式分解**（本身是貢獻）：CRS 的保證代價可分解為兩個**正交**的瓶頸，分別由不同的基礎模型能力決定，且**沒有單一基礎模型兩者都強**：

**表 9.1　兩個瓶頸與兩個基礎模型的互補性**

| | 無目標可分性（棄答 gate） | target 集合可分性（set selection） |
|---|---|---|
| OWL-ViT | **0.82**（好） | 差（α=0.3 需 12 框） |
| GroundingDINO | 0.60（差，對任何 query 都給高信心框） | **強（α=0.3 需 2.4 框），召回下限 0.007** |

OWL-ViT 擅長判斷「**該不該答**」，GroundingDINO 擅長「**答得準**」。單一基礎模型的多風險 LTT 因此必然退化：OWL-ViT 集合爆炸、GroundingDINO 被弱 gate 拖累（高棄答）。

**解法 = 兩個凍結基礎模型各司其職，用 LTT 聯合校準三風險**：

- **棄答 gate** ← OWL-ViT 分數 `top1_score`（無目標可分性 0.82）；
- **set selection** ← GroundingDINO 候選 + 分數（召回下限 0.007、集合小）；
- 兩基礎模型皆 frozen、皆不訓練；join key =`(ref_id, sent_id)`（共享 gRefCOCO 標註，expression／no_target 零 mismatch）；
- LTT 在 `(τ on OWL 分數, λ on GDINO 分數)` 二維 calib-only grid 上聯合校準 `(R1≤α, R2≤β, R3≤γ)`。

**主結果（三個 split 全量，三風險 α=β=0.3 / γ=0.5 聯合保證，calib-only grid，min-size 選點）**：

**表 9.2　Cross-Base Composition 主結果**

| split | set size（answered TP） | R1 | R2 | defer(R3) |
|---|---:|---:|---:|---:|
| val | **3.24** | 0.191 | 0.159 | 0.424 |
| testA | **2.02** | 0.261 | 0.203 | 0.443 |
| testB | **3.50** | 0.168 | 0.236 | 0.413 |

三風險全部守住。其中 testA 的 **2.02 框 ≈ GT 平均基數**，幾乎是「恰好選對數量」；testB 也在三保證下維持 <3.6 框。同一 protocol 下，純 OWL-ViT 需 8.23 / 9.25 / 7.12 框，composition 的集合 CI 與純 OWL-ViT 完全分離。公平比較下的 set-size reduction 約 **2.0×–4.5×**。

這把「沒有單一基礎模型兩者都強」的**限制**，轉成「組合兩個凍結基礎模型互補強項」的**正面方法**：純事後、不訓練、接回 C4 cross-base 主軸、非 detector 比較（是互補組合）。selection rule = 在三風險可行域內取 min calibration set size，post-hoc 合法（FWER Bonferroni 已涵蓋整個 grid）。

## 9.7 機制證實：2×2 gate×box 消融

對調 gate-base 與 box-base 的四種組合（三風險 α=β=0.3 / γ=0.5，calib-only grid，min-size 選點）：

**表 9.3　2×2 gate×box 消融**

| split | gate | box | 集合 | R1 | R2 | defer(R3) | feasible? |
|---|---|---|---:|---:|---:|---:|:--:|
| val | OWL | OWL | 8.23 | 0.239 | 0.183 | 0.366 | ✓ |
| val | GD | GD | — | — | — | — | **EMPTY** |
| val | **OWL** | **GD（COMPOSE）** | **3.24** | 0.191 | 0.159 | 0.424 | ✓ |
| val | GD | OWL（reverse） | 15.27 | 0.211 | 0.251 | 0.429 | ✓(n=2) |
| testA | OWL | OWL | 9.25 | 0.255 | 0.207 | 0.443 | ✓ |
| testA | GD | GD | — | — | — | — | **EMPTY** |
| testA | **OWL** | **GD（COMPOSE）** | **2.02** | 0.261 | 0.203 | 0.443 | ✓ |
| testA | GD | OWL（reverse） | — | — | — | — | **EMPTY** |
| testB | OWL | OWL | 7.12 | 0.234 | 0.236 | 0.413 | ✓ |
| testB | GD | GD | — | — | — | — | **EMPTY** |
| testB | **OWL** | **GD（COMPOSE）** | **3.50** | 0.168 | 0.236 | 0.413 | ✓ |
| testB | GD | OWL（reverse） | — | — | — | — | **EMPTY** |

一旦把目標棄答（R3）也納入受控風險，**pure GD（GD gate + GD box）三個 split 全部可行域變空**——GD gate 守無目標必須靠極高棄答，一旦棄答受 γ=0.5 約束就無解；reverse（GD gate + OWL box）也幾乎全空（val 僅 n=2、集合爆 15）。在三風險下，**只有 OWL gate + GD box 這條對角線同時 feasible 且 compact**。這正面回答「CRS 只是 GroundingDINO 比較強」的質疑：GD 單獨無解，是 **factorization** 才進得了可行域。factorization 主張因此嚴格成立：**OWL gate 控可用棄答／無目標；GD box 控 target 集合緊緻度**。

需限定範圍：2×2 消融在**主 split-specific LTT 協議**下證明 factorization；額外的 stricter validity check（§9.8）只施加在最終 COMPOSE 操作點，未對整個 2×2 重跑。

## 9.8 穩健性與有效性

**統計穩固性（bootstrap CI，config 固定只 resample test）**：三個 split 全量 shared keys = 14229 / 19200 / 16063。

**表 9.4　COMPOSE vs OWL-only 的 bootstrap CI**

| split | OWL-ViT only（answered TP size） | COMPOSE（answered TP size） | COMPOSE R1 | COMPOSE R2 | COMPOSE defer |
|---|---:|---:|---:|---:|---:|
| val | 8.23 [7.85, 8.57] | **3.24 [3.15, 3.36]** | 0.191 [0.179, 0.203] | 0.159 [0.146, 0.171] | 0.424 [0.405, 0.443] |
| testA | 9.25 [9.00, 9.51] | **2.02 [1.98, 2.06]** | 0.260 [0.250, 0.269] | 0.203 [0.187, 0.221] | 0.443 [0.433, 0.454] |
| testB | 7.12 [6.88, 7.35] | **3.50 [3.38, 3.63]** | 0.168 [0.159, 0.177] | 0.236 [0.219, 0.252] | 0.414 [0.401, 0.426] |

三個 split 的 R1/R2 CI 上界都低於 0.3、defer 都低於 γ=0.5；COMPOSE 與純 OWL-ViT 的 size CI 完全分離。結果不是 partial dump 或小樣本僥倖：全量 val/testA/testB 皆在三保證下達到接近 GT 基數的緊緻指稱集合。

需分清：LTT p-value + Bonferroni = 有限樣本風險控制檢定（這是「保證」）；bootstrap CI = 固定 selected config 後對 test 重抽的經驗穩定度（這**不是**保證）。兩者分開報、不混用。

**split-protocol 穩健性（堵 exchangeability）**：COMPOSE 在三種 calib/eval 切分下幾乎不動——parity（主）、5× random ref_id split、image-disjoint（同一影像不跨 calib/eval，最嚴格）：

**表 9.5　三種切分模式的 set size**

| split | parity（主） | random[5] 平均 | image-disjoint |
|---|---|---|---|
| val | 3.25 | 3.24 ± 0.04 | 3.13 |
| testA | 2.02 | 2.05 ± 0.03 | 2.06 |
| testB | 3.51 | 3.53 ± 0.11 | 3.39 |

set size 三模式幾乎重合（random std 僅 0.03–0.11），image-disjoint 亦 feasible 且 compact。誠實標註：random split 的 R2/defer 單次 variance 較大（val R2 偶到 0.26、testB 到 0.29），屬 LTT marginal 保證的正常表現——單次抽樣可略超名目，報 mean±std 並說明即可。

**formal-validity 穩健性**：針對 LTT 保證的三個深層形式問題各做一個對照協議。

**表 9.6　formal-validity 對照（COMPOSE，α=β=0.3 / γ=0.5）**

| 協議 | val | testA | testB | 結論 |
|---|---|---|---|---|
| baseline（parity, Hoeffding R1） | 3.24 | 2.02 | 3.50 | — |
| (C) ratio-free R1（fixed-n 條件風險檢定） | 3.25 | 2.02 | 3.51 | 幾乎完全重現 |
| (A) three-way split（grid/calib/eval image-disjoint 三分） | 3.36 | 3.31 | 3.35 | 三 split 全 feasible |
| (B) image-cluster（image=calib unit，最嚴格） | 3.24 (n=28) | EMPTY | EMPTY | 見下方誠實邊界 |

- **(C) ratio-free R1**：R1 是條件風險，原 Hoeffding 用 answered subset 的 random denominator。改用 `E[L|A]≤α ⟺ E[A(L−α)]≤0` 對 bounded fixed-n 變數 `Z=A(L−α)∈[−α,1−α]` 做 Hoeffding，分母固定為 target-present 全數。三個 split 與 baseline 幾乎完全一致 → random-denominator 質疑實務上不影響結論。
- **(A) data-dependent grid**：grid 從 calib scores 建、又在同 calib 上做 LTT，理論上可質疑。three-way split（grid-design / calib / eval 三個 image-disjoint 子集）下三 split 仍 feasible，set size 3.31–3.36（略升因 eval 全新 image + 樣本變少），結論不變。
- **combined protocol**：把最嚴格的兩個協議疊加（three-way + ratio-free R1）——val sz=3.36 仍 feasible（n_feas=112），且 three-way 的 5 seeds robustness：val 5/5 feasible，sz=3.29±0.11，非 seed=0 僥倖。

**candidate-pool 截斷（protocol 透明化）**：CRS 的候選池是基礎模型經 `PRED_KEEP=50` + `PRED_MIN_SCORE=0.01` 預處理後的 stored pool。下界檢查顯示：雖然 val 100% 樣本原始 `n_cands>50`，但通過 score≥0.01 門檻的候選平均僅 45.9 個（<50），代表 `KEEP=50` 的 cap 幾乎未實際裁切，主導的是 0.01 門檻；最終 set size 由 LTT 選出的 λ（遠高於 0.01）決定，非 dump cap。故「compact set 靠 cap 作弊」不成立。完整 `KEEP∈{20,50,100}` sweep 需重跑 GPU dump，列為 future robustness。

## 9.9 兩項誠實限制

**1. image-cluster 評估的樣本量限制。** 主協議以 **expression** 為 calibration unit；同影像多 expression 是 record-level exchangeability 假設。改以 **image** 為 unit（最嚴格）時，僅 val 維持 feasible（n=28，勉強），testA/testB EMPTY（見表 9.6）。這**不是 validity 崩塌**，而是樣本量限制：image-as-unit 把有效樣本數從約 9000 records 降到約 3000 images，疊加三風險 × Bonferroni（3·|grid|）的嚴格門檻後，有限校準集不足以同時認證三個保證。這是凍結零樣本基礎模型 + 三風險 + 有限校準的根本張力，誠實標為 limitation；主結果用 expression-level，並以 image-disjoint **split** 穩健性（表 9.5）作為 exchangeability 的經驗緩解。

**2. 長 expression 的 gate 退化。** R2（無目標誤選率）隨 expression **長度顯著惡化**：0–8 token（佔約 70%，gRefCOCO 主體）R2=0.13–0.21 守住，9–16 token 升到 0.40–0.61，17+ token 更高。關鍵：9–16 token 段的 OWL tokenizer 截斷率仍是 0%，R2 卻已惡化 → 這**不純是** OWL `max_length=16` 截斷造成，而是 OWL gate 對長 expression 的無目標判斷本就較弱（長句 top1_score 訊號不可靠）；17+ token 的 100% 截斷讓它雪上加霜。aggregate headline 由短句主導故站得住，但長 expression 的 gate 退化是真 limitation。更強／共享的文字編碼器是 future work。

## 9.10 貢獻定位

**表 9.7　CRS 與既有文獻的區隔**

| 既有工作 | 與 CRS 的區隔 |
|---|---|
| Conformal Object Detection（box coverage、FNR） | 封閉類別、單 box-coverage；無語言條件、無無目標、無 cardinality。CRS 控的是 referring set 的耦合三風險 |
| Conformal for Zero-Shot VLM（CVPR'25） | 只做**分類** label set；非 box set |
| GREC / HieA2G / InstanceVG（multi-target） | **全 trained**（count head / 階層 align）；CRS 是 post-hoc、distribution-free、不訓練 |
| VLM selective prediction（ReCoVERR） | 單答案 answer/abstain；無集合層保證、無 cardinality |
| True-False Verification（2509.09958） | 單答案對錯 verify；CRS 是集合層聯合保證 + 計數 |

**貢獻定位三句話**（禁用「first／第一個」）：
1. 對凍結指稱定位做 **risk-controlled box-set selection**：不追 exact-match，而是輸出 calibrated box set，並用 LTT 對三個有界風險（answered-target FNR、no-target false selection、target deferral）做有限樣本控制；
2. 把風險控制的**代價因式分解**為 gate 與 box 兩個正交瓶頸，並以可行域刻畫；
3. 用 **cross-base conformal composition** 組合兩個異質凍結基礎模型的互補強項（一個管棄答、一個管選框），在三保證下達到接近 GT 基數的精準集合——2×2 消融證明這是 factorization 而非單純 detector 比較（單 base 在三風險下退化或無解）。

## 9.11 小結

CRS 把全篇從「測量凍結基礎模型的不確定性」推進到「在凍結基礎模型上建構有分布無關保證的指稱集合」。它以 LTT 聯合校準三個有界風險、把代價因式分解為 gate 與 box 兩個正交瓶頸、並以跨基礎模型組合在三保證下達到接近真實基數的緊緻集合（val/testA/testB 為 3.24 / 2.02 / 3.50 框，集合大小信賴區間與純 OWL-ViT 完全分離）。2×2 消融、bootstrap CI、三種切分模式與 formal-validity 對照共同支撐結論的穩固性，兩項限制（image-cluster 樣本量、長 expression gate 退化）誠實標註。這是本論文「強框架 + 亮眼正面操作點」的最終解，也是 M4 撞牆之後的回答：不做點估計，改輸出有保證的集合。

---

# 第 9b 章　Decomp CRS：候選池維度（第二主結果）

> 本章是三正交維度中的 **candidate pool** 軸。CRS（第 9 章）以 frozen base 的原生候選池為輸入；
> 本章證明：在**完全不碰 detector、不訓練**的前提下，把候選池從「full-expression pool」
> 換成「decomposition-union pool」，可在同一 LTT 協議下**純賺 R1（多目標漏檢）**而不付代價。

## 9b.1 動機：候選池是一個獨立可改善的維度

第 9 章的 CRS 把風險控制代價因式分解為 gate（該不該答）與 box（答得準）。但在 box 軸之前，
還有一個更上游的瓶頸：**候選池本身**。GroundingDINO 對一條完整 referring expression 跑一次前向，
產生的候選框集合，未必涵蓋 multi-target query 的所有 GT——尤其當 expression 描述多個語意子部件時
（「the man in red **and** the woman beside him」），單次整句 forward 容易偏重其中一個子部件。

這定義了第二個正交維度：**在 gate 與 box 都不變的情況下，能不能用更好的候選池降低 R1（已作答目標漏檢率）？**
關鍵約束是不能違反 frozen——任何改善都必須來自 frozen detector 已能產出的資訊，不更新任何權重。

## 9b.2 方法：VLM-routed decomposition union pool

**流程**（training-free，weight-frozen）：

1. 對 target-present query，用一個 frozen VLM（Qwen2.5-VL-7B）作為 **router**，判斷 expression 是否含多個可拆解的指稱子部件；若是，輸出子部件清單。
2. 每個子部件各自餵 GroundingDINO 跑一次前向，得到 per-part 候選框。
3. 把整句 forward 的候選池與各子部件的候選池取 **union**，得到 decomposition-union pool。
4. **其餘完全沿用第 9 章 CRS 協議**：同樣的 OWL gate、同樣的 GDINO box scoring、同樣的三風險 LTT、同樣的 calib-only grid 與 min-size 選點。**只換候選池這一個變因**。

設計重點：decomposition 只擴充候選池，不改任何分數、不改 gate、不改 certificate。因此它與 gate 維度（第 9c 章）、certificate 維度正交，可獨立疊加。

**防洩漏的關鍵**：router 只對 target-present query 觸發會造成「decomposed 旗標 100% 蘊含 has_target」的標籤洩漏，因此 decomposition 旗標**不可**作為 gate 特徵；本章 decomposition 只用於候選池建構，gate 仍只看 frozen detector 分數（與第 9 章一致）。

## 9b.3 主結果：三 split 純賺 R1

協議：a=b=0.3、g=0.5（與第 9 章 CRS 同協議），parity split，bootstrap CI。**只有候選池不同**。

**表 9b.1　Decomp CRS vs Frozen CRS（同協議，只換候選池）**

| method | split | pool | R1 | R2 | set size | n_feas |
|---|---|---|---:|---:|---:|---:|
| Frozen CRS | val | full | 0.191 | 0.159 | 3.24 | 112 |
| Frozen CRS | testA | full | 0.260 | 0.203 | 2.02 | 58 |
| Frozen CRS | testB | full | 0.168 | 0.236 | 3.50 | 56 |
| **Decomp CRS** | val | decomp | **0.165** | 0.158 | 3.24 | 112 |
| **Decomp CRS** | testA | decomp | **0.244** | 0.201 | 1.96 | 58 |
| **Decomp CRS** | testB | decomp | **0.163** | 0.236 | 3.34 | 56 |

**結論**：Decomp 的 R1 在三個 split **全部 ≤ Frozen**（0.165/0.244/0.163 vs 0.191/0.260/0.168），
而 R2 與 set size **持平**（差異在小數第三位或 set size <0.2 框）。也就是說，換更好的候選池
**單方向降低多目標漏檢，不付任何其他風險或集合大小的代價**。這正是「候選池維度可獨立改善」的鐵證。

## 9b.4 硬防線：三道紅隊補強

主結果之外，本章對三個最可能的攻擊各補一道防線。

### 9b.4.1 Robustness（三切分模式）

Decomp R1 在 parity / random5 / image-disjoint 三種切分下均 ≤ Frozen。val random5：
Decomp R1 0.164±0.003 vs Frozen 0.191±0.007。**誠實標註**：testB random5 下 Decomp 變異較大
（R1 0.226±0.053、set size 2.55±0.65）——某些 seed 壓小集合但升高 R1，是一個不穩點，記入 limitation。

### 9b.4.2 Strong full-expression baselines（堵 matched 假象）

最強的質疑是：Decomp 的 R1 優勢只是「拿擴大的池跟未經整理的 full 池比」的 matched 假象。
為此，我們讓 full-expression pool 也吃各種 consolidation（NMS、top-K），在 val 上正面對比：

**表 9b.2　Decomp vs 強化版 full-pool baseline（val）**

| method | R1 | set size |
|---|---:|---:|
| full raw (threshold=0) | 0.191 | 3.24 |
| full + NMS@0.5 | 0.211 | 2.31 |
| full + NMS@0.7 | 0.174 | 2.99 |
| full + top-10 | 0.241 | 2.64 |
| full + top-20 | 0.237 | 2.69 |
| **Decomp** | **0.165** | 3.24 |

Decomp 的 R1 **低於所有** full-pool consolidation（NMS@0.5/0.7、top-10/20），且此結論三 split 一致。
這證明 R1 優勢來自候選池**涵蓋了 full-pool 整理不出來的 GT**，而非 matched 指標假象——硬防線成立。

### 9b.4.3 Frontier（truly-decomposed 子集的可達邊界）

把分析限縮到 router **真正觸發拆解**的子集（val n=875），畫 recall–size frontier：

**表 9b.3　Truly-decomposed 子集的 frontier（val n=875）**

| pool | rec@sz2 | rec@sz3 | rec@sz4 | size@rec0.8 | AURC |
|---|---:|---:|---:|---:|---:|
| full | 0.589 | 0.686 | 0.753 | 5.48 | 0.690 |
| decomp | **0.823** | **0.913** | **0.937** | **2.08** | **0.875** |

在真正需要拆解的 query 上，decomp 池在 set size=3 時的召回 0.913 vs full 0.686，
達到 recall 0.8 所需集合大小 2.08 vs 5.48 框。**Anti-cheat**：用 calib 選 λ、在 disjoint eval
評估（calib n=419 / eval n=456），target size~3.0 時 decomp eval rec=0.912 vs full 0.682，
結論在 held-out 上維持，非校準集過擬合。

## 9b.5 兩項誠實限制

**1. Over-decomposition（自身錯，非全資料集模糊）。** 失敗審計（P1-F failaudit）顯示，
testA 上 n_gt==1 的 negative-gain case（n=14）中，約 **50% 是標註本身的歧義（annotation ambiguity）、
50% 是 over-decomposition**——router 把一個單目標 expression 錯誤拆成多部件，反而引入多餘候選、
升高 R1。這修正了早期「退步全是標註模糊」的過度宣稱（原宣稱有一半是錯的）。抑制 over-decomposition
（更保守的 VLM router）是明確的 future work。

**2. Training-free 但非 compute-free。** Decomp 不更新任何權重，但有額外推論開銷。
成本量化（P1-E cost）：

**表 9b.4　Decomp 推論成本（vs frozen full-expression）**

| Method | VLM calls / query (val/testA/testB) | GDINO forwards / query | Avg set size |
|---|---|---|---|
| Frozen (full-expr) | 0 / 0 / 0 | 1.00 / 1.00 / 1.00 | 3.24 / 2.02 / 3.50 |
| Decomp (VLM-routed) | 0.37 / 0.77 / 0.71 | 1.06 / 1.08 / 1.07 | 3.24 / 1.96 / 3.34 |

VLM router 對**每個 target-present query** 呼叫一次（actual cost；val 0.37 / testA 0.77 / testB 0.71
calls/query，差異反映各 split 的 target-present 比例），但真正觸發拆解的只有約 6% 的 query
（平均拆 2.1 部件），故 GDINO forward 只多 1.06–1.08x。誠實 claim：**training-free 且 weight-frozen，
但有額外 inference-time compute**——不能宣稱 cheap，只能宣稱不訓練。

## 9b.6 小結

Decomp CRS 證明 candidate pool 是一個**正交且可獨立改善**的維度：在 gate、box scoring、
certificate 全部不變、不碰任何 detector 權重的前提下，VLM-routed decomposition union pool
讓 R1 三 split 全降而 R2/set size 持平。三道硬防線（robustness、strong baseline、frontier）
堵住 matched 假象與過擬合質疑；兩項誠實限制（over-decomposition、compute cost）標明邊界。
這是論文的第二主結果，也是統一框架（第 9d 章）中 pool 軸的實例化。

---

# 第 9c 章　WB-Gate CRS：閘分數維度（第三主結果）

> 本章是三正交維度中的 **gate score** 軸。第 9 章 CRS 用單一 OWL top1 分數當棄答 gate；
> 本章問：把它換成一個**可解釋的 learned gate**，能否在保留 LTT 風險保證下改善 R2/R3 與可行性？
> 答案是肯定的——但**只在同分布成立**，cross-dataset 的 learned boundary 不轉移，這構成本章
> 最有研究價值的 negative finding。

## 9c.1 動機：單一分數的 gate 太弱嗎？

CRS 的 2×2 消融（表 9.3）證明 OWL top1 score 是最佳的 gate 訊號（無目標可分性 0.82）。
但「最佳的單一分數」不等於「最佳的 gate」。OWL 的無目標分數尾巴與 target 分數高度重疊，
任何單一閾值都切不乾淨。一個自然的問題：用多個 frozen 訊號餵一個 learned gate，能不能
比單一分數閾值更好地分離「該答 / 該棄」，從而在更嚴格的 budget 下進入可行域？

約束仍是 frozen：gate 的輸入特徵全部來自 frozen detector 的副產品，不更新任何 detector 權重；
learned 的只是 gate 這個薄薄的 decision 層。

## 9c.2 方法：21-dim 可解釋 white-box gate

**特徵（21 維，全 frozen，無 GT 洩漏）**：OWL×7（top1/margin/entropy/mean-topk 等）+
GDINO×7（對稱的分數統計）+ expr×5（純語法：長度、token 數等）+ cross×2（跨 detector 一致性）。
**禁用特徵**：no_target、n_gt、gt_boxes（任何 GT 衍生量），以及 decomposition 旗標
（會洩漏 has_target，見第 9b 章）。

**模型**：EBM（Explainable Boosting Machine，additive，`interactions=0`）——每個特徵一條可畫出的
shape function，gate 決策完全可解釋，符合「白箱」定位。

**防洩漏 3-way disjoint split**（本章最關鍵的協議）：val 內部以 `ref_id mod 3` 切成三段互斥子集——
gate_train（fit 模型）/ ltt_calib（搜門檻）/ test（評估），且 ref_id + image **雙重 disjoint**，
腳本內硬 assertion 強制。scaler 與模型只 fit gate_train。這修補了早期版本「在全 val 訓練後
又在 val 子集測試」導致 in-sample AUROC 0.99 的洩漏，修補後結果存活。

## 9c.3 主結果一：HB certificate（最大實質收益，免費）

在動 gate 之前，先換 certificate。把 Hoeffding 換成 Hoeffding–Bentkus（HB = min(Hoeffding, e·Binomial-tail)），
**同切分、同 grid、同 Bonferroni，只換 p-value**：

**表 9c.1　bound 對照（image-disjoint α=0.20）**

| bound | #feas | set size | status |
|---|---:|---:|---|
| Hoeffding | 22 | 5.07 | appendix sensitivity |
| **HB** | **69** | **3.72** | main certificate |
| Bernstein | 46 | 3.72 | appendix sensitivity |

可行區 ×3（22→69），set size 5.07→3.72，**完全免費**（不補任何 dump、不訓練）。
健全性四項全過（Bentkus 含 e 常數、p-value 單調、HB⊆Hoeffding superset、test 零違反）。
採用原則：主表用 HB，附錄報 Hoeffding/Bernstein 的 sensitivity。這是 certificate 維度的實例化，
與 gate、pool 正交。

## 9c.4 主結果二：WB-21 gate 讓 infeasible 變 feasible

協議：α=0.3、β=0.2、γ=0.3，HB certificate，3-split disjoint。rule 與 WB 同等套 HB：

**表 9c.2　WB-Gate 主表**

| method | gate | bound | #feas | certified UCB R1/R2/R3 | test R1/R2/R3 | set size |
|---|---|---|---:|---|---|---:|
| rule + HB | OWL top1 | HB | **INFEASIBLE** | — | — | — |
| **WB-21 + HB** | WB-21 EBM | HB | **115** | 0.220/0.032/0.089 | 0.159/0.016/0.075 | 3.68 |

**核心 claim**：單一 OWL top1 gate 在 α=0.3 三 split 全部 **infeasible**；換成 WB-21 learned gate
後 feasible（#feas=115）。certified UCB 全部守在 target 內，test empirical risk 更低
（R1/R2/R3 = 0.159/0.016/0.075）。這證明 gate 軸確實可改善可行性與 R2/R3 trade-off。

## 9c.5 主結果三：WB × Decomp 互補（兩維度疊加）

把 gate 維度（WB）與 pool 維度（Decomp）疊加。2×2：gate(OWL/WB) × pool(full/decomp)，
gate 特徵永遠用 full-pool，只換 box 候選池：

**表 9c.3　WB-Gate × Decomp 2×2**

| version | gate | pool | #feas | test R1 | test R3 | set size |
|---|---|---|---:|---:|---:|---:|
| Rule | OWL | full | INFEASIBLE | — | — | — |
| Decomp | OWL | decomp | **INFEASIBLE** | — | — | — |
| WB-Gate | WB | full | 115 | 0.159 | 0.075 | 3.68 |
| WB+Decomp | WB | decomp | 119 | 0.248 | **0.028** | **2.37** |

**互補鐵證**：decomp 候選池單獨配 OWL gate **仍 INFEASIBLE**——pool 改善需要 gate 改善才能被 CRS
安全使用。只有 WB+Decomp 同時動兩個維度，才把 set size 從 3.68 壓到 2.37（−36%）、R3 從 0.075 降到 0.028。
代價是 R1 從 0.159 升到 0.248（pool 聚焦 + 門檻趨嚴的 trade-off）。這正面回答「兩個改善會不會互相
抵銷」：它們攻不同 risk，疊加得到 Pareto 移動而非抵銷。

**query-adaptive λ（附錄）**：decomp box score 比 passthrough 高約 7×，故分組設不同 λ 可救回 R1
（0.208→0.173），但 R3 上升（0.106→0.196）——R1 與 R3 在此系統中對抗，三方法是同一 Pareto
frontier 的不同操作點。

## 9c.6 核心 negative finding：learned gate 不跨分布轉移

這是本章最重要、也最誠實的發現。固定門檻的 pass-rate 分析揭露：learned gate boundary
**不跨 gRefCOCO split 轉移**。

**表 9c.4　has-target pass rate（rule vs WB，三 tau 操作點一致）**

| split | rule pass\|tp | WB pass\|tp |
|---|---:|---:|
| val | 0.881 | 0.893 |
| testA | 0.861 | **0.388** |
| testB | 0.805 | **0.345** |

rule gate 跨 split 幾乎穩定（0.88→0.81），WB gate **崩塌**（0.89→0.35），三個 tau 操作點一致。
也就是說：**learned target-presence boundary 在 val 有效、在 testA/testB 不對齊**。
learned signal 比 frozen rule score 更受分布影響。

**真因經三輪排除**（這是本章的方法嚴謹度所在）：
- ❌ no-target prior shift：R2/R3 是 conditional risk，對群組比例免疫（resample 20–70%，range 0.001）。
- ❌ GDINO box scoring shift：GT-cover 分數 testA/testB 反而**更高**（val 0.145 / testA 0.173 / testB 0.159），box scoring 沒退化。
- ✅ **真因（兩機制並存）**：(1) OWL no-target 高分尾巴右移（傷 R2，q90 0.155→0.204）；(2) **WB learned boundary 不轉移（傷 R3，主導）**。

**封板定位（收緊 claim）**：WB-Gate = **in-distribution** interpretable gate improvement；
cross-dataset transfer 是 open limitation。同分布結果鐵打，跨資料集 WB 反而比無訓練 rule gate 更不穩。

## 9c.7 negative finding 的研究價值

> A learned gate improves in-distribution selective risk control but may overfit dataset-specific
> target-presence cues; the untrained rule gate is weaker in-distribution yet more robust to
> distribution shift. **學一個更強的可靠性訊號 ≠ 更可靠的跨分布轉移。**

這是 selective prediction 領域一個有意思的誠實教訓，呼應全篇主線「何時該信模型」：
正是當我們訓練出一個「更會判斷該不該信」的 gate 時，它對分布變化反而更脆。這不是修復對象，
而是一個有價值的 negative finding，放 discussion。

## 9c.8 其他負結果（界線，appendix）

- **Feature 擴張 = 沒用**：語法 compositional features AUROC +0.0002（null）；cross-prompt response
  profile AUROC +0.003 但 inner feature-selection 三 split 選三個不同 group → 增益被 split 雜訊淹沒，
  降 appendix。結論：gate 判斷力幾乎全來自 detector scores，feature 層近天花板。
- **Candidate-level utility = per-box trap**：per-box 7-dim EBM utility 取代 score threshold，
  R1 沒改善、set size 2–3× 大、per-box AUROC 僅 0.76。與 evidence-detector 同坑
  （per-box relevance ≠ set-level coverage）。確認 box selection 不可做 per-box classifier。
- **Multi-crop = 診斷否決**：R1 audit 顯示 98.8% query 池中每個 GT 都已有 box cover，
  但 40.7% cover GT 的框 score<0.1（被 λ 濾掉）。R1 瓶頸是 frozen GDINO 的 **scoring**，
  非候選池 coverage；multi-crop 只擴池不改分數 → 解錯問題，不值得跑。
- **Prior-shift = 推翻錯誤歸因**：推翻早期「val→testA/testB 失守 = prior shift」的歸因
  （R2/R3 對群組比例免疫）。

## 9c.9 小結

WB-Gate CRS 證明 gate score 是第三個正交可改善維度：可解釋的 21-dim white-box gate（EBM）
在同分布下把單一 OWL 分數 infeasible 的設定變 feasible，大幅改善 R2/R3，並與 Decomp 互補疊加。
HB certificate 是最大的免費收益。但 learned gate boundary cross-dataset 不轉移
（rule 0.88→0.81 穩 vs WB 0.89→0.35 崩），WB-Gate 的優勢限於 in-distribution——
這是本章最誠實、也最有研究價值的 negative finding。

---

# 第 9d 章　統一框架：三正交維度與 frozen detector frontier

> 前三章（9 / 9b / 9c）各自證明一個維度可改善。本章把它們收斂成一個 coherent system：
> 在完全 frozen 的 base detector 上，用 post-hoc、LTT-certified 的方式把 referring expression
> grounding 轉成 risk-controlled referring set，並系統性地測繪「候選池 / 閘 / 憑證」三個
> 可改善維度的可達邊界。

## 9d.1 一句話框架

> **不訓練任何 detector 權重。所有改善都在 frozen 輸出之上的 decision/calibration 層。**

系統把 grounding 轉成 risk-controlled referring set，並釘出三個可改善維度的可達邊界與天花板。

## 9d.2 系統的三個可改善維度

```
Image + Referring Expression
        │
        ▼
┌──────────────────────────┐
│  Frozen base detectors    │  OWL-ViT (gate signal) + GroundingDINO (boxes)
│  — 權重全凍結，不訓練      │  ← 能力天花板由此決定
└──────────────────────────┘
        │
   ┌────┴──────────────────────────────────┐
   │  維度 1: CANDIDATE POOL（第 9b 章）     │  ← Decomp CRS（主攻 R1）
   │    full-expr pool  /  decomp-union pool │
   └────┬──────────────────────────────────┘
        │
   ┌────┴──────────────────────────────────┐
   │  維度 2: GATE SCORE（第 9c 章）         │  ← WB-Gate CRS（主攻 R2/R3/feasibility）
   │    OWL top1  /  WB-21 interpretable gate│
   └────┬──────────────────────────────────┘
        │
   ┌────┴──────────────────────────────────┐
   │  維度 3: CERTIFICATE（第 9c 章）        │  ← HB-LTT（主攻 feasible region）
   │    Hoeffding  /  Hoeffding-Bentkus      │
   └────┬──────────────────────────────────┘
        │
        ▼
   Risk-controlled referring set  +  R1/R2/R3 certificate (or defer)
```

三個維度**正交且可獨立替換**，這是本論文的核心結構發現。

## 9d.3 三貢獻的分工（互不打架）

**表 9d.1　三維度分工**

| 維度 | 貢獻 | 改什麼 | 主攻 | 同分布結果 | 邊界 |
|---|---|---|---|---|---|
| pool | Decomp CRS | candidate pool | R1 / multi-target recall | R1 三 split 全 ≤ frozen；frontier rec@sz3 0.91 vs 0.69 | over-decomp 反例（n_gt=1 退步一半是自身錯） |
| gate | WB-Gate CRS | gate score | R2/R3 / feasibility | rule INFEASIBLE→WB feasible；R2 0.016 R3 0.075 | cross-dataset learned gate 不轉移（0.89→0.35） |
| cert | HB-LTT | bound | feasible region | #feas ×3，set size 5.07→3.72 | （免費，無邊界） |

**互補的鐵證**：
- decomp-pool 配 OWL-gate **仍 INFEASIBLE**；配 WB-gate 才 feasible → pool 改善需要 gate 改善才能被 CRS 安全使用。
- WB-Gate 主攻 R2/R3，R1 幾乎不動；Decomp 主攻 R1，R2/R3 持平 → 兩者攻不同 risk，疊加得 Pareto 移動而非互相抵銷。

## 9d.4 統一評估協議（所有貢獻共用）

- frozen base：OWL-ViT gate + GroundingDINO box，零訓練。
- 三風險 LTT：R1 answered-target FNR / R2 no-target false-sel / R3 target deferral。
- certificate：HB bound（主），Hoeffding/Bernstein（附錄 sensitivity）。
- 切分：calib/test 互斥；robustness = parity / random5 / image-disjoint。
- 防洩漏：calib-only grid；learned 模組三段互斥切分（gate_train/calib/test）。

任何 (pool, gate, bound) 組合都套同一協議，所以四個版本（Rule / Decomp / WB-Gate / WB+Decomp）
可在同一張表比較。

## 9d.5 整體 claim（三層）

**L1 base claim（CRS，第 9 章）**
> Frozen base + post-hoc LTT 即可把 grounding 轉成有限樣本風險保證的 referring set，無需訓練。

**L2 dimension claims（第 9b / 9c 章）**
> - Decomp：改善 candidate pool 在同分布下降低 R1（multi-target recall），且贏過所有 full-pool consolidation（NMS/top-K），非 matched 指標假象。
> - WB-Gate：可解釋 learned gate 在同分布下把 infeasible 設定變 feasible，大幅改善 R2/R3。
> - HB：更緊的 certificate 免費擴大 feasible region。

**L3 unifying claim（本章）**
> 三維度正交可組合；在 frozen detector 給定下，(pool, gate, bound) 三軸共同決定可達的
> risk-cost frontier。我們測繪了此 frontier 並釘出天花板。

## 9d.6 全景 limitation / negative findings（誠實層）

**表 9d.2　全景限制**

| 項目 | 內容 |
|---|---|
| 天花板 | frozen OWL+GDINO 的 scoring 決定上限；不碰 detector 的手段無法突破 |
| WB-Gate 泛化 | learned gate boundary cross-dataset 不轉移，比無訓練 rule gate 脆弱 |
| Decomp 反例 | n_gt=1 退步一半是 over-decomposition（自身錯），非全資料集模糊 |
| Decomp 成本 | training-free 但非 compute-free（每 target-present +1 VLM call） |
| 已否決 | candidate-level utility（per-box trap）、multi-crop（pool 非瓶頸）、prompt-template（不穩） |
| 已修正歸因 | cross-dataset 失守 ≠ no-target prior shift（受控實驗推翻） |

## 9d.7 天花板定性

整條線證明：**在不碰 frozen detector 的前提下，系統能力上限由 frozen OWL-ViT（gate）+
GroundingDINO（box scoring）決定。** 我們做的一切（白箱 gate、HB、Decomp、adaptive λ）都是在
榨乾這兩個 frozen 模型已產出的資訊，並把可達的 risk-cost frontier 完整測繪出來。

要抬此天花板，唯一有效方向是換更強的 frozen detector（DINO-X / GD-1.5，不違反 frozen，但 API-gated）。
multi-crop / 加 feature / candidate-level 等「不碰 detector」的手段已證明無法突破。

## 9d.8 future work（不在本研究範圍）

- 換更強 frozen detector（DINO-X / GD-1.5）：不違 frozen，抬天花板，API-gated。
- trained scoring head：僅作 upper-bound diagnostic，不進主線。
- WB-Gate cross-dataset 泛化修復（weighted conformal 等）：開新題，暫不做——這是第 9c 章
  negative finding 的自然延伸，可能獨立成一篇「learned reliability signal 為何不轉移」的研究。
- decomp over-decomposition 抑制（更保守的 VLM router）。

## 9d.9 小結

本論文的三條 contribution 不是三個獨立技巧，而是同一個 frozen selective grounding 系統的
三個正交維度。CRS 給出 base claim（frozen + LTT = 有保證的 referring set）；Decomp、WB-Gate、HB
分別在 pool、gate、certificate 三軸上把可達 frontier 往有用方向推；互補性論證證明它們攻不同 risk、
可疊加。我們不只給保證，還測繪了在 frozen detector 給定下的整個 risk-cost frontier，並誠實釘出
天花板與每個維度的邊界——這是本論文相對於任何單點方法的結構性貢獻。

---

# 第 10 章　C4：定位不確定性的 base-invariant 結構

C1–C3 在各自的凍結基礎模型上量測了不確定性訊號的資訊量。本章追問一個更深的問題：**這些訊號所捕捉的，是各基礎模型各自的特性，還是定位任務本身的、跨基礎模型共享的結構？** 這是全篇最接近「科學發現」的貢獻，也是對抗「只是 calibration／threshold」批評的護城河——因為 per-pipeline 的驗證方法（VIRO、True/False Verification）結構上無法提出 transfer 主張。

## 10.1 動機：transfer 數字背後是什麼？

先前的跨基礎模型轉移結果顯示：把 CLIP-VG 上定義的 raw consistency 訊號零參數、無 refit 直接套到 OWL-ViT，其風險—覆蓋 AURC（0.488）接近 OWL-ViT 自身原生 gate（0.489），比隨機降 15.9%。

但「一個 transfer 數字」不足以支撐 thesis-level 主張。本章追問其**機制**：consistency 為什麼能轉移？轉移的到底是訊號的什麼性質？並誠實界定**哪些性質可轉移、哪些不可**。

兩個基礎模型結構上極為不同（CLIP-VG 是單框迴歸、OWL-ViT 是候選評分偵測器），且 RefCOCO val 上準度差近一倍（**CLIP-VG 0.843 vs OWL-ViT 0.419**）。若同一不確定性訊號在如此不同的兩個基礎模型上仍以相同方式運作，則定位失敗存在基礎模型共享的結構。分析在兩基礎模型逐列對齊的 10834 個共同樣本上進行。

## 10.2 發現一：訊號的資訊性／排序結構是 base-invariant 的

對兩基礎模型共有的訊號（cross_prompt_consistency、prompt_box_dispersion）逐一比較其對 correctness 的資訊性：

**表 10.1　共有訊號在兩基礎模型上的資訊性**

| 訊號 | CLIP-VG within AUROC | OWL-ViT within AUROC | corr(訊號, 答對) CLIP-VG | corr OWL-ViT |
|---|---|---|---|---|
| cross_prompt_consistency | 0.722 | 0.641 | 0.288 | 0.271 |
| prompt_box_dispersion | 0.654 | 0.634 | 0.153 | 0.231 |

**判讀**：同一個與定位相關的訊號，在準度差近一倍的兩個基礎模型上，都以**相同方向、相近相關性**預測錯誤（consistency corr 0.288 vs 0.271，幾乎相等）。這不是「訊號剛好在各自基礎模型有用」的巧合，而是訊號所捕捉的「referential 不穩定性」是定位任務本身的性質。

## 10.3 發現二：同一難度軸對兩基礎模型同向有效

更根本的證據不在訊號統計，而在錯誤本身。以 CLIP-VG 的 consistency 作為**共享難度軸**（分 10 個等量分箱），檢視兩基礎模型在每個分箱的 error rate：

- 兩條 error-rate 曲線**同向**：consistency 越低（查詢越不穩定），CLIP-VG 與 OWL-ViT 的 error rate **都上升**。
- 亦即：同一個訊號標出的 hard sample，對兩個獨立、準度迥異的凍結基礎模型都更容易答錯。難度排序是查詢的性質，跨基礎模型共享。

（圖 `c4_hardness.png`：x = 共享 consistency 軸，雙曲線分別為兩基礎模型的 bin error rate。）

## 10.4 誠實的邊界：什麼**不**可轉移

base-invariant 是有限定的。三個誠實的限制必須寫明，否則過度宣稱：

1. **判別強度（effect size）是 base-specific 的。** 以最低 vs 最高 consistency 的 20% 尾端比較 error lift：CLIP-VG 達 **13.9×**（0.348 vs 0.025），OWL-ViT 僅 **1.32×**（0.677 vs 0.513）。consistency 對 CLIP-VG 是極強的難度指標，對 OWL-ViT 只是弱訊號——**方向不變，強度大不同**。

2. **逐樣本 correctness 的跨基礎模型相關性是弱的**（phi = 0.186）。並非「同一批 sample 兩基礎模型一起對／一起錯」，而是「低 consistency 區域兩基礎模型各自的 error 都統計性偏高」。可轉移的是**統計趨勢**，不是**逐樣本一致性**。

3. **訊號的原始數值刻度不可轉移。** 把 CLIP-VG 的 consistency 原始值直接拿去排 OWL-ViT 的 correctness，AUROC 僅 0.551（近隨機）；多特徵 fitted gate 的最佳組合權重也 base-specific（CLIP-VG consistency:dispersion ≈ 1.58:0.86，OWL-ViT native ≈ 0.66:0.07）。可轉移的是**結構**，不是**校準**。

## 10.5 收斂主張

> **與定位相關的不確定性訊號（cross-prompt consistency）所定義的「查詢難度排序」是 base-invariant 的**：同一訊號在準度差近一倍的兩個凍結基礎模型上，都以相同方向、相近相關性預測錯誤，且其標定的 hard sample 對兩基礎模型同向更難。**但其判別強度（effect size）、逐樣本一致性、與最佳組合權重（校準）是 base-specific 的。**

換句話說：**結構可轉移，校準不可轉移。** 這一句同時解釋了 C2 的兩個觀察——為什麼 rank-based 的風險—覆蓋能轉移（只看排序結構），以及為什麼 fitted 多訊號 gate 轉移較差（吃絕對數值與 base-specific 權重）。它也呼應 C1 的觀察：絕對分數最弱、分布／擾動訊號較強，因此跨基礎模型轉移必須避開絕對分數尺度、改用 base-normalized 特徵。

**在 thesis 中的角色**：這是本研究最接近「科學發現」的貢獻，也是對 VIRO / True-False Verification 唯一站得住的真區辨——它們是 per-pipeline / per-VLM，結構上無法提出「不確定性有基礎模型共享結構」這類 transfer 主張。需誠實標註：phi 弱、effect size 差異大，主張必帶限定詞。

## 10.6 與 CRS 的關係與前向指標

需區分兩個不同的 cross-base 主張，寫作時勿混：

- **C4 的 cross-base transfer**：同一訊號跨基礎模型重用，講「訊號結構共享」。
- **CRS 的 cross-base composition**（第 9 章，OWL gate + GDINO box）：不同基礎模型的能力互補因式分解，講「不同基礎模型的能力互補」。

「結構可轉移、校準不可轉移」這個 C4 結論，在 CRS 章升級成一個正面的 **label-efficiency** 前向命題：既然 belief 訊號的*結構*跨基礎模型共享（consistency 兩基礎模型 within-AUROC 近相等），那麼把 CRS 的 LTT 保證重標定到一個新凍結基礎模型，理論上只需**少量** calibration label——「不可轉移的校準」從限制翻成「少量 label 即可重建保證」的正面命題。但需嚴守 wording 紀律：label-efficiency 目前**僅為 future-work 方向，尚未實證**，不可當成本研究的 contribution。C4 提供這個猜測的訊號基礎，其應用出口留待後續工作。

## 10.7 小結

C4 把跨基礎模型轉移從「一個數字」升級成一個有機制、有結構的發現：定位不確定性的**難度排序結構是 base-invariant 的**（同一訊號在兩個準度迥異的凍結基礎模型上同向預測錯誤、同向標出 hard sample），但其判別強度、逐樣本一致性與校準權重是 base-specific 的——**結構可轉移，校準不可轉移**。這是本研究最接近科學發現的貢獻，也是對 per-pipeline 驗證方法唯一站得住的真區辨；同時為 CRS 的跨基礎模型組合與未來的 label-efficiency 命題提供了訊號基礎。

---

# 第 11 章　Cost–Risk Pareto：讓「輕量」成為實驗主張

本論文全程把「輕量」當作可量測的實驗主張，而非形容詞。本章以最誠實的成本軸——**每查詢前向傳遞次數（forward/query）**——對上它換來的可靠性（reliability），回答一個具體問題：**在接近 1× 推論成本下，凍結基礎模型 + 事後校準器能吃到多少可靠性 benefit？** 文獻方法（HieA2G／VIRO）基礎模型不同，僅作定性座標對照，不在同一可靠性軸上直接比較。

## 11.1 成本軸的關鍵區分

不同訊號的推論成本差異，全部歸結到「需要幾次前向傳遞」：

**表 11.1　訊號的前向成本**

| 訊號類別 | forward/query | 為何 | 代表方法 |
|---|---|---|---|
| score-based（OWL-ViT entropy／no-target／margin） | **1×** | 候選分數是**單次前向的免費副產品** | C2 score_entropy gate、C3 no-target gate |
| forced-output／單一 threshold | 1× | 同上 | baseline |
| cross-prompt consistency | **K=4×** | 需 K 個改寫提示各跑一次前向 | C1/C2 consistency gate、C4 transfer 用 |

GB10 實測 throughput：OWL-ViT base-patch32 約 **36.6 query/s/forward**；CLIP-VG batched 約 204 q/s（batch eval，effective）。

## 11.2 Pareto 表

reliability 為 val split 點估計（CI 見各章 bootstrap 節）：

**表 11.2　Cost–Risk–Coverage Pareto**

| 方法 | base | 訓練成本 | forward/query | query/s | reliability |
|---|---|---|---|---|---|
| forced-output base | OWL-ViT | 無 | 1× | 36.6 | no-target AUROC 0.50（無棄答能力） |
| score-threshold (top1) | OWL-ViT | 無 | 1× | 36.6 | no-target AUROC 0.822 |
| **score_entropy gate** | OWL-ViT | 無 | **1×** | 36.6 | **C2 AURC vs random −22.6%** |
| **learned no-target gate (C3)** | OWL-ViT | 小（logreg） | **1×** | 36.6 | **no-target AUROC 0.824** |
| consistency gate (C2) | CLIP-VG | 小（logreg） | 4× | 51 | C2 AURC vs random −52.3% |
| consistency gate (C2) | OWL-ViT | 小（logreg） | 4× | 9.2 | C2 AURC vs random −22.9% |
| HieA2G（文獻，**base 不同**） | RN101 | 高（全監督） | — | — | full-GREC N-acc ~0.60 |
| VIRO（文獻，**base 不同**） | GDINO/Qwen | 0 base / 重型 pipeline | — | E2E 12.92 q/s | balanced acc 0.611 |

## 11.3 主張

1. **1× 成本即取得主要可靠性。** OWL-ViT 的 score 副產品免費，single-forward 即得 no-target AUROC **0.824**（C3）與 C2 AURC 相對隨機降 **22.6%**。這是「輕量」的硬證據——不需額外前向、不重訓基礎模型。

2. **K=4× 成本買到的是 cross-base transferability（C4），不是更高的 within-base 可靠性。** consistency 需 4× 前向，但它是 base-agnostic、零參數可跨基礎模型轉移的訊號（C4 主結果）。亦即成本與「可轉移性」掛鉤，而非單純準度。這也呼應第 3 章的「成本↔可轉移性」二分：(A) 擾動訊號貴但可轉移、(B) 分數訊號免費但 base-specific。

3. **相對重型 pipeline。** VIRO 走 LLM program + per-operator verifier（E2E 12.92 q/s 含重型推論），HieA2G 全監督專訓 counting head。本研究佔「訓練成本 ↓、推論成本 ↓」光譜的最輕量端；表中文獻欄僅定性對照，因基礎模型不同不主張直接比 accuracy。

（圖 `cost_pareto.png`：x = forward/query，y = reliability，標註 1× 的核心主張。）

## 11.4 小結

Cost–Risk Pareto 把「輕量」從形容詞變成可量測主張：在 1× 推論成本、零或極小訓練成本下，凍結基礎模型 + 事後校準器即取得 no-target AUROC 0.824 與相對隨機降 22.6% 的選擇性風險改善；額外的 4× 前向成本買到的是跨基礎模型可轉移性，而非更高的 within-base 準度。相對於 HieA2G（全監督）與 VIRO（重型 neuro-symbolic pipeline），本研究穩居成本—可靠性光譜的最輕量端。這條成本軸也是 CRS（第 9 章）的背景：CRS 的跨基礎模型組合需要兩個基礎模型各一次前向，仍在這個輕量光譜內，卻換來三風險的分布無關保證。

---

# 第 12 章　結論與限制

## 12.1 總結

本論文研究一個與「把基礎模型訓練得更準」正交的問題：**凍結式指稱定位基礎模型，是否在其前向傳遞的廉價副產品中暴露出可重用、且與定位相關的不確定性結構，使我們能在不更新任何基礎模型參數、近乎零訓練成本下，事後校準出可靠的作答／棄答／集合化決策。** 我們把基礎模型視為凍結黑盒，只在其決策層附加一個輕量信心策略 `π`，並在 RefCOCO 系列與 gRefCOCO 上以 CLIP-VG、OWL-ViT（CRS 章再加 GroundingDINO）為實例，從「量測可靠度」一路推進到「建構有分布無關保證的指稱集合」。

四項量測地基確立了「凍結基礎模型的不確定性訊號確實 informative、且其結構可跨基礎模型重用」：

- **C1** 證明哪些信心訊號對答對與否與無目標有資訊量——跨提示一致性（與定位相關的擾動訊號）對答對與否的 AUROC 約 0.72；OWL-ViT 上分數熵對答對與否的 AUROC 約 0.68。
- **C2** 以校準後信心的選擇性預測，在 RefCOCO 的風險—覆蓋曲線上優於隨機與各單訊號閾值，並以 oracle gap 量化未用盡的訊號。
- **C3** 在 gRefCOCO 上以輕量無目標閘把基礎模型原本為零的棄答能力提升到無目標 AUROC 0.74–0.82（三個 split）。
- **C4** 顯示僅用 base-normalized 特徵的策略可不重新擬合地從一個基礎模型轉移到另一個，風險—覆蓋表現接近原生——這是 per-pipeline／per-VLM 驗證方法無法主張的「跨基礎模型可重用結構」。

接著，**M4** 把點預測策略推到完整 GREC 的 exact-match 評估，明確界定事後校準停止有效之處：逐樣本 oracle-τ 上界僅 0.19–0.24，瓶頸是多目標的計數／集合預測，超出單一信心閾值的能力。這道邊界不是失敗，而是促成框架最後一步的動機。

主結果 **CRS（跨基礎模型保形組合指稱集合）** 讓 `π` 輸出基數可為 0／1／多的風險受控集合，以 Learn-then-Test（+ Bonferroni）在校準 split 上聯合校準三個有界風險（R1 已作答目標漏檢率、R2 無目標誤選率、R3 目標棄答率）並給出有限樣本保證，再把代價因式分解為 gate（OWL-ViT）與 box（GroundingDINO）兩個正交瓶頸。在 α=β=0.3 下，COMPOSE 於 val/testA/testB 輸出 **3.24 / 2.02 / 3.50** 個框，三風險全守、集合大小信賴區間與純 OWL-ViT 完全分離（公平比較下縮小 2.0–4.5 倍）；2×2 消融證明唯有對角線組合能同時守三風險並產生緊緻集合，因此這是 factorization 而非單純集成。

在 CRS 的 base claim 之上，本論文進一步把系統拆成**三個正交、可獨立替換的可改善維度**，並各給出實例化（第 9b–9d 章）：

- **Decomp CRS（候選池維度）** 以凍結 VLM router 的 decomposition-union pool 取代原生候選池，同一 LTT 協議下只換候選池，R1 三 split 全 ≤ frozen（0.165/0.244/0.163），R2／set size 持平，且贏過所有 full-pool consolidation——純賺 R1 的第二主結果。
- **WB-Gate CRS（閘分數維度）** 以 21 維凍結特徵上的可解釋 EBM 閘取代單一分數閘，同分布下把 infeasible 變 feasible（#feas=115）、大幅改善 R2/R3，與 Decomp 互補疊加（set size 3.68→2.37）；**憑證維度** 以 HB 免費把可行區 ×3（set size 5.07→3.72）。
- **統一框架** 證明三維度正交可組合，共同決定 frozen detector 給定下可達的 risk–cost frontier，並釘出天花板（frozen OWL+GDINO scoring 上限）。

綜合而言，本論文的貢獻不在於一個更準的定位器，而在於把「凍結基礎模型的不確定性結構」量化為可靠性、成本、跨基礎模型可轉移性與 oracle gap，在點預測撞牆之處給出有分布無關保證的集合化解法，並沿三個正交維度測繪出此解法在 frozen detector 給定下的完整可達邊界。

## 12.2 限制

本論文誠實標註以下限制，避免過度宣稱。

**1. 影像層級分群評估的樣本量不足。** 在以影像為單位的分群（image-cluster）穩健性檢驗中，testA 與 testB 的部分分群為空，樣本量不足以支撐該層級的結論。我們因此只在有足夠樣本的設定下報告分群結果，並以 parity／random／image-disjoint 三種切分模式的重合度佐證整體穩健性，但承認影像層級的細粒度結論仍受樣本量限制。

**2. 長指稱語句下的閘退化。** CRS 的 R2（無目標誤選率）在長指稱語句子集上會退化，反映 OWL-ViT 作為 gate 對長句的弱點：長句的開放詞彙比對較不穩，使無目標誤選的控制變難。這是當前 gate 選擇（OWL-ViT）的內在限制，而非校準協定的缺陷；換用對長句更穩的 gate 基礎模型可能改善，但本論文未驗證。

此外，本論文的「保證」一律僅指 LTT 的有限樣本檢定，bootstrap CI 僅為經驗穩定度；CRS 不宣稱解決完整 GREC 的 exact-match，而是把評測從點預測 pivot 到集合預測。這些界定在前面各章已逐一聲明。

**3. WB-Gate 的 learned 閘不跨資料集轉移（第 9c 章核心 negative finding）。** 固定門檻的目標通過率在 testA/testB 崩塌（0.89→0.35），而無訓練的規則閘穩定（0.88→0.81），三個 tau 操作點一致。真因經三輪排除（非 no-target prior shift、非 GDINO box scoring 退化），確認為 learned target-presence boundary 不對齊。WB-Gate 的優勢限於 in-distribution；這不是修復對象，而是一個有價值的誠實教訓——「學一個更強的可靠性訊號 ≠ 更可靠的跨分布轉移」。

**4. Decomp 的 over-decomposition 與 compute cost。** 單目標語句的退步約半數源於 VLM router 過度拆解（自身錯，非全資料集模糊）；且 Decomp 雖 training-free、weight-frozen，但**非 compute-free**——每個目標語句多一次 VLM router 呼叫（val 0.37 / testA 0.77 / testB 0.71 calls/query）。

## 12.3 未來工作

以下方向均為**尚未進行**的延伸，列為後續可能：

- **抬天花板——更強 frozen detector**：以 DINO-X / GD-1.5 等更強凍結偵測器替換 OWL-ViT / GroundingDINO（不違反 frozen，但 API-gated），檢驗三維度可達 frontier 是否整體上移。multi-crop / 加 feature / candidate-level 等「不碰 detector」手段已證無法突破天花板。
- **WB-Gate cross-dataset 泛化修復**：以 weighted conformal / domain adaptation 嘗試救回 learned 閘的跨分布轉移。這是第 9c 章 negative finding 的自然延伸，可能獨立成一篇「learned reliability signal 為何不轉移」的研究。
- **Decomp over-decomposition 抑制**：更保守的 VLM router，降低單目標語句的過度拆解。
- **trained scoring head（僅作上界診斷）**：輕量訓練 detector 的 scoring head 以量化天花板上界，不進主線（碰 frozen 紅線）。
- **標籤效率與風險網格 sweep**：量化 CRS 校準所需的標註量下界；對 (α, β, γ) 做更細的網格掃描，描繪可行域完整邊界。
- **OOD／egocentric 穩健性**：以 RefAdv、RefEgo 評測 CRS 的分布外穩健性。

這些方向都建立在本論文已確立的地基上：凍結基礎模型的不確定性結構可被事後、低成本地校準成有保證的決策，且這種結構在基礎模型之間是可重用的。

