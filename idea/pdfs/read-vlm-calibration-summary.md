# 論文精讀：Enabling Calibration In The Zero-Shot Inference of Large Vision-Language Models

> 為 CRS（Conformal Referring Set）碩論之「可靠性／校準」背景與基線對照所做的深度導讀。

---

## 一、書目資訊

- **標題**：Enabling Calibration In The Zero-Shot Inference of Large Vision-Language Models
- **作者**：Will LeVine、Benjamin Pikus（兩人並列第一作者）、Pranav Raja、Fernando Amat Gil
- **單位**：Scale AI
- **發表場合**：ICLR 2023 Workshop paper（workshop 等級，非主會議長文）
- **arXiv**：2303.12748（v1 提交於 2023-03-11，最終 v4 修訂於 2023-04-18）
- **領域**：cs.CV / cs.LG
- **篇幅**：正文約 5 頁 + 參考文獻 + 附錄（屬短篇 workshop paper，方法輕量、實證為主）

**一句話定位**：這是一篇「zero-shot VLM（以 CLIP 為代表）到底校不校準？」的系統性量測研究，並提出一個與 zero-shot 使用範式相容的溫度縮放變體（Zero-Shot-Enabled Temperature Scaling, ZS-TS）。它是「VLM miscalibration」這個命題的代表性早期實證文獻之一。

---

## 二、問題定義與動機

### 2.1 為什麼要談校準（calibration）

模型的可信度（trustworthiness）與安全使用的一大關卡，是它的**信賴度估計是否反映真實正確率**。直觀定義（Guo et al. 2017 的經典例子）：若取 100 個信賴度都是 0.8 的預測，理想上應有 80 個是正確的。若成立，模型即為「已校準（calibrated）」。

形式化定義（論文式 (1)）：令 $\hat p(x_i,\hat f)=\max_c \hat f_c(x_i)$ 為模型對 $x_i$ 的信賴度（取 softmax 後最大類別機率）。模型已校準的條件是

$$\text{acc}(\hat f, D^{\text{test}}_p) = p \quad \forall p\in[0,1]$$

其中 $D^{\text{test}}_p$ 是所有信賴度恰為 $p$ 的測試樣本子集。注意：這是**對一群樣本**定義的（單一樣本算不出 accuracy），所以需要分箱（binning）的經驗近似。

### 2.2 為什麼是 zero-shot VLM 這個缺口

校準在「傳統監督分類」上已被大量研究（Guo et al. 2017；Kull et al. 2019 等）：固定類別數、單模態、train/val/test 切分、在 val 上調校。

但 **CLIP（Radford et al. 2021）這種 zero-shot 推論範式打破了上述所有前提**：

1. 資料是多模態（影像 + 文字）。
2. 推論時的「類別」由自然語言 prompt 即時定義，**訓練時並未把這些類別當作目標類別**。
3. 因此沒有「per-inference-dataset 的校準集」可用——若要求每換一個下游任務就重新蒐集標註校準集，就違背了 zero-shot 的初衷。

論文明確指出：在本文之前，**既沒有對 CLIP zero-shot 校準的系統性量測，也沒有把校準方法套用到 CLIP zero-shot 設定的研究**（Minderer et al. 2021 僅順帶提及 CLIP 校準，未跨 prompt/資料集/架構分層研究）。這就是本文要補的缺口。

### 2.3 兩大貢獻

1. **量測（analytical study）**：把 CLIP 的校準按「架構（architecture）、資料集（pre-training + inference）、輸入 prompt」三軸分層做系統量測，發現 **CLIP zero-shot 推論普遍 miscalibrated**。
2. **方法（ZS-TS）**：提出與 zero-shot 範式相容的溫度縮放——**對每個（架構 × 預訓練資料集）配對只學一個純量溫度 $T$**，且此 $T$ 可跨「inference 資料集 + prompt」泛化使用，推論時免再訓練、免調校、免校準集。

---

## 三、方法與技術細節

### 3.1 CLIP 的 logit 與信賴度

CLIP 把 logit 定義為影像嵌入 $E_{im}(x_i)$ 與各類別文字嵌入 $E_{lang}(y_c)$ 的餘弦相似度，乘上固定常數 100：

$$L^{\text{CLIP}}_c(x_i) = 100 \cdot \frac{E_{im}(x_i)\cdot E_{lang}(y_c)}{\lVert E_{im}(x_i)\rVert\,\lVert E_{lang}(y_c)\rVert}$$

（這個 100 本身就是 OpenAI 官方實作裡的標準純量溫度乘數。）再經 softmax 轉成類別機率 $\hat f_c(x_i)$，取 $\max_c$ 得信賴度。

### 3.2 量測工具：Reliability Diagram + ECE

- **Reliability diagram（信賴度圖）**：把樣本依信賴度分到 $M$ 個等寬箱 $B_m$，畫出每箱「真實正確率 vs 平均信賴度」的差距，偏離對角線 $f(x)=x$ 即為 miscalibration。粉紅 = 過度自信（overconfidence），紫色 = 信心不足（underconfidence）。本文固定 $M=10$（業界標準）。同時畫信賴度直方圖 $|B_m|$——**樣本多的箱若失準，比樣本少的箱失準更嚴重**。
- **Expected Calibration Error（ECE，Naeini et al. 2015）**：把 reliability diagram 的失準量總結成單一純量：

$$\text{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{|D|}\,\Big|\,\hat p(\hat f,B_m)-\text{acc}(\hat f,B_m)\,\Big|$$

即各箱「平均信賴度與真實正確率之差」的絕對值，依箱內樣本占比加權求和。**越低越好**。

### 3.3 監督式溫度縮放（傳統基線）

Temperature Scaling（Guo et al. 2017）對凍結網路做事後校準：把 logit 整體除以一個純量溫度 $T$：

$$L^{\text{calibrated}}_c(x_i;T)=L_c(x_i)/T$$

$T$ 由一個**與推論資料集同分布的校準集** $D^{\text{calibration}}$ 透過最小化交叉熵習得。論文也在附錄列了其他監督式方法（Isotonic Regression、Histogram Binning）與 Unsupervised TS 作對照——但這些方法都需要「逐 inference 資料集的校準集」或標籤，與 CLIP 的用法不符，故僅供 context，不適合廣泛採用。

### 3.4 本文方法：Zero-Shot-Enabled Temperature Scaling（ZS-TS）

核心洞見很簡單但定位很巧：

- 對某個固定的（架構 × 預訓練資料集），**只在一個輔助資料集（auxiliary dataset）上用標準 TS 學一個溫度 $T$**。本文一律用 **ImageNet-1k** 當輔助集、prompt 用 `"a photo of {}"`。
- 學好之後，**這個 $T$ 套用到該模型的所有下游推論**，不論換什麼 prompt、什麼 inference 資料集，都直接把 CLIP logits 除以 $T$ 即可，無需再訓練／調校／另備校準集。

為何這仍算「zero-shot」？論文的論證：CLIP 本身就是先在 pre-training 集上訓參數，再對任意未見分布做 zero-shot——ZS-TS 完全平行：在輔助集上訓一個 $T$，再對任意未見 inference 分布免訓練套用。且 $T$ 的變動軸（架構 + 預訓練集）正好等同 CLIP 參數的變動軸，所以「per 架構／預訓練集配對各一個 $T$」是與 CLIP 範式一致的。

---

## 四、實驗與數據

### 4.1 設定

- **模型來源**：全部來自 OpenCLIP（Ilharco et al. 2021）。
- **架構**：ViT-B-16、ViT-B-32、ViT-L-14、ViT-H-14、ResNet-50。
- **預訓練資料集**：ViT 系列用 LAION-400M 或 LAION-2B；ResNet 用 YFCC15M（YFCC100M 子集）或 Conceptual Captions 12M。
- **inference 資料集**：CIFAR10（10 類）、CIFAR100（100 類）、SUN397（397 類）；輔助集 ImageNet-1k（1000 類）。
- **prompt**：多種模板（如 `"a photo of a {}"`、`"a blurry photo of a {}"`、`"a bad photo of a {}"` 等，CIFAR 用了 18 種變體）。
- ECE 以 $M=10$ 計算，每格數字為跨 3 個 inference 資料集（各自帶 prompt）的平均；全為百分比。

### 4.2 核心數據（Table 1，ECE %，越低越好）

| 架構 | 預訓練集 | CLIP（無校準） | CLIP + ZS-TS | CLIP + 監督式 TS |
|---|---|---|---|---|
| ViT-B-16 | LAION-400M | 6.34 | 2.22 | 0.91 |
| ViT-B-16 | LAION-2B | 4.65 | 2.96 | 0.98 |
| ViT-L-14 | LAION-400M | 6.68 | 1.36 | 0.72 |
| ViT-L-14 | LAION-2B | 3.17 | 2.38 | 0.85 |
| ViT-B-32 | LAION-400M | 4.69 | 3.06 | 1.66 |
| ViT-B-32 | LAION-2B | 3.88 | 2.69 | 0.80 |
| ViT-H-14 | LAION-2B | 3.67 | 2.47 | 0.88 |
| ResNet-50 | YFCC15M | **26.69** | 7.60 | 2.61 |
| ResNet-50 | CC12M | **26.56** | 6.18 | 3.31 |

### 4.3 三個關鍵發現

1. **未校準 CLIP 確實 miscalibrated**：ViT 系列裸 ECE 約 3–7%，ResNet（弱資料集 YFCC15M / CC12M）更高達 ~26–27%。校準病理的嚴重程度**強烈受預訓練資料集與架構影響**——資料越乾淨／架構越強，裸校準越好。
2. **ZS-TS 一致改善、但不及監督式 TS**：ZS-TS 在所有設定都優於裸 CLIP（例如 ResNet-50/YFCC15M 從 26.69% → 7.60%），但仍明顯遜於監督式 TS（→ 2.61%）。作者誠實標註「仍有改進空間，留給未來 zero-shot 校準方法」。
3. **單一溫度跨 prompt／資料集泛化（最重要的實證結論）**：Figure 2 顯示，對固定（架構 × 預訓練集），不同 inference 資料集、不同 prompt 的**最佳溫度幾乎相同**，且都很接近在大型輔助集上學到的那個 $T$——儘管這些資料集的類別數天差地別（10 / 100 / 397 / 1000）。但不同（架構 × 預訓練集）配對的最佳 $T$ 確實略有差異（左圖 ~1.55、右圖 ~1.35），所以 $T$ 必須 per 配對各學一個。

### 4.4 附錄補充

附錄用 Isotonic Regression、Histogram Binning、Unsupervised TS 對照，三者都能大幅降低 miscalibration，但都需要逐資料集校準集或標籤，與 CLIP 實務用法不符。

---

## 五、對 CRS 的意涵（重點）

> CRS = 在凍結 grounding base（CLIP-VG / OWL-ViT / GroundingDINO）之上，用 conformal / LTT 風險控制做事後可靠性層，輸出帶覆蓋與棄答雙保證的 referring set。

### 5.1 Framing 引用：「VLM 本身就 miscalibrated → 需要可靠性層」

這篇是極佳的**動機引用**。它用乾淨的實證（Table 1）量化了「zero-shot VLM 的原生信賴度不可信」：

- 裸 CLIP ECE 在強模型上仍有 3–7%，在弱預訓練上飆到 26%。
- 換言之，**直接拿 frozen VLM 的 softmax / score 當機率用是不安全的**——這正是 CRS 立論的起點。

CRS 在 background 可這樣寫：「即便是最廣泛使用的 zero-shot VLM（CLIP），其原生信賴度也系統性失準（LeVine et al. 2023），且失準程度隨架構與預訓練分布大幅波動；因此任何把 frozen base 用於高風險決策（如 referring grounding）的系統，都需要一個**事後可靠性層**來提供可被信任的不確定性語意。」這篇把「需要可靠性層」這件事從直覺變成可引用的數字。

### 5.2 Baseline 對照：point-calibration（TS/ECE）vs distribution-free guarantee（conformal）

這篇代表的是**傳統校準典範**，正好是 CRS 要對照、要超越的對象。對照可寫成：

| 面向 | Temperature Scaling / ECE（本文） | CRS（conformal / LTT） |
|---|---|---|
| 校準對象 | softmax 機率的**點估計**（讓信賴度逼近正確率） | 預測**集合**的風險（覆蓋率 / 召回 / 棄答率） |
| 保證型態 | **無有限樣本保證**；ECE 只是事後的聚合誤差度量，不保證任何下游事件的機率 | **distribution-free、有限樣本**的高機率風險上界（LTT） |
| 是否依賴分布假設 | 隱含假設校準集與推論分布同分布；換分布即失效 | 僅需 exchangeability；對 base 與分布更穩健 |
| 輸出語意 | 「這個機率比較準」——仍是單點軟分數 | 「這個集合以 $\ge 1-\alpha$ 機率含正解 / 棄答風險 $\le\beta$」——可操作的保證 |
| 校準資源 | 監督式 TS 需逐資料集校準集；ZS-TS 用單一輔助集學一個 $T$ | 用一個 calibration split 校準 $\lambda$，輸出對 test 的保證 |

關鍵論述句（供論文用）：「Temperature scaling 改善的是**平均意義下的信賴度品質**（ECE 下降），但 ECE 是一個聚合統計量——它不對任何**個別預測**或**特定下游事件**（例如『正解被涵蓋』）提供機率保證。CRS 改用 conformal / LTT 風險控制，把可靠性從『讓分數更準的點校準』升級為『對覆蓋與棄答的 distribution-free、有限樣本保證』。」

### 5.3 Gap：分類校準 vs grounding set

兩者任務層級不同，這是 CRS 的差異化空間：

- **本文做的是封閉/開放詞表的影像分類校準**：輸出是 single-label 的 top-1 機率，校準的是「最大類別機率 vs 正確率」。
- **CRS 做的是 referring grounding 的集合預測**：輸出是一組 box / region（可能含棄答、含 no-target），要保證的是「正解框落在集合內」的召回與集合大小，跨兩個異質 base（gate + box）做聯合校準。

Gap 的三層：
1. **輸出空間**：標量機率 → 結構化集合（box 集合 + 棄答）。ECE 不適用於集合輸出；需要 conformal set 的 size / coverage 度量。
2. **保證語意**：分類校準求「機率準」；grounding 求「集合涵蓋 + 控棄答」，這是 TS 完全沒處理的維度。
3. **多 base 組合**：本文一個模型一個 $T$；CRS 要在 OWL gate + GroundingDINO box 的 factorization 上做聯合 LTT，溫度縮放沒有對應機制。

因此 CRS 可定位為：「沿用本文揭示的『VLM 原生不可信』動機，但把可靠性工具從**點校準（TS/ECE，無保證）**換成**集合層級的 distribution-free 風險控制**，並從**單模型分類**推廣到**多 base 的 referring grounding**。」

---

## 六、ECE vs Conformal Guarantee：概念區分（可直接放進論文 background）

> 校準（calibration，以 ECE 量、以 temperature scaling 修）與 conformal 風險控制（CRS 採用）解決的是**兩個不同層級的可靠性問題**，不應混為一談。

**ECE / temperature scaling 是「點校準」**：其目標是讓模型輸出的軟分數（softmax 機率）在統計平均上逼近真實正確率。ECE 把預測依信賴度分箱，量測各箱「平均信賴度與真實正確率」的加權絕對差。它是一個**聚合的、事後的誤差度量**——ECE 低只代表「整體而言信賴度數字比較可信」，但它**不對任何單一預測、也不對任何特定下游事件提供機率上界**；而且它**沒有有限樣本的覆蓋保證**，校準集一旦與推論分布不同就可能失效。temperature scaling 則是對應的修法：學一個純量 $T$ 把 logits 壓縮/拉伸，使平均信賴度對齊正確率。它改變的是分數的「銳利度」，輸出仍是不帶任何保證的單點機率。

**Conformal / LTT 是「集合層級的 distribution-free 保證」**：它不試圖把某個分數修準，而是直接建構一個**預測集合**（CRS 中是 referring box 集合，含棄答選項），並透過校準集選一個閾值 $\lambda$，使得在**僅需 exchangeability、不需分布假設**的前提下，給出**有限樣本、高機率**的風險上界——例如「正解以至少 $1-\alpha$ 的機率落在輸出集合內」「棄答/錯誤風險不超過 $\beta$」。這是一種**可操作的、針對指定事件的保證**，而非平均意義的分數品質。

一句總結：**ECE 回答「我的機率數字平均而言準不準」；conformal 回答「我這個集合涵蓋正解的機率有沒有被嚴格保證在門檻之上」。前者是 point calibration（無保證），後者是 set-level distribution-free guarantee（有保證）。CRS 屬於後者，本文（temperature scaling / ECE）是前者的代表，正是 CRS 要對照與超越的傳統基線。**

---

## 七、個人評價

### 7.1 引用優先級

- **動機引用（高優先）**：作為「zero-shot VLM 原生 miscalibrated」的實證錨點，Table 1 的數字（尤其 ResNet 26% 對比 ViT 3–7%）很適合一句話帶過引用。屬於 CRS background 該放的 framing 引用之一。
- **基線對照（中高優先）**：作為「傳統校準典範 = temperature scaling / ECE」的代表，用來凸顯 conformal 的差異。它比直接引 Guo et al. (2017) 更貼題，因為它**就是在 VLM 上做**，與 CRS 同物件（VLM）但不同方法層（point vs set）。建議與 Guo et al. (2017) 一起引（後者是 TS 原典，本文是 VLM 版應用）。
- **方法借鑒（低優先）**：ZS-TS 方法本身對 CRS 幫助有限（CRS 不用 TS），不需深入。

### 7.2 限制（撰文時要誠實標註，也可作為 CRS 的對比優勢）

1. **Workshop paper、實證輕量**：方法是 TS 的一個小變體（換輔助集 + per-配對單溫度），新穎性有限；它的價值主要在量測與定位，不在方法深度。
2. **僅分類、僅 top-1 機率**：完全沒碰偵測 / grounding / 集合輸出，也沒碰 open-set / no-target，與 CRS 的任務有實質 gap——這恰好是 CRS 的差異化空間。
3. **ECE 本身的已知缺陷**（本文未深究）：ECE 受分箱數、分箱方式影響，且只反映平均失準、可能掩蓋局部嚴重失準；它不是保證、只是診斷。CRS 引用時可順帶點出「即使 ECE 低也不等於有覆蓋保證」。
4. **泛化結論的範圍**：「單一 $T$ 跨資料集泛化」是在 CIFAR/SUN397/ImageNet 這類自然影像分類上觀察到的，未必能外推到偵測/grounding 或大幅分布偏移；CRS 不依賴此假設（只需 exchangeability），是更穩健的設計。
5. **未與 conformal 類方法比較**：本文完全在點校準框架內，沒有比較分布無關保證的方法——這不是它的缺點（時代與範疇所限），但正好是 CRS 可以接力的論述縫隙。

**總評**：一篇定位清晰、誠實、適合當「VLM miscalibration + 傳統校準基線」雙重引用的小品。對 CRS 的價值在 framing 與對照，而非方法移植。引用時要把「point calibration 無保證」對上「conformal distribution-free 有保證」這條軸線，才能把 CRS 的貢獻講清楚。

---

*導讀依據 arXiv:2303.12748v4 全文（ICLR 2023 Workshop）整理。*
