# 論文精讀：Conformal Risk Control for Non-Monotonic Losses

## 書目資訊

- **標題**：Conformal Risk Control for Non-Monotonic Losses
- **作者**：Anastasios N. Angelopoulos（Arena；亦即 CRC、LTT、conformal prediction tutorial 的核心作者本人）
- **出處 / 識別碼**：arXiv:2602.20151v1 \[stat.ME\]，2026-02-23
- **致謝對象**（顯示其學術定位）：Rina Foygel Barber、Stephen Bates、John Duchi、Yaniv Romano、Ryan Tibshirani、Vladimir Vovk、Tijana Zrnić 等，並提及《Philosophical Transactions of the Royal Society》匿名審稿，屬於 conformal 圈核心人物的「奠基／統整」型論文。
- **程式碼**：https://github.com/aangelopoulos/nonmonotonic-crc

一句話總結：**這篇把「Conformal Risk Control（CRC）為什麼有效」重新解讀成「演算法穩定性（algorithmic stability）」的後果，從而把 CRC 的保證從「一維、單調損失」推廣到「多維參數 θ（屬於 d 維實空間）、可能非單調的損失」。代價是保證會多出一個與演算法穩定度 β 成正比的鬆弛項；越不穩定的演算法，保證越鬆。**

---

## 一、問題定義與動機：為什麼「非單調損失」對 CRC 是個問題

### 1.1 標準 CRC 的設定與限制

設定：可交換（exchangeable）的資料序列 \(D_{1:n+1}=((X_1,Y_1),\dots,(X_{n+1},Y_{n+1}))\)，前 \(n\) 個是校準集、第 \(n+1\) 個是測試點。有界損失 \(\ell(x,y;\theta)\in[0,1]\)，參數 \(\theta\in\mathbb{R}^d\)。目標是用校準資料挑一個 \(\hat\theta\)，使測試點的**期望損失**被控制：

\[
\mathbb{E}[\ell(X_{n+1},Y_{n+1};\hat\theta)]\le\alpha. \tag{1}
\]

原始 CRC（Angelopoulos et al. 2024b）**只處理 \(d=1\) 且 \(\ell\) 對 \(\theta\) 單調非遞增**的情形。它的做法是取滿足經驗風險約束的最小 \(\theta\)：

\[
\hat\theta=\inf\Big\{\theta:\tfrac{1}{n+1}\textstyle\sum_{i=1}^n \ell(X_i,Y_i;\theta)\le \alpha-\tfrac{1}{n+1}\Big\}. \tag{2}
\]

### 1.2 非單調 = 致命

論文在引言就明講：上式（root-finding，挑最小可行 θ）對**非單調損失「可能任意嚴重地失效」（fail arbitrarily badly）**，這正是 Angelopoulos et al. (2024b) 的 Proposition 1 已證明的反例。

直覺原因：單調損失下，風險曲線 \(R(\theta)\) 只會跨越 α 一次，所以「挑最左邊低於 α 的點」是穩定且正確的。但**非單調損失下，經驗風險曲線可能在好幾個不相連的區域都觸碰 α**；只要在某處下穿、又在別處上穿，那麼「拿掉一個校準點」就可能讓你選到的 θ 大幅跳動（root 從一個交點跳到另一個遙遠的交點）。一旦這個跳動大，校準集上看起來控制住的風險，到測試點就不再成立——交換性論證的核心（校準與測試可互換）被破壞。

**這就是動機**：要在非單調情形下重建保證，必須量化「拿掉 / 加入一個點會讓選出的 θ（進而風險）變動多少」——這正是 **algorithmic stability** 的語言。

---

## 二、方法詳解

### 2.1 核心定義：β-stability（相對於參考演算法）

設 \(\mathcal{A}\) 是把資料集映到 θ 的演算法，\(D_{-i}\) 是拿掉第 \(i\) 點的資料集。稱 \(\mathcal{A}\) 相對於**參考演算法** \(\mathcal{A}^*\) 與損失 \(\ell\) 是 **β-stable**，若：

\[
\mathbb{E}\Big[\tfrac{1}{n+1}\sum_{i=1}^{n+1}\ell(X_i,Y_i;\mathcal{A}(D_{-i}))\Big]
\le
\mathbb{E}\Big[\tfrac{1}{n+1}\sum_{i=1}^{n+1}\ell(X_i,Y_i;\mathcal{A}^*(D_{1:n+1}))\Big]+\beta. \tag{3}
\]

直白說：「在 \(n\) 點上跑 \(\mathcal{A}\)（leave-one-out）」的平均風險，不會比「在全部 \(n+1\) 點上跑理想參考演算法 \(\mathcal{A}^*\)」的風險高出超過 β。這是一種 **leave-one-out 穩定性**。全文假設演算法是 **symmetric（排列不變 / permutation-invariant）**。

### 2.2 主定理（Theorem 1）：穩定即可控制風險

**假設**：\(\mathcal{A}\) 對稱、相對 \(\mathcal{A}^*\) 為 β-stable；\(D_{1:n+1}\) 可交換；且參考演算法在全資料上「留出 β 餘裕」地控制風險：

\[
\mathbb{E}[\ell(X_{n+1},Y_{n+1};\mathcal{A}^*(D_{1:n+1}))]\le\alpha-\beta. \tag{4}
\]

**結論**：

\[
\mathbb{E}[\ell(X_{n+1},Y_{n+1};\mathcal{A}(D_{1:n}))]\le\alpha. \tag{5}
\]

**證明只有三行**：由可交換性 + 對稱性，把「測試點期望」換成「\(n+1\) 點平均期望」；對 \(\mathcal{A}\) 與 \(\mathcal{A}^*\) 兩邊都這樣換；再套 β-stability（式3）與餘裕假設（式4）即得。

**值得注意的兩點**（作者自己強調）：
1. Theorem 1 **對 θ 所在的空間沒有任何要求**（任意空間皆可，已含多維 \(\mathbb{R}^d\)）。
2. **證明中完全沒用到 \(\ell\) 有界**。有界性與維度的問題，只在「之後估 β 的具體上界」時才會回來。

> **工作流（actionable workflow）**：(i) 找一個在全 \(n+1\) 點上能控制風險的參考演算法 \(\mathcal{A}^*\)；(ii) 用只跑 \(n\) 點的 \(\mathcal{A}\) 去逼近它；(iii) 證明兩者風險差 ≤ β。然後把 α 收緊成 α−β 來跑即可。

### 2.3 把舊 CRC 收編：單調即 0-stable（Proposition 1 + Corollary 1）

作者證明：單調損失下，標準 CRC 演算法相對於對應的 \(\mathcal{A}^*\) 是 **0-stable**（β=0）。代入 Theorem 1（β=0）**完全重現**經典 CRC 保證。

> **本文最關鍵的觀念解耦**（原文 "key takeaway"）：CRC 有效性的證明可拆成兩塊——(1) 單調 ⇒ 穩定；(2) 穩定 ⇒ 風險控制。**Theorem 1 證明了 (2) 對一般演算法都成立**。本文其餘部分就是「把 (1) 換成其他能保證穩定的演算法 / 條件」，以涵蓋非單調損失。

### 2.4 三類「可保證穩定」的演算法（Section 2.2，皆 d=1）

#### (a) 一般有界損失 → 靠離散化（Proposition 2）
把 Θ 離散成網格 \(\Theta_m=\{0,\tfrac1m,\dots,1\}\)，並假設存在安全解 \(\ell(\cdot;1)=0\)。則離散化的 root-finding 演算法滿足

\[
\mathbb{E}[R(\hat\theta_n)]\le\alpha+\tilde{\mathcal{O}}\!\left(\tfrac{1}{\sqrt n}\right),
\]

明確常數以 Lambert \(W_{-1}\) 函數表示（附錄用 Hoeffding + union bound over 網格證明）。**對任意非單調有界損失都成立**，代價是 \(\tilde O(1/\sqrt n)\) 的鬆弛——這比下面的 Lipschitz 情形差（\(1/n\) 對 \(1/\sqrt n\)）。

#### (b) 連續、Lipschitz 損失且在 α 有「強交叉點」（Proposition 3 + Corollary 2）
若損失對 θ 連續、L-Lipschitz，且**經驗風險在「最左邊跨越 α 的點」附近具足夠斜率的局部線性，且跨越後不再太靠近 α**（式13 的三個條件），則演算法是 \(\tfrac{L}{m(n+1)}\)-stable，於是

\[
\mathbb{E}[R(\hat\theta_n)]\le\alpha+\frac{L}{m(n+1)}.
\]

**這就是處理非單調的核心幾何直覺**（原文）："the risk cannot touch α in two highly disjoint regions, otherwise the selected parameter may be unstable"——**風險不能在兩個高度分離的區域都觸碰 α，否則選出的參數會不穩定。** 這正是非單調之所以危險的精確刻畫，也是「為何需要額外條件」的本質。

#### (c) Selective classification（最詳盡的案例，含顯式常數）
這是本文視為非單調 CRC「最重要應用」的窄案例。目標是挑信心門檻 \(\hat\theta\) 使

\[
\mathbb{P}(\hat Y_{n+1}\neq Y_{n+1}\mid \hat P_{n+1}>\hat\theta)\le\alpha,
\]

等價於一個**非單調**損失 \(\ell=\mathbf 1\{\text{錯且過門檻}\}-\alpha\mathbf 1\{\text{過門檻}\}+\alpha\) 的期望控制。每個樣本的損失是**分段常數、只有一個變化點**（恰在 \(\hat P_i\)）。

- **Proposition 4**：該演算法是 β-stable，其中
  \[
  \beta=\frac{2\max\{\alpha,1-\alpha\}\,\mathbb{E}[K]}{n+1},\quad K=\max_i|\hat\jmath_{-i}-\hat\jmath_{n+1}|,
  \]
  \(K\) 是「選出的門檻在排序索引空間中，因 leave-one-out 而移動的最大格數」。是**distribution-free** 的穩定性刻畫。
- **Corollary 3**：選擇性準確率下界 \(\mathbb{P}(\hat Y=Y\mid\hat P>\hat\theta)\ge 1-\alpha-\tfrac{2\max\{\alpha,1-\alpha\}\mathbb E[K]}{n+1}\)。
- **Proposition 5**：\(\mathbb E[K]\) 被「累積平均錯誤率 \(\bar E_j\) 穿越一條圍繞 α 的收縮細帶 \((\alpha+\tfrac{1-\alpha}{j},\,\alpha+\tfrac{2-\alpha}{j}]\) 的次數」上界（Figure 1）。**模型排序越好（信心高者越準），\(\bar E_j\) 越快遠離 α，K 越小、β 越小。** 排序差或對抗情形 K 才大。

### 2.5 多維 + ERM（Section 2.3）——這是把保證推到 \(d>1\) 的部分

研究正則化 ERM：\(\mathcal{A}(D)=\arg\min_\theta \hat R_D(\theta)+\tfrac\lambda2\|\theta\|_2^2\)，損失可凸、可無界、\(d\ge1\)。

- **損失尺度（Proposition 6 + Corollary 4）**：若 ℓ 凸且 \(\rho\)-Lipschitz，則 ERM 是 \(\beta\le\tfrac{2\mathbb E[\rho^2]}{\lambda(n+1)}\)-stable。這是 Bousquet–Elisseeff (2002) 經典穩定性結果的小變形。
- **梯度尺度（Theorem 2 + Proposition 7/8）**：引入**多維 β（向量值，逐分量偏序）**的穩定性概念，控制「期望梯度接近 0」。Proposition 7 給出 \(\beta\sim\frac{\mathbb E[\text{test grad}]+\mathbb E[\text{train grad}]}{(\mu+\lambda)(n+1)}\)。Proposition 8 加一個線性項 \(\gamma\mathbf 1_d^\top\theta\) 把所有分量往負方向推，**保證地（conservatively）**得到 \(\mathbb E[\nabla\ell]\preceq 0\)。
- **應用（Corollary 6）**：用 OLS 後處理黑箱模型，得到**對所有（可重疊的）群組同時近似無偏**——亦即 distribution-free 的 multigroup / multiaccuracy / multicalibration 型保證。

### 2.6 怎麼在實務中估 β（Section 2.4）——直接 bootstrap

關鍵實務細節：**不要用上面那些解析上界（太鬆），而是直接從 β 的定義用 bootstrap 估**。對每個 bootstrap 重抽資料 \(D^{(b)}\)，算

\[
\Delta^{(b)}=\tfrac{1}{n+1}\sum_i\big[\ell(Z_i^{(b)};\mathcal A(D_{-i}^{(b)}))-\ell(Z_i^{(b)};\mathcal A^*(D^{(b)}))\big],
\]

再取 \(\hat\beta_{\rm def}=(\overline\Delta)_+\)（bootstrap 均值的正部）。作者坦承 subsampling 下 bootstrap 的有效性「並非平凡」，是未來工作。

---

## 三、實驗（Section 3）：三種方法的對照

論文一律對照三種方法：

- **CRC-C**：保守版，在收緊後的 \(\alpha'=\alpha-\hat\beta_{\rm def}\) 跑（本文主推方法）。
- **CRC**：不修正、直接用 α 跑（在 β 很小時其實就夠）。
- **LTT（Learn-then-Test）**：高機率保證 \(\mathbb P(\mathbb E[\ell\mid D_{1:n}]\le\alpha)\ge1-\delta\)（取 δ=0.1）。**作為「非單調風險控制的標準 baseline」與對照。**

**全篇一致觀察**（極重要）：**LTT 在所有實驗中都比 CRC / CRC-C 更保守、變異也更大**，因為 LTT 給的是「高機率」保證，而 CRC 系列給的是「期望」保證。

各實驗的 β 估值（顯示「在這些任務上非單調幾乎不咬人」）：
- **ImageNet selective classification**（ResNet-152，n=1000）：β=0.006，「CRC 基本上不修正也安全」。
- **Polyp segmentation FDR**（PraNet，n=500）：β≈0.00007，CRC 與 CRC-C 幾乎完全重合。
- **Polyp segmentation IoU**（ERM）：β=0.000056。
- **COMPAS 多群去偏**（n=1000，5 個重疊群組指標）：β=0.001839，幾乎不需修正，所有群組達近似無偏。

> **實驗的 take-away**：對「排序合理 / 損失夠平滑」的真實模型，非單調帶來的不穩定性 β **小到可忽略**，CRC 直接用就行；CRC-C 的修正是廉價的保險；LTT 是更保守、更貴的安全網。

---

## 四、對 CRS（本論文 / 碩論主結果）的理論意涵

> 提醒 CRS 設定：CRS = Cross-Base Conformal Composition，用 **OWL-ViT gate + GroundingDINO box**，以 **LTT（Learn-Then-Test）在一個 grid 上、用 Bonferroni** 在多維校準參數上**聯合控制 recall risk + abstention risk**，輸出一個 conformal referring set，在 gRefCOCO / RefCOCO。

### 4.1 CRS 的 recall / abstention 風險是不是「非單調」？——**幾乎可以確定是，且是多維的**

判斷如下，分兩層：

**(a) 維度層面：CRS 是 d>1（多維 θ），標準 CRC（式2）根本不適用。**
標準 CRC 的閉式 \(\hat\theta=\inf\{\dots\}\) 與 Corollary 1 的 0-stable 證明，**前提是 \(d=1\) 且損失對單一 θ 單調**。CRS 同時調 OWL gate 門檻與 GD 相關參數（grid over 多維），**這已經直接落在「標準 CRC 不涵蓋」的區域**，無關乎單調與否。這一點本身就足以排除「直接套 vanilla CRC（式2）」。

**(b) 單調性層面：recall 損失對門檻多半非單調，abstention 與之張力使聯合損失更不可能單調。**
- 把 gate 門檻調低（更願意輸出框）→ recall miss 變少（recall 損失下降），但 abstention 也下降——這兩個風險**方向相反**。
- 但 recall 損失對「同一個 box-side 參數」未必單調：composition 下，改變 box 參數會同時影響「哪些框進集合、集合是否命中 ground-truth」，與 selective classification 的非單調損失（Section 2.2.3，分段常數、含 \(-\alpha\mathbf 1\{\text{過門檻}\}\) 這種負項）**結構同型**——本文正是把這類「過門檻才開始算、且帶 α 折扣」的損失列為**典型非單調案例**。
- 更直接的類比：CRS 的 abstention/no-target 控制與本文 selective classification 的 \(\ell=\mathbf 1\{\text{錯且過門檻}\}-\alpha\mathbf 1\{\text{過門檻}\}+\alpha\) 幾乎是同一隻怪物。**本文明說這隻 ℓ「is a non-monotonic loss」。** CRS 的 recall 損失沿門檻移動時「漏掉一個框」的事件，正是這種「過門檻才觸發、可在不同門檻區段反覆觸發」的非單調。

**結論判斷：CRS 的（recall, abstention）聯合風險是「多維 + 非單調」的，標準 CRC 在 CRS 上既不適用（維度）、也不安全（非單調，可能 fail arbitrarily badly）。**

### 4.2 那這篇對 CRS 是「支撐用 LTT 的理由」還是「提供更緊的替代」？——**兩者都是，但對 CRS 的主用途是『支撐 LTT』，本文 CRC-C 是『可選的更緊替代』。**

**(1) 作為「為何不能用 vanilla CRC、需要 LTT」的理論引用——強力支撐。**
- 本文白紙黑字：標準 CRC 對非單調損失「can fail arbitrarily badly」（引 Angelopoulos 2024b Prop 1）。
- 本文把 LTT 當成**非單調風險控制的「standard baseline」**並在每個實驗使用它。
- 因此 CRS 在方法章可以這樣引用：「由於 CRS 的聯合（recall, abstention）損失在多維校準參數上非單調，標準 CRC（Angelopoulos et al. 2024b）的單調性假設被違反、可能任意失效（見 Angelopoulos 2026, §1, §2.2），故我們採用對損失形態不作假設的 LTT，並以 Bonferroni 在 grid 上聯合控制多個風險。」**這是把『我為什麼用 LTT+grid+Bonferroni』從『工程選擇』升級成『有理論必要性的選擇』的關鍵一句。**

**(2) 作為「更緊的替代」——技術上可行，但對 CRS 有重要保留，列為 future work 較穩。**
本文的 CRC-C（β-stability + 在 α−β 上跑 CRC）相較 LTT **一致地更不保守、變異更小**（這是全篇實驗的核心賣點）。理論上 CRS 可以考慮改用 CRC-C 取代 LTT 以**得到更小的 referring set / 更高召回**。但有四個務必注意的限制：

- **CRC-C 控制的是「期望風險」\(\mathbb E[\ell]\le\alpha\)，LTT 控制的是「高機率風險」\(\mathbb P(\mathbb E[\ell\mid D]\le\alpha)\ge1-\delta\)。** 這是**不同的保證語意**。CRS 目前對審稿人主打的「雙保證 + CI 上界」敘事是 LTT 式的；換成 CRC-C 等於把保證從 PAC 型降成期望型，**這是 downgrade 而非 free lunch**，不能含糊。
- **本文的乾淨穩定性常數（Prop 4 的 β 公式）只證到 d=1 的 selective classification**；多維走的是 ERM/梯度路線（Prop 6–8），那是「argmin + 正則化」的演算法，**與 CRS「在 grid 上挑滿足約束的門檻組合」的 root-finding 結構不同**。要對 CRS 的具體 composition 演算法套用 Theorem 1，得**自行證一個 β-stability bound 或用 bootstrap 估 β**——本文沒有現成定理直接覆蓋「OWL gate × GD box 的多維 root-finding」。
- **Theorem 1 假設演算法 symmetric（排列不變）**。CRS 的校準流程須確認對校準樣本排列不變（一般 split-conformal 式的 grid 校準是滿足的，但若有任何順序相依的步驟需檢查）。
- **多風險聯合**：本文 Theorem 2 的多維 ⪯ 是針對「梯度向量逐分量」；CRS 的「recall risk + abstention risk 兩個純量風險聯合」更接近「對一個向量損失逐分量控制」。Theorem 2 的框架**可以**容納（把 g 設成 (recall_loss, abstention_loss) 的向量），但需要一個對應的向量 β-stability bound，且仍是期望型保證——**它不會自動給你 Bonferroni 那種同時高機率保證**。

### 4.3 方法章該加哪一句話（建議直接可用）

> 建議在 CRS 方法章「為何採用 LTT 而非 CRC」處加入（中文版）：
>
> 「CRS 的校準同時調整跨基底的多維參數（OWL-ViT 閘門門檻與 GroundingDINO 框參數），且其(召回, 棄答)聯合損失在這些參數上並非單調——隨門檻移動，『漏框』事件可在多個不相連的參數區段反覆觸發，與選擇性分類中的非單調損失同型。標準 Conformal Risk Control 仰賴一維、單調損失的假設（Angelopoulos et al., 2024b），在此被違反，且對非單調損失可能任意失效（Angelopoulos, 2026）。因此我們採用對損失形態不作假設的 Learn-Then-Test（Angelopoulos et al., 2021/2025a），並以 Bonferroni 校正在離散網格上聯合控制召回與棄答兩項風險，取得同時成立的有限樣本保證。」
>
> （英文版 one-liner，可放 related work / method 註腳）：
> *"Because CRS calibrates a multi-dimensional cross-base parameter and its joint (recall, abstention) loss is non-monotonic in that parameter, the monotonicity assumption underpinning vanilla CRC is violated and CRC can fail arbitrarily badly (Angelopoulos et al., 2024b; Angelopoulos, 2026); we therefore adopt LTT with a Bonferroni-corrected grid, which makes no monotonicity assumption."*

> （可選的「未來工作 / 限制」一句，誠實地點出 CRC-C 是更緊但語意不同的替代）：
> 「若僅需期望層級（而非高機率）的保證，Angelopoulos (2026) 的 stability-based CRC-C 提供了較 LTT 更不保守的替代；將其延伸至 CRS 的多維 composition、並導出對應的演算法穩定性界，是值得探索的方向。」

---

## 五、個人評價

### 可用性
- **引用優先級：高（建議納入，且放在「方法選擇正當性」的位置，而非僅 related work）。** 這篇是 CRC 原作者親自寫的「非單調 / 多維」延伸，正好命中 CRS「為何不能用 vanilla CRC」的要害。它把你的 LTT 選擇從「工程上比較安全」升級為「有定理背書的必要選擇」，對審稿人說服力強。
- **它同時是把 LTT 定位成『非單調 baseline』的權威來源**——你用 LTT 不是冷僻選擇，而是這條線最新論文裡的標準對照組。

### 限制（對 CRS 而言）
1. **本文沒有給你「現成可套用到 CRS composition 的定理」**。它給的乾淨穩定性常數限於 d=1 selective classification（Prop 4）與凸 ERM（Prop 6–8）；CRS 的多維 root-finding-over-grid 不在其封閉形式覆蓋範圍。要當「更緊替代」用，你得自己證 β 或 bootstrap 估 β，並接受**期望型（非高機率型）保證**的語意降級。
2. **CRC-C 與 LTT 的保證語意不同**：別把「CRC-C 比 LTT 不保守」誤讀成「免費更緊」。對主打 PAC 風格 CI 的 CRS，貿然換成 CRC-C 會弱化敘事。建議**主結果續用 LTT，把 CRC-C 當 future work / 附錄對照**。
3. **bootstrap 估 β 的有效性作者自承未證（subsampling 下非平凡）**，若要用 CRC-C，這是個可被審稿人攻擊的點。

### 最務實的用法
- **立刻可做（零成本、高回報）**：在方法章加上 §4.3 那段引用，把「用 LTT+grid+Bonferroni」正當化。**這是這篇對 CRS 最直接、最值得的價值。**
- **可選加值（中成本）**：在附錄或 future work 提一句 CRC-C 作為「期望型、更緊」的替代方向，並誠實標註語意差異與 bootstrap-β 的開放性。
- **不建議**：把主結果的 LTT 直接換成 CRC-C——保證語意降級、且 CRS 的多維 composition 沒有現成穩定性界，風險大於收益。

### 一句話裁決
**CRS 的（召回, 棄答）聯合損失確為多維且非單調，標準 CRC 既不適用也不安全；這篇論文是『為何必須用 LTT』的強力理論支撐（首要用途），同時提供了一條『CRC-C 更緊但保證語意不同』的可選替代（次要、列 future work）。方法章請加上 §4.3 的引用句。**
