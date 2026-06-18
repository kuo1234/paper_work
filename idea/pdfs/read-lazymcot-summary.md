# LazyMCoT 深讀筆記與 CRS 切割分析

> 深讀對象：與本論文（CRS）最接近的兩篇 2026 競品之一。逐項比對其 conformal 用途，並給出審稿防禦切割文字。

---

## 一、書目

- **標題（已自原始頁面核對）**：*Focus When Necessary: Adaptive Routing and Collaborative Grounding for Training-Free Visual Grounding*
  - 註：任務指派的「LazyMCoT」是論文內部的方法代號，**正式論文標題並不含 LazyMCoT 字樣**；arXiv 標題就是 "Focus When Necessary..."。引用時方法名用 LazyMCoT，論文題名用全名。
- **arXiv**：2606.16158v1 [cs.CV]，2026-06-15
- **作者群**：Yifan Wang, Peiming Li, Shiyu Li, Zhiyuan Hu, Xiaochen Yang, Wenming Yang, Yang Tang, Zheng Wei（標註 Tencent BAC；通訊作者 Wenming Yang / Yang Tang / Zheng Wei，Zheng Wei 為 Project Lead）
- **程式碼**：https://github.com/TencentBAC/LazyMCoT
- **骨幹 VLM**：Qwen2.5-VL-7B-Instruct、Qwen3-VL-8B-Instruct、InternVL3-8B-Instruct（全程 frozen，greedy decoding）
- **視覺專家**：SAM3（"Sam 3: segment anything with concepts", 2025）

---

## 二、問題定義與動機

### 任務設定
這是**高解析度影像的「training-free 視覺定位 / 細粒度感知」**問題，而且**答題格式是多選題（multiple-choice）**，最終以 multiple-choice accuracy 衡量。不是 box-level 的 referring expression comprehension，而是「VLM 看高解析度圖回答選擇題，需要時做局部放大/裁切來看清小目標」。

### 核心痛點（論文的切入觀察）
現有 training-free 視覺定位方法（ZoomEye、DyFo、HiDe、DeepScan、ViCrop 等）**對所有樣本一視同仁地施加重度視覺操作**（image scaling + localized cropping），這帶來兩個問題：

1. **計算冗餘**：作者在 V\*、HR-Bench 4K/8K、TreeBench 統一基準上量測，發現 **平均 67.17% 的樣本** 光靠原始 VLM 一次 forward 就能答對（Fig.2），對這些樣本做重度 grounding 純粹浪費算力。
2. **準確率反傷**：對簡單樣本強制做局部裁切，會**截斷必要的全域上下文**並**引入無關背景噪聲**，把本來會答對的題目帶歪（Fig.3）。在「推理重」的任務上尤其明顯——盲目 grounding 反而拉低整體準確率（TreeBench 上 HiDe 把 Qwen3-VL-8B 從 43.0% 拖到 41.7% 就是實例）。

### 核心假設
**base VLM 的初始預測不確定性，是「該不該再花力氣做視覺探索」的強指標。** 因此應該**選擇性**地施加 grounding，而非一律施加。這個「selective / 按難度分配 grounding 努力」的母題，與本論文（CRS）的 selective grounding 母題在精神上同源——這是要特別警覺的相似面。

---

## 三、方法詳解

LazyMCoT = **Adaptive Routing（自適應路由）** + **Collaborative Grounding（協同定位）** 兩個模組。流程：圖 I + 多選題 Q → 單 token forward 拿到「直接答案 + 首 token 統計量」→ Router 判斷 → 簡單題直接回答；難題才進 Collaborative Grounding 產生 Localized Panel Display (LPD) → 用 LPD 重新 query VLM 得最終答案。

### 3.1 First-token uncertainty 是什麼

關鍵設計：用**單次 forward pass、`max_new_tokens=1`** 取第一個答案 token 的 logits $\mathbf{z}\in\mathbb{R}^V$，從中抽兩個**零成本**統計量（$\mathcal{O}$ 為候選選項字母的 token index 集合，靠 regex 從題目解析）：

- **option top probability**：$\mathrm{topp}=\max_{i\in\mathcal{O}}\tilde{p}_i$（在「只保留選項」重新歸一化分布 $\tilde p$ 上的最高選項機率）——衡量機率集中度。
- **option-versus-non-option logit gap**：$\Delta_{\mathrm{logit}}=\max_{i\in\mathcal{O}}z_i-\max_{j\notin\mathcal{O}}z_j$——衡量「選一個合法選項 vs 任何非選項 token」的信心。

作者用 1,000 個樣本驗證：這兩個量與 base VLM 的 predictive entropy（option entropy $H(\tilde p)$）呈**單調相關**，而且能乾淨地把「原始就答對（ori-correct）」與「原始答錯（ori-wrong）」兩類分開（Fig.4）。這就是「first-token uncertainty」——本質是**單一純量化的、一次前傳即得的信心分數**，用來代理樣本難度。

### 3.2 Adaptive Routing：conformal 在這裡到底保證什麼

把上述 $(\mathrm{topp},\Delta_{\mathrm{logit}})$ 餵進一棵 **GBDT（Gradient Boosting Decision Tree，300 estimators / max depth 3 / lr 0.05 / 5-fold CV）** $g_\theta$，預測「ori-wrong 機率」$\hat p(\mathbf x)$，再取 logit 得 routing score：
$$s(x)=\log\frac{\hat p(\mathbf x)}{1-\hat p(\mathbf x)}.$$
GBDT 只在 held-out routing 集 $\mathcal D_{\mathrm{cal}}$ 上訓練一次，部署時固定、不更新 base VLM、不引入額外可學參數。

**Conformal threshold calibration（這是與 CRS 最危險的交集點）**：
- 在 $\mathcal D_{\mathrm{cal}}$ 內取一個 **must-recall 子集** $\mathcal D_{\mathrm{mr}}$（= 所有「ori-wrong 且能從 grounding 受益」的樣本）。
- 取這些樣本的 out-of-fold routing score $\{s_i\}$，對目標 miscoverage rate $\alpha$ 設門檻為經驗 $\alpha$ 分位數：
$$s_{\mathrm{floor}}=Q_\alpha\bigl(\{s_i\}_{i\in\mathcal D_{\mathrm{mr}}}\bigr).$$
- **保證**：依建構，最多 $\alpha$ 比例的 must-recall 樣本會落在 $s_{\mathrm{floor}}$ 之下——即對「困難樣本」提供一個**可控的 recall 下界**。

**Routing rule**：
$$\hat y=\begin{cases}\mathrm{Direct}(I,Q),& s(x)<s_{\mathrm{floor}}\ \text{（信心足，直接回答、零額外成本）}\\ \mathrm{CG}(I,Q),& s(x)\ge s_{\mathrm{floor}}\ \text{（不確定，進協同定位）}\end{cases}$$

$\alpha$ 小 → 門檻低 → router 保守 → 多數樣本都被送去 grounding（recall 困難樣本更穩）；$\alpha$ 大 → 積極跳過 → 多 fallback 到 base VLM。論文預設 $\alpha=0$（最嚴 must-recall 保證）。

> **關鍵判讀**：LazyMCoT 的 conformal **保證的是「router 把困難樣本送去 grounding 的 recall 下界」，也就是一個內部排程/觸發決策的覆蓋率**。它**不對最終輸出（答案集合）做任何覆蓋率或正確率保證**——最終答案仍是 VLM 重 query 後吐的單一選項，沒有集合、沒有 size 保證、沒有 abstention。換言之，它的 conformal 是「**效率/觸發層**」的工具，目的是在保住難題 recall 的前提下盡量跳過簡單題以省延遲。

### 3.3 Collaborative Grounding：協同定位流程（只對被路由的難題執行）

1. **Entity decomposition**：用 rule-based template（沿用 HiDe 的做法）讓 VLM 把問題 Q 拆成 canonical entity list $\mathcal E=\{e_1,\dots,e_M\}$。
2. **兩路平行偵測**：
   - *Visual expert branch*：把每個 entity 當獨立 text prompt 餵 SAM3，得 $\mathcal B_{\mathrm{exp}}$（每 entity 留 ≤ k=10 框，cross-entity NMS IoU=0.7 去重）。傾向 recall 最顯著的實例，但會漏小/被遮擋目標。
   - *Attention branch*：在 VLM 輸入後接 "Search the following entities..." prompt，記錄 cross-modal attention $A\in\mathbb R^{T\times N}$（T=entity token 數，N=visual token 數），逐 token reshape 到空間網格、Gaussian blur（σ=3）、歸一化、跨 token 平均成 saliency map $\mathcal A(I)$；相對門檻 τ=0.5 取 connected components 得 $\mathcal B_{\mathrm{att}}$。覆蓋題目相關區域但較噪。
     - *Per-VLM attention 層選擇*：Qwen2.5-VL-7B / InternVL3-8B（28 層）取第 15 層；Qwen3-VL-8B（36 層）因 DeepStack 多層注入，改聚合全部層。
3. **Two-stage refinement**：
   - Stage 1：取聯集 $\mathcal B^{(1)}=\mathcal B_{\mathrm{att}}\cup\mathcal B_{\mathrm{exp}}$ 當粗證據池。
   - Stage 2：對每個「未被 $\mathcal B_{\mathrm{exp}}$ 覆蓋的 attention 框 b」，裁切放大區域後**重新 query SAM3**，新發現的框 $\Delta\mathcal B$ 映回原圖座標併入，得 $\mathcal B^{(2)}=\mathcal B^{(1)}\cup\Delta\mathcal B$。
4. **Render LPD**：把 $\mathcal B^{(2)}$ 用彩色框 + 文字 legend 畫成一張 **Localized Panel Display $\hat I$**，讓 VLM 在一次 forward 內對多個證據 patch 一起推理，輸出最終選項。

---

## 四、實驗

### 資料集與指標
- **V\* Bench**：191 張、平均 2246×1582，分 Direct Attribute (Att.) / Spatial Relationship (Spa.)。
- **HR-Bench-4K / 8K**：4K、8K 高解析；各分 Single-Instance (Sin.) / Cross-Instance (Cro.)。人類準確率約 87%。
- **TreeBench**：405 題、平均 2152×1615，10 子類分 Perception（Attributes / Material / Phy. State / Obj. Retr. / OCR）與 Reasoning（Per. Trans. / Ordering / Con.&Oc. / Spa. Cont. / Comparison）。
- **指標**：一律 **multiple-choice accuracy**。**完全沒有 box-level IoU、沒有 GREC 指標、沒有任何 referring-set / 集合大小指標。**

### 主結果（Table 1, HR-Bench + V\*）
- LazyMCoT 對三個骨幹相對 base 的平均提升：InternVL3-8B **+3.3**、Qwen3-VL-8B **+7.9**、Qwen2.5-VL-7B **+8.4** 點。
- Training-free 法中**每個 aggregate 欄都排第一**，平均超過 HiDe。
- 在 Qwen2.5-VL-7B 上 V\* 平均達 **90.6%**，匹配甚至超過 training-based 法（DeepEyes / Thyme-VL / TreeVGR）。
- 最大增益在 V\*-Spatial（Qwen2.5-VL-7B **+17.1** 點）。

### TreeBench（Table 2，反傷驗證）
- Qwen3-VL-8B：HiDe 把 base 從 43.0% **拉低到 41.7%**（盲目 grounding 傷推理題），LazyMCoT **拉回 43.5%**。
- Qwen2.5-VL-7B：LazyMCoT 41.7%，Perception +3.4 點，勝所有 training-free 競品。

### Latency（Fig.6）
- 單 H20、batch=1。只開 Collaborative Grounding 比 HiDe 稍慢（多了 SAM3 驗證），**但開 router 後可短路簡單題，平均延遲下降**。

### 消融
- **AR + CG（Table 3）**：CG 全開把 V\* 79.1%→90.0%、HR-4K 71.8%→77.4%，但 TreeBench 增益受限；再加 AR 把 V\* 推到 90.6%、TreeBench 到 41.7%——兩者互補（CG 給難題精度，AR 擋簡單題被傷）。
- **兩階段 CG（Table 4）**：$\mathcal B_{\mathrm{att}}$ 單獨 80.3%、$\mathcal B_{\mathrm{exp}}$ 單獨 85.9%；Stage 1 聯集 88.4%；加 Stage 2 達 90.6%。
- **conformal $\alpha$ 掃描（Table 5，Qwen2.5-VL-7B / V\*）**：$\alpha=0$ → $s_{\mathrm{floor}}=-0.25$、skip rate 24.7%、V\* 90.6%（最佳甜蜜點，省掉約四分之一樣本的 grounding）。$\alpha$ 越大 skip rate 越高（最高 94.2%）但準確率掉到 79.25%。**這張表正是「conformal 控的是 routing 觸發率 / 跳過率與準確率之 trade-off」的直接證據**。

### 與 CRS 錨點的對照（重要）
- **完全沒碰 gRefCOCO、RefCOCO、RefCOCOg。** 三個 benchmark 全是高解析多選題基準。
- **完全沒有 no-target / multi-target / abstention（棄答）** 概念。任務裡每題都有唯一正解選項，不存在「圖中無此目標」或「多個目標」的判定，也不會輸出「拒答」。
- **完全沒有 cross-base 異質組合校準**：它是「單一 frozen VLM + 一個 SAM3 expert」的 pipeline；conformal 校準的是 router 一個純量門檻，不是兩個異質 base 的聯合風險。

---

## 五、與 CRS 的逐項 diff 表

| 維度 | LazyMCoT（2606.16158） | CRS（本論文） |
|---|---|---|
| **任務 / 輸出** | 高解析多選題 VQA，輸出**單一選項字母** | gRefCOCO/RefCOCO 指涉定位，輸出 **conformal referring SET（一組框）** |
| **指標** | multiple-choice accuracy | GREC 指標（Pr@(F1=1)、N-acc、T-acc）、集合大小、recall/risk CI |
| **conformal 用在哪一層** | **效率/觸發層**：校準 router 門檻 $s_{\mathrm{floor}}$ | **輸出層**：校準輸出框集合本身 |
| **conformal 保證什麼** | 「困難樣本被送去 grounding」的 **recall 下界**（routing recall），$\le\alpha$ must-recall 樣本被誤跳 | **輸出集合的 recall（覆蓋）下界 + abstention/棄答風險上界，同時保證**（LTT 聯合控制 R1/R2/R3） |
| **保證對象** | 內部決策（要不要進 grounding） | 終端使用者看到的輸出集合性質 |
| **校準的統計工具** | 經驗 α-分位數門檻（single-threshold conformal） | **LTT（Learn-then-Test）+ Bonferroni 多風險**、calib-only grid 防 leakage |
| **base 結構** | **單一 frozen VLM**（Qwen2.5/3-VL、InternVL3）+ 1 個 SAM3 expert | **cross-base 異質組合**：OWL-ViT gate + GroundingDINO box，聯合校準 |
| **是否有棄答 / abstention** | **無** | **有**（abstention 是核心保證之一） |
| **gRefCOCO / multi-target / no-target** | **全無**（每題唯一正解，無 no-target） | **核心**（gRefCOCO no-target / multi-target 正是主場） |
| **factorization / ablation 論證** | AR vs CG 互補、兩階段 CG | OWL gate × GD box 的 2×2 factorization（非 ensemble） |
| **router / 決策有沒有可學參數** | GBDT 路由器（held-out 訓練一次，部署固定） | frozen 組合，LTT 純後處理校準（無新增可學參數於輸出端） |
| **selective 的對象** | selective **要不要花力氣**（算力/延遲） | selective **要不要輸出 / 輸出多大集合**（可靠性/棄答） |

**一句話總結差異**：LazyMCoT 把 conformal 當「**省算力的開關閥門**」（保證難題不被漏送進 grounding 管線）；CRS 把 conformal 當「**輸出可靠性的合約**」（保證使用者看到的框集合同時滿足召回與棄答風險）。兩者都叫 "conformal + selective grounding + frozen backbone + training-free"，**但作用平面正交**：一個在效率層、一個在輸出層。

---

## 六、論文該怎麼引用 / 切割（審稿防禦）

這幾乎一定是審稿人會丟出「你被搶先了」的那篇——因為標籤高度重合：training-free、frozen backbone、conformal calibration、selective/按難度 grounding、甚至連 $\alpha$ miscoverage、recall lower bound 這些字眼都出現。必須在 Related Work 與 contribution 兩處主動切割，**先發制人**。可直接取用以下中／英兩版段落。

### 中文切割段（放 Related Work / 差異討論）
> 與本工作最接近的並行研究是 LazyMCoT（Wang et al., 2026, *Focus When Necessary*）。兩者皆在 frozen backbone 上採 training-free 的 selective grounding，並借用 conformal 校準提供 recall 下界，表面高度相似。**但兩者的 conformal 作用平面正交。** LazyMCoT 的 conformal 校準的是「自適應路由器」的觸發門檻 $s_{\mathrm{floor}}$，其保證對象是**內部排程決策的 recall**——即「困難樣本被送入 grounding 管線的比例下界」，目的是在不漏送難題的前提下短路簡單題以**節省推理延遲**；它的最終輸出仍是單一多選答案，**不對輸出本身提供任何覆蓋率、集合大小或棄答保證**，且實驗僅在高解析多選 VQA（V\*、HR-Bench、TreeBench）上，不涉及 gRefCOCO 的 no-target/multi-target 與棄答情境。相對地，CRS 的 conformal（透過 LTT 多風險聯合控制）直接校準**終端輸出的 referring set**，同時保證召回下界與棄答風險上界，並建立於 OWL-ViT 與 GroundingDINO 兩個**異質 base 的 cross-base 組合**之上。因此，LazyMCoT 解決的是「**何時值得花力氣**」（效率），CRS 解決的是「**輸出能不能被信任、何時該棄答**」（可靠性合約）；前者的 conformal 是效率閥門，後者的 conformal 是輸出層覆蓋保證。

### 英文切割段（可直接放 paper）
> The closest concurrent work is LazyMCoT (Wang et al., 2026), which, like ours, performs training-free selective grounding on a frozen backbone and uses conformal calibration to obtain a recall lower bound. The two methods are nonetheless **orthogonal in what the conformal guarantee controls.** LazyMCoT calibrates the *triggering threshold* of an adaptive router: its guarantee bounds the recall of *hard samples routed into the grounding pipeline*, so as to short-circuit easy cases and reduce inference latency. Its final prediction remains a single multiple-choice answer, and it provides **no coverage, set-size, or abstention guarantee on the output itself**; its evaluation is confined to high-resolution multiple-choice VQA (V\*, HR-Bench, TreeBench), with no no-target/multi-target referring or abstention. In contrast, CRS calibrates the **terminal output set** (a conformal referring set) via Learn-then-Test multi-risk control, *simultaneously* guaranteeing a recall lower bound and an abstention-risk upper bound, on a **cross-base composition of two heterogeneous detectors** (OWL-ViT gate + GroundingDINO box). Hence LazyMCoT answers *"when is grounding worth the compute"* (an efficiency valve), whereas CRS answers *"can the output be trusted and when to abstain"* (an output-level reliability contract).

### 切割要強調的三個硬差（審稿人一看就懂）
1. **保證的平面不同**：routing-recall（內部觸發）vs output-set coverage + abstention（終端輸出）。這是最強、最不可辯駁的差異。
2. **任務與評測不同**：多選 accuracy vs GREC（含 no-target/multi-target/棄答）。LazyMCoT 連 referring set 的概念都沒有。
3. **base 結構不同**：單模型 vs cross-base 異質組合 + 聯合風險（LTT/Bonferroni）。

---

## 七、個人評價

**對 CRS 新穎性的威脅判定：低至中、不致命，但「必須引用且必須主動切割」。**

- **為何不致命**：LazyMCoT 的 conformal 與 CRS 的 conformal 雖共用詞彙（recall lower bound、α miscoverage），但**作用對象完全不同**。LazyMCoT 校準一個純量路由門檻、保證「難題進管線」的觸發 recall，目標是省延遲；它的輸出沒有任何集合層的覆蓋或棄答保證。CRS 的招牌——**對輸出 referring set 同時保證召回下界與棄答上界、且建立在兩個異質 base 的 cross-base 聯合校準上**——LazyMCoT 一項都沒做。任務（多選 VQA vs gRefCOCO 指涉集合）、指標（accuracy vs GREC）、有無棄答、有無 no-target/multi-target，全是乾淨的非交集。因此「被搶先」的指控在技術上站不住腳。

- **為何仍需嚴肅對待**：它是同月（2026-06）、同標籤（training-free + frozen + conformal + selective grounding）、且來自 Tencent 的高曝光工作。審稿人很可能只看摘要關鍵字就斷定重複。**風險不在技術，而在表面相似的 framing。** 防禦方式就是上面那段「conformal 作用平面正交」——把「routing recall（效率閥門）vs output-set coverage+abstention（可靠性合約）」這組對比寫死在 Related Work 顯眼處，並在 contribution 列點中明確聲明 CRS 是「輸出層、跨異質 base、含棄答」的保證。

- **可順手吸收的養分**：
  1. LazyMCoT 的 first-token uncertainty（topp / Δlogit / option entropy 單調相關）是個漂亮、零成本的 reliability 信號。CRS 若日後要做「何時棄答 / 何時觸發第二 base」的成本-Pareto 分析，這套首 token 統計量可當一個便宜的 routing prior 來做 cost-aware 版本（且可在 Related Work 借它佐證「first-token 統計可作可靠性指標」這個 claim 在文獻中已有支撐）。
  2. 它的 Fig.2/Fig.3「盲目 grounding 反傷簡單樣本」與 TreeBench 反傷數據，是 CRS「selective grounding 有必要」這個母題的**現成外部佐證**，可引用來強化動機。
  3. 它的 ablation Table 5（α–skip rate–accuracy）是很好的「conformal 的用途是效率 trade-off」對照組——CRS 可在差異討論直接指這張表，說「對方的 α 控的是 skip rate，我方的 α/δ 控的是 output recall/abstention risk」，差異一目了然。

- **小瑕疵備忘**（若要在 review 反擊其侷限）：(a) 它的 "conformal" 其實只是經驗分位數單門檻，沒有 LTT/多風險的嚴謹性，也未討論 exchangeability 假設在 must-recall 子集上的成立性；(b) recall 下界只在 calibration 分布上成立，未做 cross-split / image-disjoint robustness（CRS 有做三 split 模式）；(c) 完全無棄答能力，遇到 no-target 會強制亂答——這正是 CRS 的賣點落差。

**結論**：引用它、在 Related Work 用「正交保證平面」一段切乾淨即可，不需要改 CRS 主結構。它非但不削弱 CRS，反而可被當作「selective grounding 動機」與「first-token reliability signal」的外部支撐。
