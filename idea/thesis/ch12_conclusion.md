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
