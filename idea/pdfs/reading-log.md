# 閱讀記錄 — CRS 相關文獻精讀

> 主線：CRS = Cross-Base Conformal Composition（OWL-ViT gate + GroundingDINO box，LTT 聯合校準 recall + abstention 雙保證，輸出 gRefCOCO/GREC 上可棄答的 conformal referring set）。
> 本檔記錄已精讀完畢的論文、每篇的 takeaway、對 CRS 的角色與引用優先級。

---

## 第一批：REF/grounding 經典與 benchmark（2026-06-18 完成）

| 論文 | 年份/ID | 狀態 | 對 CRS 角色 | 摘要檔 |
|------|---------|------|------------|--------|
| Modeling Relationships in Referential Expressions | CVPR 2016 / 1607.07702 | ✅ 讀畢 | REC modular 起點；Subj/Loc/Rel/Dist ablation 範式可對照 CRS 2×2 factorization | [modeling-relationships-summary.pdf](modeling-relationships-summary.pdf) |
| COPS-Ref | CVPR 2020 | ✅ 讀畢 | controlled distractor 資料設計；可做第三資料集驗證 cross-base；後繼 FineCops-Ref (EMNLP'24, 2409.14750) | [copsref-summary.pdf](copsref-summary.pdf) |
| GREC（技術報告） | 2023 / 2308.16182 | ✅ 讀畢 | 主場 benchmark：gRefCOCO + Pr@(F1=1)/N-acc；threshold baseline 正是 CRS LTT 要取代的對象。**引用改用期刊版 GREx 2601.05244** | [grec-summary.pdf](grec-summary.pdf) |
| HieA2G | AAAI 2025 / 2501.01416 | ✅ 讀畢 | A2G=Alignment-enhanced Adaptive Grounding；AGC 計數器是英雄、無 calibration 是縫隙；可做第三 base | [hiea2g-summary.pdf](hiea2g-summary.pdf) |
| Zero-Shot True/False Verification | 2025 / 2509.09958 | ✅ 讀畢 | 頭號 zero-shot 威脅；但無統計保證、不做 gRefCOCO、威力綁 GPT-5 閉源（開源版崩到 44.6） | [zeroshot-tf-summary.pdf](zeroshot-tf-summary.pdf) |

---

## 第二批：護城河 + 方法正當性核心（2026-06-18 完成）

挑選邏輯：直接回應「base 越來越強是否還需要 CRS」這題的兩條防線 —
護城河（最近鄰居 #2/#3）、方法新穎性（結構相似的 conformal 偵測/分割 #1/#5）、方法理論正當性（#4）。

### #1 SeqCRC — Conformal Object Detection by Sequential Risk Control
- **2505.24038**（2025-05）→ [read-seqcrc-summary.pdf](read-seqcrc-summary.pdf)
- **角色**：結構上最近的鄰居（同 post-hoc/frozen、同有兩依序校準參數）。審稿人首選質疑對象。
- **裁定**：三條不可被吸收的差異軸 —
  1. abstention/no-target 軸（SeqCRC **完全沒有**棄答，confidence threshold 只是 filter）
  2. cross-base factorization（SeqCRC 兩參數是同一偵測器內部旋鈕；CRS 是兩異質凍結模型分工，2×2 ablation 證成非 ensemble）
  3. 風險語意+保證型態（SeqCRC = CRC 期望值/marginal 保證；CRS = LTT+Bonferroni PAC 高機率）
- **戰術**：主動引用、承認結構同源、把差異收斂到三軸；借力 SeqCRC 自承弱點（只控 recall 不控 precision、set size heavy-tail）。
- **引用優先級**：高（必引必切割）

### #2 LazyMCoT — Focus When Necessary（Adaptive Routing & Collaborative Grounding）
- **2606.16158**（2026-06-15，Tencent）→ [read-lazymcot-summary.pdf](read-lazymcot-summary.pdf)
- 註：LazyMCoT 是方法代號、不在標題裡；引用時分開。
- **角色**：2026-06 最近鄰居之一，最可能被說「被搶先」。
- **裁定**：威脅低-中。它的 **conformal 用在 routing/效率層**（校準路由器觸發門檻，保證困難樣本被送進管線的 recall 下界），最終輸出仍是單一多選答案、**對輸出本身無覆蓋率/集合大小/棄答保證**。CRS 的 conformal 在終端輸出（referring set recall+棄答雙保證+cross-base）→ **保證平面正交**。它完全沒碰 gRefCOCO/no-target/multi-target。
- **真正風險**：表面 framing 相似（同月、training-free+frozen+conformal+selective grounding 關鍵字全中）。防禦＝把「conformal 作用平面正交」寫進 Related Work 顯眼處。
- **額外收穫**：它「盲目 grounding 反傷簡單樣本」的數據可借來當 selective grounding 動機外部佐證。
- **引用優先級**：最高（必引必切割）

### #3 BCEA — Look Again Before You Abstain
- **2606.16667**（2026-06-15，Jian Xu/Delu Zeng/John Paisley）→ [read-bcea-summary.pdf](read-bcea-summary.pdf)
- **角色**：2026-06 最近鄰居之二，框架語彙與 CRS 高度重疊。
- **裁定**：外殼撞、內核正交，威脅低。三承重維度不同 —
  - 任務（object-existence 是非題驗證 vs referring-set 框定位）
  - 保證（單一 false-assertion rate vs LTT 同時控 recall+abstention 多風險）
  - 組合（單一 LVLM 自身 vs 跨兩異質 frozen base factorization）
- **定位**：「同期、同調、不同任務」的最佳對照，襯托 moat 而非競品。
- **引用注意**：BCEA 正文寫 Qwen2.5-VL，bib 卻列 Qwen3-VL (2511.21631)，版本對不上，引用前核對。
- **引用優先級**：最高（必引必切割）

### #4 Conformal Risk Control for Non-Monotonic Losses
- **2602.20151**（2026-02-23）→ [read-crc-nonmonotonic-summary.pdf](read-crc-nonmonotonic-summary.pdf)
- **角色**：給 CRS 方法理論正當性的最重要一篇。
- **裁定**：CRS 損失是「多維 + 非單調」，標準 CRC 既不適用（維度：vanilla CRC 前提 d=1）也不安全（非單調：本文白紙黑字「standard CRC can fail arbitrarily badly」）。CRS 漏框 recall 事件與本文非單調 selective-classification 損失同型。
- **最大收穫**：本文把 **LTT 定位成非單調風險控制的標準 baseline** → CRS 的 LTT+grid+Bonferroni 從「工程選擇」升級成「有定理背書的必要選擇」。
- **用法**：方法章直接加一句（agent 已寫好中英版+future-work 註腳在摘要 §4.3）。CRC-C 是更緊但控期望值（非 PAC）的替代，列 future work，別當免費午餐（無現成定理覆蓋多維 root-finding-over-grid）。
- **引用優先級**：高（方法章理論引用）

### #5 Conformal Prediction Sets for Instance Segmentation
- **2602.10045**（2026-02-10，Lu/Kluger/**Bates**/Wang, MIT）→ [read-conformal-iseg-summary.pdf](read-conformal-iseg-summary.pdf)
- **角色**：「set 內 ≥1 命中」同型保證，鄰近任務（像素 query 的 instance seg）。
- **裁定（最辛辣，要聽進去）**：威脅在**修辭層、不在方法層**。兩者保證骨架相同（set ≥1 高品質、marginal ≥1−α），且這套 existential recall 保證在 Angelopoulos LTT 早就有 → **CRS 不能再把「set ≥1 命中保證」當核心賣點**，會被蓋過。
- **護城河仍站得住**：語言 query (REC)、abstention/no-target 雙保證（本文無棄答）、cross-base factorization（本文是單模型掃參數，與 CRS 分工相反）。
- **可借鏡 4 點**：(1) 去重+重校準 IoU 門檻兩階段技巧 Algorithm 2（可移植到 box set）；(2) α–τ 可行性 frontier 畫法；(3)「單參數 LTT/CRC 不可行 + IoU loss 非單調」論證模板；(4)「hit vs containment」保證語意區分。
- **不宜照搬**：別把 CRS 改成單模型掃參數，會丟掉 factorization 賣點。
- **引用優先級**：高（必引必切割 + 借鏡 nonconformity 設計）

---

## 核心問答：「base 越來越強，所以不需要 CRS 了吧？」

**結論：恰好相反。base 越強，CRS 越有價值。**
1. **準確率 ≠ 保證**：92.7% 的意思是 7.3% 會錯而你不知道是哪 7.3%；CRS 在 base 上加一層有限樣本、分布無關的風險上界。
2. **base 越強 → conformal set 越小、棄答越少 → CRS 越漂亮**（set 大小反比於 base 品質）；「base 變強」是 CRS 的順風，要寫進 motivation。
3. **no-target/棄答是正交軸**，準確率解不掉（forced-output N-acc 0.18 → policy 後 0.94，是結構性 gap）。
4. **92.7% 有水分**：RefCOCO 飽和有捷徑；同方法搬到 gRefCOCO multi/no-target 從 85% 掉到 26–58%。

**必守的防線**：framing 絕不能踩準確率戰場（否則 base 一強就沒戲）；且必做 LTT vs naive-threshold 對照，證明「強 base + 簡單 threshold」給不出有限樣本保證或會過度棄答。

---

## 第三批：SOTA 定位 + 威脅切割 + framing + 實驗擴充（2026-06-18 完成）

挑選邏輯：對準下一步論文動作 —— SOTA 定位、威脅切割、方法 framing、實驗擴充四面向。

### #6 InstanceVG — Improving Generalized Visual Grounding with Instance-aware Joint Learning
- **2509.13747**（TPAMI 2025，Ming Dai/東南大學）→ [read-instancevg-summary.pdf](read-instancevg-summary.pdf)
- 註：InstanceVG 是系統/repo 名，標題不含此前綴。
- **角色**：現任 GREC SOTA，要 position-under 的方法層天花板。
- **精確數字（已核實）**：GREC val Pr@F1=1/N-acc **73.5/72.8**、testA 70.2/71.1、testB 60.8/65.2（與 memory 一致）。⚠️「all metrics SOTA」略誇 —— GRES N-acc 輸給 DeRIS（72.84 vs 77.03）。
- **關鍵性質**：N-acc 來自訓練式 BCE existence 分支的 **point estimate + 後處理 thr_q**（val 上手動掃 0.7→0.9），**無 calibration/coverage/distribution-free 保證**；Conclusion 自承「target existence determination 準確度不足」。
- **定位（強化而非威脅 CRS）**：(1) 不比絕對分數比可靠性，caption 誠實點明 CRS Pr@F1=1(0.26–0.61) 低於 73.5 是「保證 vs 點估計」取捨；(2) **SOTA 表用雙區塊 + 「Guarantee?」欄**（Trained 標 ✗ vs Frozen CRS 標 ✓）；(3) InstanceVG 當 box base → **不入主結果、列 future-work**（in-domain trained 非泛用 base，會稀釋 frozen 故事），最漂亮做法是小 robustness 附錄「用 InstanceVG 框替換 GD box，CRS 保證是否仍成立」展示 base-agnostic。
- **引用優先級**：高（SOTA 對照表核心）

### #7 VIRO — Verification-Integrated Reasoning Operators
- **2601.12781v2**（POSTECH，Suha Kwak/Jungseul Ok 等）→ [read-viro-summary.pdf](read-viro-summary.pdf)
- **角色**：威脅切割 —— neuro-symbolic REC，踩同一條軸（post-hoc/training-free/no-target/同用 CLIP+GroundingDINO）。
- **裁定**：不構成實質威脅。三刀切割 —
  1. 保證類型（VIRO 棄答靠啟發式 per-label CLIP 門檻、無有限樣本保證、TNR 是點估計；CRS 用 LTT 給 `P(risk>α)≤δ`）
  2. 輸出語意（VIRO 單框或 ∅、二元 0/1、不處理 multi-target；CRS size+覆蓋受控 referring set）
  3. 架構（single pipeline vs cross-base factorization）
- **額外彈藥**：VIRO **未用官方 GREC Pr@(F1=1)/T-acc**，自定二元 TPR/TNR/Balanced Acc，no-target 與 target-present 分 split 各自量 → 口徑不同不可並列。**可反過來當 CRS 的 baseline**（官方 GREC metric 下重評它、凸顯無保證）。
- **引用優先級**：高（abstention 軸必引必切割）

### #8 Are foundation models for computer vision good conformal predictors?
- **2412.06082v3**（Fillioux/Silva-Rodríguez/Ben Ayed/Dolz）→ [read-fm-conformal-summary.pdf](read-fm-conformal-summary.pdf)
- **角色**：方法 framing —— 直接背書 CRS 的 post-hoc 設計。
- **支撐發現**：(iv) 對 17 個凍結 FM 套 Temperature Scaling 後，adaptive conformal set 效率**一致退化**（APS set size 全面變大）→ 背書「不重校準 base 原始分數」；(v) few-shot（更過自信）反而改善 conformal 分數 → 「校準分數 ≠ 更好 conformal」；(i) 凍結 FM conformal 指標優於監督式重訓 ViT。
- **⚠️ 外推 caveat（必寫）**：本文非一致性分數建立在封閉 K 類 softmax；CRS 用 grounding/box 分數非 softmax，APS/RAPS 機制無法原樣搬，**數值結論不可直接外推**。定位＝放 motivation（不重校準的設計理由）+ method justification 附 caveat 句；CRS 對應現象仍須自身 2×2 ablation 證成。
- **引用優先級**：中高

### #9 Enabling Calibration In The Zero-Shot Inference of Large VLMs
- **2303.12748v4**（Scale AI, LeVine et al., ICLR 2023 **Workshop**）→ [read-vlm-calibration-summary.pdf](read-vlm-calibration-summary.pdf)
- **角色**：方法 framing 雙重用途。
- **裁定**：(1) 動機引用 —— 裸 CLIP zero-shot 確實 miscalibrated（ViT ECE 3–7%、弱預訓練 ResNet 26–27%），當「frozen VLM 原生信賴度不可信」實證錨點，與 Guo et al.(2017) 並列引；(2) baseline 對照 —— 它是傳統 point-calibration(TS/ECE) 代表，正是 CRS 要超越的對象。
- **ECE vs Conformal 區分（可貼進 background）**：ECE/TS = point calibration 無保證（只讓軟分數平均逼近正確率，換分布即失效）；Conformal/LTT = set-level distribution-free guarantee（僅需 exchangeability 給有限樣本高機率上界）。一句話：ECE 答「機率數字平均準不準」、conformal 答「集合涵蓋正解機率有無被嚴格保證在門檻上」。Gap：ECE 不適用集合輸出 = CRS 差異化空間。
- **引用優先級**：中高（framing + baseline）

### #10 VL-SAM-v3 — Memory-Guided Visual Priors for Open-World Object Detection
- **2605.03456v3**（北大王選所）→ [read-vlsamv3-summary.pdf](read-vlsamv3-summary.pdf)
- **角色**：實驗擴充 —— 評估第 4 個 frozen base 可行性。
- **裁定：不適合，推薦度「低」**。三硬傷：(1) 介面錯位（吃**類別字串**不吃 referring expression，零 RefCOCO/gRefCOCO 證據）；(2) 非 frozen（核心需 fine-tune base detector）；(3) 無公開 checkpoint + 重型依賴（64GiB FAISS + Qwen3-VL + DINOv3）。
- **改用什麼（修正最高 ROI 實驗標的）**：**首選 Grounding-DINO-1.5 (2405.10300)**（現成 frozen、吃自然語言 query、出 box、與現有 GD 介面相容、最低整合風險）；次選 **DINO-X (2411.14347)**（分數分布更異質，強化 cross-base）。
- **引用優先級**：低（排除為 base，但結論有價值：把「加第四 base」標的修正到 GD-1.5）

---

## 補讀：MAttNet（2026-06-19）

### #16 MAttNet — Modular Attention Network for REC
- **1801.08186v3**（CVPR 2018，Licheng Yu/UNC+Adobe）→ [mattnet-summary.pdf](mattnet-summary.pdf) / [mattnet-summary.md](mattnet-summary.md)
- **角色**：REC modular 經典/多年強 baseline，*Modeling Relationships*（CVPR'16）的直接後繼。`idea/pdfs/` 中唯一有原文卻缺 summary 者，補齊。
- **方法**：語言 soft attention 自動把句子解析成 subject/location/relationship 三模組（取代外接 parser，勝 ~5%）+ 雙視覺 attention（subject in-box 看框內細節、relationship out-of-box 看框外支援物件）+ 屬性預測分支；res101-frcn 全模型 RefCOCO testA 85.26、RefCOCOg test 78.12，box+pixel 兩層級刷新 SOTA。box 定位與分割解耦（選 box→Mask 分支出 mask）優於 FCN 式。
- **定位（強化非威脅）**：(1) SOTA 雙區塊表的 **「Trained」代表**，標 Guarantee ✗（point estimate 無分布無關保證）對照 Frozen CRS ✓，高絕對分數（85.26）正好佐證「準確率≠保證」framing；(2) subj→loc→rel→attr 逐步加法 ablation = CRS 2×2 factorization 的可對照前例（差別：同模型內模組可加 vs 跨異質 frozen base 分工非 ensemble）；(3) single-target+需訓練+無 calibration = 正是 selective grounding/CRS 要超越的對照組。
- **引用優先級**：中（REC 演進敘事 + SOTA 對照表用；非直接競品，無切割壓力）

---

## 第四批：方法骨幹 + 新 base + framing + benchmark（2026-06-19 進行中）

挑選邏輯：把 reading-log「下一步候選」7 篇一次讀完 —— 方法骨幹補完(2)、新 base 標的(2)、framing/動機(1)、benchmark 延伸(2)。

### #17 RCPS — Distribution-Free, Risk-Controlling Prediction Sets
- **2101.02703v3**（JACM 2021，Bates/Angelopoulos/Lei/Malik/Jordan, Berkeley）→ [read-rcps-summary.pdf](read-rcps-summary.pdf) / [read-rcps-summary.md](read-rcps-summary.md)
- **角色**：CRS 所用 **LTT 的理論祖先**（同作者群）。方法章理論源頭。
- **核心**：(α,δ)-RCPS = holdout calibration + UCB（選整段右側信賴帶壓在 α 下的最小 λ），把任意黑箱改造成 PAC 風險受控集合預測器。**兩大假設＝單參數嵌套 + 損失單調**；Theorem 1 靠單調性把逐點收斂升級成資料驅動選 λ 有效（免 uniform convergence）。Remark 2：初始模型可來自不同分布，只要 calib/test 同分布。Bound：二元用 exact binomial、非二元有界用 WSR。
- **定位**：(1) 必引理論源頭（CRS PAC 型態繼承自此）；(2) Remark 2 正當化「不重訓 frozen base」；(3) **反襯為何要 LTT**——CRS 雙參數+非單調 recall 損失突破 RCPS 兩假設，故升級到 LTT（與 CRC-Non-Monotonic 論證合流）。
- **引用優先級**：高（方法章理論源頭必引）

### #18 SCRC — Selective Conformal Risk Control
- **2512.12844v2**（2026-04，Xu/Guo/Wei, NJIT）→ [read-selective-crc-summary.pdf](read-selective-crc-summary.pdf) / [read-selective-crc-summary.md](read-selective-crc-summary.md)
- **角色**：方法骨幹標的中**與 CRS 最直接相關**＝結構最近鄰居（雙門檻雙階段、棄答+risk 雙保證）。必引必切割。
- **核心**：兩階段——λ1 棄答 gate + λ2 集合大小。關鍵理論＝**conditional exchangeability（Lemma 1）**：選擇破壞 exchangeability，需 symmetric selection rule（門檻用 calib+test 對稱算）才能在被選子集恢復可交換，Theorem 2 給「條件風險 + 選擇覆蓋」雙保證。SCRC-T(exact/每點重算) vs SCRC-I(PAC/可部署)。
- **切割三軸**：(1) 跨 base factorization（SCRC 雙門檻同屬一模型 f/g；CRS 兩異質 frozen base + 2×2 ablation 證非 ensemble）；(2) 任務維度（封閉 K 類 softmax vs 開放詞彙 box grounding）；(3) 棄答語意（低信賴 reject vs no-target 正交軸）。
- **可借鏡（最高價值）**：conditional-exchangeability 修補——CRS OWL gate 棄答後校準 GD recall 同樣會破壞 exchangeability，SCRC 的對稱門檻修補可移植（對照 [[crs-redteam-p0-fixes]] 已修的 calib-only leakage）。feasibility check `m ≥ 1/α−1` 可用於高棄答率 robustness 討論。
- **引用優先級**：高（必引必切割，同期同源；切割比照 BCEA「concurrent 不同任務」）

---

## 累計閱讀總表（18 篇）

- 第一批（5）：Modeling Relationships、COPS-Ref、GREC、HieA2G、Zero-Shot True/False
- 第二批（5）：SeqCRC、LazyMCoT、BCEA、CRC Non-Monotonic、Conformal Instance-Seg
- 第三批（5）：InstanceVG、VIRO、Are-FM-Conformal、VLM-Calibration、VL-SAM-v3
- 補讀（1）：MAttNet（REC modular 經典 baseline，SOTA 雙區塊表 Trained 代表）
- 第四批（進行中）：RCPS、SCRC ✅；GD-1.5、DINO-X、Grounding-Hallucination、AgroVG、Ref-Adv（待續）

---

## 下一步候選（第四批剩餘）
- ✅ 方法骨幹補完：RCPS (2101.02703)、Selective CRC (2512.12844)
- ⏳ 新 base 實驗標的：**GD-1.5 (2405.10300, 首選)** / DINO-X (2411.14347)
- ⏳ framing/動機：Does Object Grounding Really Reduce Hallucination? (2406.14492)
- ⏳ benchmark 延伸：AgroVG (2605.22034, 跨域 GREC)、Ref-Adv (2602.23898, ICLR'26)
- 投稿前人工掃描：Google Scholar `"conformal referring"` / `"selective referring expression"`

## 浮現的論文動作（讀完 15 篇後）
1. **改措辭**：別主打「set ≥1 命中保證」（instance-seg+LTT 已有），改主打「語言 grounding 上 recall+abstention 跨 base 分工聯合校準」。
2. **方法章理論句**：用 CRC-Non-Monotonic (2602.20151) 證成「為何 LTT 而非純 CRC」。
3. **Related Work 點名切割**：SeqCRC（結構同源三軸）、LazyMCoT（conformal 平面正交）、BCEA（同調不同任務）、VIRO（abstention 軸三刀）、Conformal-Instance-Seg（修辭層非方法層）。
4. **SOTA 表**：雙區塊 + 「Guarantee?」欄（Trained 標 ✗ vs Frozen CRS 標 ✓）。
5. **最高 ROI 實驗**：加 GD-1.5 當第四 base（非 VL-SAM-v3）。
