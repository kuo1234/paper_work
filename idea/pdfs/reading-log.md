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

## 下一步候選（尚未讀）
- 新 frozen base 候選（強化 cross-base 主張，最高 ROI 實驗）：VL-SAM-v3 (2605.03456) / DINO-X (2411.14347) / GD-1.5 (2405.10300)
- framing/動機：Enabling Calibration in Zero-Shot VLM (2303.12748)、Are foundation models good conformal predictors? (2412.06082)、Does Object Grounding Really Reduce Hallucination? (2406.14492)
- benchmark 延伸：AgroVG (2605.22034, 跨域 GREC)、Ref-Adv (2602.23898, ICLR'26)
- 投稿前人工掃描：Google Scholar `"conformal referring"` / `"selective referring expression"`
