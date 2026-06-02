---
type: paper-note
aliases:
  - "OpenVLA"
year: 2024
stage: "5-zero-shot與generalist"
tags:
  - VLA
  - open-source
  - LoRA
  - baseline
  - backbone候選
  - 5-zero-shot與generalist
summary: "7B開源VLA（Llama2+DINOv2+SigLIP，970k OpenX），可LoRA微調當「弱語言→z」前端；當baseline非重訓。"
---
# OpenVLA: An Open-Source Vision-Language-Action Model

> Kim, Pertsch, Karamcheti et al., 2024 (arXiv:2406.09246v3)。Stanford / UC Berkeley / TRI / Google DeepMind / Physical Intelligence / MIT。
> 提出 70 億參數的「開源」VLA（Vision-Language-Action）模型，建立於 Llama 2 7B + DINOv2 + SigLIP 之上，並在 [[OpenX_Embodiment|Open X-Embodiment]] 的 97 萬條真機機器人軌跡上做 action prediction fine-tune；同時系統性地探討 LoRA fine-tune 與量化部署，是當下 generalist 機器人策略的開源代表作。

---

## 0. 閱讀總覽（白話）

OpenVLA 想解決兩件事：
1. **VLA 不能只是黑盒**：先前最強的 VLA（如 Google 的 [[RT2|RT-2]]、RT-2-X）權重、資料、訓練細節都不公開，社群難以延伸、難以做新機器人 fine-tune 的研究。
2. **VLA 要能被便宜地 fine-tune**：機器人實驗室通常只有少量 demonstrations 與消費級 GPU，需要一條「拿大模型→低成本適配新場景」的可行路徑。

論文的做法非常工程化、目標導向：
- 拿一個既有的 VLM（Prismatic-7B：DINOv2+SigLIP 雙視覺編碼 + MLP projector + Llama 2 7B 語言模型）。
- 把連續機器人 7 維 end-effector 動作 **離散化成 256 個 bin**，每維對應一個 token，覆蓋掉 Llama tokenizer 裡最少用的 256 個 token。
- 用 97 萬條 OpenX 機器人軌跡，標準 next-token prediction loss 做 supervised fine-tune（27 epoch，64 A100，14 天）。
- 推論時把預測 token decode 回連續動作直接送給機器人。
- 量化：4-bit 在效能上幾乎等同 bfloat16；LoRA fine-tune 只動 1.4% 參數就能逼近 full fine-tune。

實驗結論很直接：在 BridgeData V2 與 Google Robot 兩個 embodiment 上，OpenVLA 用 1/8 參數打敗封閉的 RT-2-X 16.5%；在 Franka 上 fine-tune 後也超過 Diffusion Policy（多任務、含 language grounding 時尤其明顯）。

對你（使用者）而言，這篇是「**語言 → robot action 的開源 backbone 代表作**」，是 robot play + 弱語言 + offline meta-RL 路線下，討論「為什麼不直接用 VLA」與「VLA 哪些東西可以拿來當 task-spec encoder」時，必須引用且可能要當 baseline 的論文。

---

## Abstract

- 主張：把網際網路規模 VL 預訓練的 prior 帶進機器人，最好的形式是「fine-tune VLM 來生成動作」（即 VLA）。
- 既有 VLA 的兩個痛點：封閉、無 fine-tune best practice。
- 提出 OpenVLA：7B 參數、開源、用 Open-X 970k 條 demonstration 訓練；以 Llama 2 為語言骨幹，視覺編碼用 DINOv2+SigLIP 融合特徵。
- 結果：
  - 多 embodiment generalist：在 29 個任務上比 55B 的 RT-2-X 高 16.5%（絕對成功率），但模型小 7 倍。
  - Fine-tune 表現：在多物件、多指令的場景（強 language grounding）上比 from-scratch Diffusion Policy 高 20.4%。
  - 部署：LoRA 與量化讓消費級 GPU 也能 fine-tune／serve。
  - 釋出模型權重、fine-tune notebook、訓練 codebase。

---

## 1. Introduction

**問題定位**
- 訓練好的機器人策略對 distractor、新物件、新指令 robustness 不足；但 CLIP/SigLIP/Llama 2 等基礎模型已展現出強 OOD 泛化能力。
- 機器人資料規模（最多 ~1M 軌跡）遠小於網際網路語料，因此要「把網路 prior 透過 VLM 帶進機器人」。

**為什麼是 VLA 而非別的整合方式**
- 過去整合方式包括：VLM 作為表徵 encoder、modular task planner、symbolic 工具等。
- VLA（如 RT-2）的特點是 **直接 fine-tune VLM 輸出機器人動作**，最大程度承接 VLM 的訓練 infrastructure 與泛化能力。

**RT-2 系列的兩個缺點**
1. 全部閉源、不公開資料 mixture 與 hyperparam。
2. 沒有提供「對新機器人、新任務」的高效 fine-tune 流程，特別是不能在消費級 GPU 跑。

**OpenVLA 的三大貢獻**
1. **新 SOTA generalist 策略**：970k 軌跡 + 雙視覺編碼 + Llama 2，在 29 個跨機器人任務上贏 RT-2-X 16.5%。
2. **首次系統性探討 VLA 的 fine-tune**：在 7 個 Franka 任務上比 [[Octo|Octo]] 與 Diffusion Policy 強，多物件多指令場景優勢最明顯。
3. **首次展示 LoRA + 量化讓 VLA 落到 consumer GPU**：8x compute reduction、bfloat16 與 int4 推論效能幾乎相等。
4. **全部開源**（模型、fine-tune notebook、PyTorch codebase，支援 Open-X 大規模訓練）。

---

## 2. Related Work（簡）

**VLM**
- 主流架構：vision encoder → projector → LLM，採「patch-as-token」風格。
- OpenVLA 採 Karamcheti et al. (2024) 的 Prismatic-7B：multi-resolution 視覺特徵（DINOv2 提供低層空間、SigLIP 提供高層語意）。

**Generalist robot policy**
- Octo 等代表作通常將語言 embedding 或視覺 encoder 與一個從零訓練的 transformer policy「拼接」。
- OpenVLA 走更 end-to-end：直接把動作當成 LLM token，從 VLM 一路 fine-tune 下來，scalability 更好。

**VLA**
- RT-2、RT-2-X、RFM-1、LINGO-2、3D-VLA 等。
- 與 RT-2-X 的差異：
  1. 結合更強的 open VLM + 更大 robot mixture，結果更好但小 1 個數量級。
  2. 系統性探討 fine-tune（RT-2-X 不支援）。
  3. 首個展示 PEFT + 量化的 VLA。
  4. 唯一開源 generalist VLA。

---

## 3. OpenVLA Model（方法與架構）

### 3.1 VLM 預備知識

OpenVLA 是直接拿 **Prismatic-7B** 當 backbone，三個元件（見圖 2）：
1. **Vision encoder（~600M 參數）**：DINOv2 + SigLIP 兩條 ViT 並行，把 patch 過完之後 **channel-wise concat**。DINOv2 帶來空間/低層特徵（對機器人控制很重要），SigLIP 帶來語意對齊。
2. **Projector（2 層 MLP）**：把視覺特徵映射到 LLM 的 token embedding 空間。
3. **LLM backbone（Llama 2 7B）**：純 decoder。

VLM 預訓練資料來自 LLaVA 1.5 mixture（約 1M image-text 樣本）。

### 3.2 Training Procedure（核心：動作 token 化）

**設計核心：把「動作預測」變成「下一個 token 預測」**
- 把 7 維機器人 end-effector 動作（Δx, Δθ, Δgripper）每一維 **獨立離散化到 256 bins**。
- Bin 邊界用訓練資料每一維的 **1%–99% quantile** 均勻分割（不用 min-max，避免 outlier 把 bin width 拉太寬）。
- 結果：每一條 (image, instruction, action) 樣本 → N=7 個整數 ∈ [0..255] → 拼成一個 token 序列。

**Tokenizer trick**
- Llama tokenizer 只預留 100 個 special token，不夠裝 256 個動作 bin。
- 做法：直接「覆蓋」Llama 詞表中最少用的 256 個 token，把它們重新解釋為動作 token。

**訓練目標**
- 標準 next-token prediction；cross-entropy loss 只算在「動作 token」這 7 個位置上（前面 image patches + instruction prompt 不算 loss）。
- Prompt 格式：「`What should the robot do to {task}? A:` 」+ 動作 token。

### 3.3 Training Data：970k OpenX 軌跡

- 來源：Open X-Embodiment（70+ 個 dataset、2M+ 軌跡）。
- 兩個目標：
  1. **輸入/輸出 space 一致**：只留 ≥1 個第三人稱攝影機 + single-arm end-effector control 的 dataset。
  2. **embodiment / task / scene 平衡**：沿用 Octo 的 mixture weight，下調過於單一的 dataset。
- DROID 一度以 10% 權重納入，但 action token accuracy 始終偏低，最終第三輪訓練移除。

### 3.4 Design Decisions（重要工程經驗）

| 決策 | 選擇 | 原因 |
|---|---|---|
| VLM backbone | Prismatic（>LLaVA>IDEFICS-1） | 雙視覺編碼的空間推理優勢；單物件下 LLaVA≈IDEFICS，但多物件 language grounding 時 LLaVA +35%、Prismatic 再 +10% |
| 影像解析度 | 224×224（vs 384×384） | VLA 上看不到 resolution 提升的好處，但 384 訓練時間 3 倍 |
| Vision encoder 是否凍結 | **解凍**（與一般 VLM 經驗相反） | 機器人控制需要細粒度空間細節，需讓 vision encoder 隨機器人資料微調 |
| Epoch 數 | 27 epoch（vs VLM 1–2 epoch） | 機器人 action token 準確率要到 95% 以上才能跑出好策略 |
| 學習率 | fixed 2e-5，無 warmup | sweep 後最佳；和 VLM 預訓練同 lr |

### 3.5 Infra

- 訓練：64 張 A100、14 天、批次 2048、共 21,500 A100·hours。
- 推論：bfloat16 下 15GB GPU mem，RTX 4090 大約 6Hz。
- 釋出 **遠端 inference server**：機器人端不需要強 GPU，遠端 stream action 過來即可。

---

## 4. Codebase（一句話）

PyTorch、模組化、支援 AMP / FlashAttention / FSDP / HuggingFace AutoModel / LoRA / 量化推論，從單卡 fine-tune 到多節點預訓練 VLA 都涵蓋。

---

## 5. 實驗

### 5.1 跨機器人 out-of-the-box 評測

**設定**
- WidowX（BridgeData V2）+ Google Robot（[[RT1|RT-1]]/RT-2 用的 mobile manipulator）。
- 評測軸：visual / motion / physical / semantic generalization + language conditioning（多物件、指令指定目標）。
- BridgeData V2 共 170 rollouts（17 任務×10），Google Robot 60 rollouts（12 任務×5）。

**對手**
- RT-1-X（35M, 從零訓練）
- Octo（93M, 從零訓練，open-source SOTA）
- RT-2-X（55B, 閉源 VLA SOTA）

**主要結果**
- OpenVLA 在 BridgeData V2 顯著贏 RT-2-X，Google Robot 與 RT-2-X 持平；總體 +16.5%。
- RT-2-X 在 semantic generalization（網路概念遷移）上略勝，OpenVLA 在其他軸（visual / motion / physical / language grounding）都贏或持平。
- 質性觀察：OpenVLA / RT-2-X 都展現出「接近正確物件」、「end-effector 對齊物件方向」、「抓不穩會重抓」等行為，RT-1-X 與 Octo 常常亂揮。

**贏 RT-2-X 的原因（作者歸因）**
1. 訓練資料更大且更乾淨（970k vs 350k，並過濾掉 BridgeV2 全零 action）。
2. DINOv2+SigLIP 融合 vision encoder。
3. 詳細 cleanup（Appendix C/D 的消融）。

### 5.2 對新機器人的 fine-tune

**設定**
- Franka-Tabletop（5Hz）、Franka-DROID（15Hz）。
- 每個任務 10–150 demonstrations。
- Full fine-tune（5.2 節）vs PEFT（5.3 節）。

**對手**
- Diffusion Policy（從零訓練 SOTA）
- Diffusion Policy (matched)：輸入輸出規格與 OpenVLA 對齊
- Octo（fine-tune 過）
- OpenVLA (scratch)：拿 base Prismatic VLM 直接在 target dataset 上 fine-tune，不經過 OpenX 預訓練 → 作為「pretraining 是否真有效」的對照組

**結果**
- 窄、單一指令任務：Diffusion Policy 仍很強（dexterous、軌跡平滑、有 action chunking）。
- 多物件、多指令、需 language grounding：OpenVLA / Octo > Diffusion Policy。
- **OpenVLA 是唯一在所有 task 上都 ≥50% 成功率的方法**，OpenX 預訓練的好處在 OpenVLA (scratch) 的較差結果中可看到。
- 結論：OpenVLA 適合當「downstream task 的 default」，特別是有多樣語言指令時。

### 5.3 Parameter-Efficient Fine-Tuning

策略比較（Franka-Tabletop）：

| 策略 | 成功率 | 可訓參數 | VRAM (bs=16) |
|---|---|---|---|
| Full FT | 69.7% | 7,188M | 163.3 GB* |
| Last layer only | 30.3% | 465M | 51.4 GB |
| Frozen vision | 47.0% | 6,760M | 156.2 GB* |
| Sandwich (vision + embed + last) | 62.1% | 914M | 64.0 GB |
| **LoRA r=32** | **68.2%** | **97.6M** | **59.7 GB** |
| LoRA r=64 | 68.2% | 195M | 60.5 GB |

要點：
- 凍結 vision encoder 或只動最後一層都不行，視覺仍需 adapt。
- LoRA 只訓 1.4% 參數即可逼近 full fine-tune，**單張 A100 10–15 小時** 就能 fine-tune 完一個任務（vs full fine-tune 約 8 倍 compute）。
- LoRA rank 對結果影響小，預設 r=32。

### 5.4 量化部署

| Precision | Bridge 成功率 | VRAM |
|---|---|---|
| bfloat16 | 71.3% | 16.8 GB |
| int8 | 58.1% | 10.2 GB |
| **int4** | **71.9%** | **7.0 GB** |

- int8 變慢（1.2Hz）導致系統動態不匹配，反而掉效能。
- int4 在 A5000 可跑 3Hz，接近 5Hz 訓練時的系統動態，成功率與 bfloat16 持平，記憶體不到一半。
- token-level 準確率三種精度其實都接近，差異主要來自推論速度造成的閉迴路 dynamics 變化。

---

## 6. Discussion & Limitations

1. **僅支援單張影像** —— 沒有多視角、本體 proprioception、observation history；未來可考慮用支援 interleaved 圖文的 VLM。
2. **推論速度不夠快** —— 目前 ~6Hz，不足以支撐 ALOHA 等 50Hz 雙手控制；可引入 action chunking、speculative decoding。
3. **可靠度仍不夠高** —— 多數任務 < 90% 成功率。
4. **設計空間仍未充分探索**：VLM 大小對 VLA 影響？是否需要與網路圖文資料 co-training？哪種視覺特徵最適合？這些都是希望開源後社群一起回答。

---

## 與本研究主線的關聯

你的主線：**robot play 蒐集低結構資料 → 極少弱語言標籤 → offline meta-RL（[[PEARL|PEARL]] / [[VariBAD|VariBAD]] 風格的 task embedding）→ 文字 / task spec 做 zero-shot meta-learning**。OpenVLA 與這條線的接觸點如下。

### 1. VLA 的「輸入 / 輸出設計」對你研究的具體意涵

**輸入端**：image (224×224) + 任務字串 prompt（「`What should the robot do to {task}? A:`」）。
- 對你而言重點是：**它示範了「自然語言 task 字串 → 直接驅動策略」的工程可行性**。意味著如果你要把 task 字串當 zero-shot specification，這個介面格式幾乎是業界事實標準。
- 你的「弱語言標籤」如果想成為 OpenVLA-style 的 task embedding，最直接的做法是把它送進語言端（Llama 2 7B）拿 hidden state，再對齊到你 meta-RL 的 task latent。

**輸出端**：把 7 維連續動作離散化成 7 個 token（256 bin/維），用 next-token prediction loss。
- 這與你 offline meta-RL 中 policy 的設計**不一定相容**：PEARL 風格通常是連續 actor + state-conditioned policy + 上層 task latent z。
- 但有兩個啟示：
  1. **「discrete action token」是個低成本可遷移介面**。如果你的 robot play 資料粒度也離散化成相同格式，那等於可以借用 OpenVLA 的整條訓練 pipeline。
  2. **跨 embodiment 的 trick**就是「動作 token 強迫對齊到同樣的詞表」，這給 cross-embodiment offline meta-RL 一個非常具體的工程模板。

### 2. 哪些部分可被拿來當「語言 → task spec / embedding」的 backbone？

- **理想 backbone candidate**：
  - Prismatic VLM（DINOv2+SigLIP+Llama 2）的「**多模態特徵 → 語言空間**」這部分（即 vision encoder + projector + LLM 的前半段隱表徵），可以直接當 **「(觀察, 弱語言) → task embedding z」** 的 encoder。
  - 具體做法：取 LLM 在 task 字串最後幾個 token 的 hidden state（或 instruction 經 average pooling），作為 PEARL 中 q(z|context) 的「字串支持版本」。
- **較不適合直接搬的部分**：
  - 動作 token decoder + 整個 Llama 2 7B autoregressive 解碼是為了 supervised imitation 設計，跟你的 offline meta-RL 目標（policy improvement、context-conditioned exploration）關係較淡，硬接會 over-parameterize 並且犧牲 meta-RL 的 sample-efficient 特性。
- **LoRA 經驗的價值**：論文證明在這種規模下 **LoRA rank=32 就足夠**，這直接告訴你：如果未來想接 OpenVLA 當 encoder，並在 robot play 上 task-conditioned fine-tune，**幾百萬參數的 adapter 即可**，不必 full fine-tune。

### 3. 該把 OpenVLA 當「比較對象」還是「從頭重訓」？

明確結論：**應該把 OpenVLA 當比較對象（baseline / strong upper bound），不應該自己從頭再訓 7B VLA**。原因：

1. **規模差距**：OpenVLA 用 64×A100×14 天 + 970k 軌跡。你的研究核心是 robot play + 弱語言 + offline meta-RL 的設計創新，不應該把預算花在重複預訓練。
2. **它已經開源**：模型權重、PyTorch codebase、LoRA + 量化都釋出，重訓不會有 marginal contribution。
3. **它的弱點正好是你的賣點**：
   - OpenVLA 仰賴大量人類示範 demonstration（97 萬條），**無法處理 robot play 這種無語言或弱語言的資料**。
   - OpenVLA 沒有「zero-shot 新 task」的機制；它的「泛化」其實仍仰賴語意上有出現過的概念。
   - OpenVLA 缺乏 task posterior / belief update 概念，無 meta-RL 的 exploration 行為。
   - → 你的論文應該定位為「**極少語言下，offline meta-RL 比 imitation-style VLA 更 sample-efficient、更 zero-shot**」。

4. **可實際採用的整合策略**：
   - **Baseline A**：OpenVLA 直接 zero-shot（不 fine-tune）→ 表現的是「沒看過你的 embodiment 的 generalist」。
   - **Baseline B**：OpenVLA 用你 robot play 的 (subset, with weak language label) 做 LoRA fine-tune → 模擬「直接走 supervised 路線可以走多遠」。
   - **你的方法**：用 PEARL/VariBAD 風格的 task latent + 弱語言條件，凸顯在 zero-shot 新任務上的優勢；同時把 OpenVLA 的 LLM 拿來當 weak-language → z 的 encoder（不訓練／LoRA 微調）。

### 4. 對 meta-learning × zero-shot 並行方向的具體影響

- OpenVLA 的「token 化動作 + Llama 詞表」可作為一個跨 embodiment 介面，**但無法處理 zero-shot novel 任務**（它只能 condition on 訓練見過的指令）。
- 你的方向（task embedding + posterior over z）正好填補這塊：**把 OpenVLA 的語言端當 backbone**，但讓 z 由 meta-RL 的 context（最少量 trajectory + 弱語言）推斷，就同時拿到「網路語言泛化」與「offline meta-learning 的 task-shot 適應」。

---

## 一句話總結

OpenVLA 把 Llama 2 7B + DINOv2+SigLIP 的 VLM 透過「動作 token 化 + 97 萬條 Open-X 軌跡 supervised fine-tune」變成開源 SOTA 的 generalist 機器人策略，同時驗證 LoRA + 量化可在消費級 GPU 落地，因此對你而言它是「**語言到動作介面的開源工程標竿，應拿來當 baseline 與語言端 backbone，但其純 imitation 路線正是你 robot-play + 弱語言 + offline meta-RL + zero-shot task-embedding 要超越的對象**」。
