---
type: paper-note
aliases:
  - "RT-2"
  - "RT2"
year: 2023
stage: "5-zero-shot與generalist"
tags:
  - VLA
  - vision-language-action
  - web-knowledge
  - baseline
  - 5-zero-shot與generalist
summary: "VLA：action當text token，co-fine-tune VLM於web+robot data，遷移web知識到robot semantics。"
---
# RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control 精讀筆記

> Brohan et al., Google DeepMind, CoRL 2023
> 直接把大型 VLM (PaLI-X / PaLM-E) 透過「把 robot action 編成 text token」的方式 co-fine-tune 於網路資料 + 機器人示範資料，得到一個端到端的 Vision-Language-Action (VLA) 模型，使網路尺度的語義與視覺知識能在低階閉迴路控制中直接生效，並湧現出對符號、推理、多語、人物識別等的零樣本理解能力。

---

## 0. 閱讀總覽（白話）

這篇是一篇「系統 + 配方」型論文，核心想法很簡單但震撼：

- 既然網路上訓出來的 VLM 已經懂「草莓配水果碗」、「拿石頭可以當鎚子」、「accidente的西語意思」這些語義常識，那就不要再額外設計什麼 high-level planner 餵 low-level controller 的兩段式系統，直接讓 VLM 自己輸出機器人動作。
- 怎麼做？把連續動作（6-DoF 末端位移 + 夾爪開合 + 終止旗標）離散化成 256 個 bin，再把這 256 個 bin 映射到 VLM 既有的詞表中的某些 token，這樣動作就「變成一種語言」，和自然語言走同一條 next-token prediction 通道。
- 訓練配方關鍵：co-fine-tune，亦即在 fine-tune 階段同時餵 robot data 與原本的 web VQA / captioning 資料，避免 VLM 把網路尺度知識忘掉。
- 結果：在已見任務上和 [[RT1|RT-1]] 持平，在未見物件 / 背景 / 環境的泛化測試上贏 RT-1 / VC-1 / R3M / MOO 約 2-6 倍，且湧現出符號理解（move apple to 3）、視覺推理（move apple to cup with same color）、多語（mueve la manzana al vaso verde）、人物識別（move coke can to Taylor Swift）、甚至 chain-of-thought（先輸出 plan 再輸出 action token）的能力。
- 一句話：RT-2 = 「Action-as-Language」的 VLM fine-tune 配方，第一次大規模驗證 web-scale 語義知識可以直接灌進閉迴路操控策略。

對使用者主線研究（robot play + 極少弱語言 + offline meta-RL + text→task spec/embedding 做 zero-shot）而言，這篇主要的價值是：
（i）證明 LLM/VLM 的世界知識足以讓 robot 在語義層做零樣本泛化；（ii）action tokenization 是把 RL/IL 跟 LLM 串起來的關鍵介面；（iii）告訴你「pretraining 語義 + 機器人資料」的混訓配方可行，但同時也指出 RT-2 沒解 task inference / meta-RL，這正是你的切入點。

---

## Abstract

- 問題：如何讓 internet-scale 訓練過的 VLM 直接進入 end-to-end 機器人控制，同時得到泛化與湧現的語義推理？
- 方法：對 SOTA VLM 同時 fine-tune 於機器人軌跡資料與網路尺度 vision-language 任務（VQA、captioning）。為了把「自然語言回答」與「機器人動作」塞進同一格式，作者把動作表示為 text token，與 language token 一視同仁。
- 命名：把這類模型統稱為 vision-language-action（VLA）模型，本論文的具體實例叫 RT-2。
- 評估規模：約 6,000 次真實機器人試驗。
- 重點結果：對新物件的泛化大幅提升、能解釋未在機器人資料出現過的指令（例如「放到某個數字 / icon 上」）、能做基礎推理（「拿最小 / 最大的物件」、「最接近某物的」），加上 chain-of-thought 後可做多步語義推理（例如「用什麼當鎚子？石頭」、「累的人喝什麼？能量飲料」）。

## 1. Introduction

- 大背景：LLM / VLM 已能做 fluent text、emergent reasoning、open-vocabulary 視覺辨識，這些能力對 generalist robot 都很有用，但 robot 不能直接套用，因為 robot 需要 grounded low-level action（例如 Cartesian end-effector command）。
- 既有路線一：硬蒐千萬筆 robot 資料 — 不現實。
- 既有路線二：用 LLM/VLM 當 high-level planner，把語意指令拆成 pick-and-place 等 primitive，再交給獨立的 low-level controller — 問題是底層 controller 不會「享受到」網路尺度的語義知識。
- 本論文核心問題：能否讓大型預訓 VLM 直接整合進低階機器人控制，從而提升泛化並湧現語義推理？
- 做法摘要：直接訓練一個本來做 VQA / dialogue 的 VLM，使其同時輸出 low-level robot action token。將 action tokenize 成 text token，組成「multimodal sentences」，輸入是相機影像 + 任務描述，輸出是動作字串。
- 與既有 VLM-for-robot 方法（如 CLIPort）、從零設計新架構（如 Gato）的差異：RT-2 完全不增加新參數，直接讓既有 VLM 輸出 text-encoded action。
- 命名：VLA 模型；本論文的實例 RT-2，沿用 RT-1 的資料協定，但 backbone 換成大型 VLM。
- 觀察到的湧現能力：物件泛化、語義變化指令泛化、把 robot data 中的 pick-and-place 技能重新部署到語義指向的位置（如「放到 3 上」）、物件關係理解（決定誰要放哪）、CoT 推理（鎚子 = 石頭、累 → 能量飲料）。
- 主要貢獻：RT-2 模型家族（最大 55B 參數），6,000 次評估顯示物件 / 場景 / 指令泛化大幅改善，並繼承 web-scale 預訓帶來的湧現能力。

## 2. Related Work（精簡）

- VLM 兩大類：（1）表徵學習型（CLIP）；（2）{vision, text} → {text} 生成型。本論文聚焦第二類，因為要利用其「自由生成」的格式來直接吐出 action token。
- Robot 泛化：歷來嘗試從大規模多樣資料學到對新物件、新組合、新指令、新語義類別、新環境的泛化。本論文目標是「在所有這些軸向上同時泛化」，靠的是「比機器人經驗更廣的 pretraining」。
- Pre-training for manipulation：過去多半只用 pretrained vision encoder（ImageNet、R3M、VC-1）或 pretrained language encoder 做 instruction encoding 或 high-level planning。RT-2 直接用 pretrained VLM 而且「動作與語言走同一個輸出空間」，模型權重在語言與動作任務之間完全共享，沒有 action-only 的 layer，這是與 CLIPort / MOO 的關鍵差異。

## 3. Vision-Language-Action Models（方法核心）

### 3.1 Pre-Trained Vision-Language Models（基座）

- 用兩個既有 VLM：PaLI-X（multilingual vision-language model，ViT-22B + 32B encoder-decoder UL2-like backbone）與 PaLM-E（embodied multimodal decoder-only LLM，把影像投到 language embedding 空間）。
- 對應的 VLA 版本：RT-2-PaLI-X 與 RT-2-PaLM-E，規模從數十億到 550 億參數。

### 3.2 Robot-Action Fine-tuning（最關鍵的工程設計）

**Action space**
- 6-DoF 末端 translation 位移 + 6-DoF rotation 位移 + 夾爪 extension + 一個離散 episode termination 旗標，共 8 維。
- 連續維度全部 uniform 離散化成 256 bin，每維變一個離散整數，整個動作可用 8 個整數表示。

**Action tokenization：把動作映射到 VLM 既有 token**
- 需要在 VLM tokenizer 中保留 256 個 token 當作 action token。挑選方式依 backbone 而異：
  - PaLI-X：tokenizer 中本來 1000 以下整數每個都有專屬 token，於是直接拿「對應整數的 token」當該 action bin 的 token。
  - PaLM-E：沒這種便利，作者直接覆寫「最少使用的 256 個 token」當作 action vocabulary。這實際上是 symbol tuning 的一種形式（Wei et al. 2023）。
- 動作 string 格式：terminate Δposx Δposy Δposz Δrotx Δroty Δrotz gripper，例如 "1 128 91 241 5 101 127"。
- 訓練輸入仿照 VQA：`Q: what action should the robot take to [task instruction]? A:`，輸出即上述動作字串。

**Co-Fine-Tuning（配方核心）**
- 不只用 robot data 來 fine-tune，而是 robot data 與原本的 web 資料一起混訓（co-fine-tune）。
- 動機：避免遺忘預訓階段學到的抽象視覺與語義概念，並讓模型在 fine-tune 階段同時接觸 web concepts 與 low-level actions。
- 操作：增大 robot data 的 sampling weight，使其在 batch 中比例放大（RT-2-PaLI-X 約 50%，RT-2-PaLM-E 約 66%）。

**Output Constraint**
- 推論時若 prompt 是 robot-action task，輸出 vocabulary 限定只能 sample action token，避免輸出無效字串；但對一般 VQA prompt 仍可輸出全部自然語言 token。

### 3.3 Real-Time Inference

- 最大的 RT-2-PaLI-X-55B 有 550 億參數，無法在桌機或機載 GPU 跑 closed-loop。
- 解法：部署在 multi-TPU 雲端服務，機器人透過網路 query 該服務。可同時服務多台機器人。
- 控制頻率：55B 版本 1-3 Hz，5B 版本約 5 Hz。
- 註：作者強調這是目前已知用於 direct closed-loop robot control 的最大模型（比過去大一個數量級以上）。

## 4. 實驗與 Emergent 能力

四個研究問題：
1. RT-2 對已見任務 / 對新物件、背景、環境的泛化表現？
2. 是否能觀察到並量測湧現能力？
3. 泛化如何隨參數量與設計選擇變化？
4. 能否做 chain-of-thought 推理？

實驗規模：約 6,000 次評估軌跡，使用 7-DoF mobile manipulator。

訓練資料：
- Web 端：來自 PaLI-X / PaLM-E 的原始 mixture（VQA、captioning、interleaved image-text），底層 WebLI 約 10B image-text，過濾後 1B。
- Robot 端：RT-1 蒐集的 mobile manipulator demo，13 台機器人、17 個月、辦公室廚房環境，每筆都有自然語言指令（動詞 + 物件名詞）。

基線：
- RT-1：35M transformer，純 robot data，無 VLM pretraining。
- VC-1：robotics-targeted visual foundation model（用 USE 加 language conditioning）。
- R3M：以人類活動影片學表徵，再接 RT-1 policy。
- MOO：用 VLM 標出「興趣物件」的彩色像素，但 VLM 本身的表徵不進入 policy；屬於「VLM 作為獨立模組增強感知」的代表。

### 4.1 已見任務與泛化（Q1）

- 已見任務：~200 個 instruction，pick / knock / upright / move / open & close drawer / put-into-drawer。RT-2 與 RT-1 在 seen tasks 上相當，其他基線略低。
- 未見泛化：新物件 / 新背景 / 新環境，又各分 easy / hard（hard = 更難抓的玩具、更不同的背景、辦公桌 vs. 廚房水槽等）。共 280 多個 instruction。
- 結果：RT-2 兩個變體在泛化測試上平均約 2× 領先 RT-1 / MOO、約 6× 領先 VC-1 / R3M。PaLM-E 版在較難的泛化場景表現好一些，PaLI-X 版在較簡單場景好一些，平均接近。
- 額外比較：Open Language-Table benchmark（Lynch et al. 2022 開源模擬環境），把 PaLI 3B co-fine-tune 後達 90% 成功率，相較 LAVA / RT-1 / BC-Zero 顯著進步。

### 4.2 湧現能力（Q2）

把湧現能力切成三大類：
- **Symbol understanding（符號理解）**：例如 "move apple to 3"、"push coke can on top of heart"。這些符號在 robot data 從未出現。
- **Reasoning（推理）**：視覺推理（同色配對）、數學（move banana near sum of two plus one）、多語言（西語、法語、德語指令）、邏輯（pick a healthy drink）。
- **Human recognition（人物識別）**：例如 "move coke can to the person with glasses"、"move coke can to Taylor Swift"。

用 A/B testing framework 嚴格比較 RT-2-PaLI-X、RT-2-PaLM-E vs. RT-1、VC-1。RT-2 平均比次佳基線 RT-1 高 3 倍以上。PaLI-X 版總體最強，但在 math reasoning 上 PaLM-E 版較佳（作者歸因於 PaLM-E 的 pretrain mixture 較強數學能力）。

### 4.3 規模與訓練策略消融（Q3）

- 比較 5B vs. 55B，比較三種訓練：from-scratch、純 fine-tune（只用 robot data）、co-fine-tune。
- 結果：
  - From-scratch 表現極差（連 5B 都崩），所以乾脆不再跑 55B 的 from-scratch。
  - Co-fine-tune > fine-tune > from-scratch，且優勢與模型規模一致。
  - 更大模型 → 更佳泛化（55B 平均 63%，5B fine-tune 42%）。
- 結論：「保留 VLM 原本的訓練資料一起 fine-tune」是避免遺忘世界知識的關鍵。

### 4.4 Chain-of-Thought 推理（Q4）

- 在 RT-2-PaLM-E 上以幾百步 fine-tune，引入 augmented data，要求模型先輸出 `Plan: ...` 自然語言、再輸出 `Action: ...` token。
- 範例：`Instruction: I'm hungry. Plan: pick rxbar chocolate. Action: 1 128 124 136 121 158 111 255.`
- 觀察：能執行更複雜的指令（"Bring me a drink"、"Pick the object that is different"、"I need to hammer a nail, what object might be useful → Rocks"）。
- 意義：把「VLM/LLM 當 planner」與「low-level policy」合二為一在同一個 VLA 模型，雛形可行。

## 5. Limitations（重要！）

- 雖然 VLM pretraining 大幅提升語義 / 視覺概念泛化，但不會帶來新的物理動作技能 — 模型能做的物理動作仍受限於 robot data 的 skill 分布；它只是學會用新方式「部署」這些技能。未來方向：用人類影片資料擴展技能空間。
- 計算成本高：55B 推論頻率僅 1-3 Hz，高頻控制場景可能成瓶頸。未來方向：量化、蒸餾、開源 VLM。
- 失敗案例（Appendix G）：抓物件特定部位（如把手）、新動作（擦拭、工具使用）、精細靈巧（折毛巾）、需多層間接推理。

## 6. Conclusions

- 提出 VLA 範式：把預訓 VLM 與機器人軌跡資料 co-fine-tune，動作以 text token 輸出。
- 兩個實例 RT-2-PaLI-X、RT-2-PaLM-E 都在泛化與湧現上顯著勝出。
- 機器人領域因此能直接受惠於 VLM 的進步。

---

## 為什麼 VLM/LLM 知識能幫 robot semantics（讀後重點）

- **語義不是 free**：「草莓屬於水果」、「3 是個數字」、「Taylor Swift 是個人」這類常識，要從機器人 demo 蒐到天荒地老都不會收斂；但網路圖文 1B 樣本中早就稠密分佈。
- **VLM 已把「視覺 grounding × 語言」對齊**：傳統 robotics 用 USE / CLIP encode 指令時，語言走獨立路徑，視覺與指令的關聯仍要從 robot data 重學；而 VLM 內部視覺 token 與語言 token 已在 attention 上充分混合，pick-and-place 的物件選擇可以直接借用 VLM 內建的「指涉解析」與「視覺推理」能力。
- **Output 空間統一帶來 weight sharing**：CLIPort、MOO 都把 VLM 當輔助模組，policy 仍有 action-only 層；RT-2 直接讓動作 token = 語言 token，整個模型在所有任務間共享，所以 web data 訓練留下的知識在每次 action 預測時都會「順便發揮」。
- **Symbol tuning 角度**：把「最少使用的 token」改寫成 action token 等同於 Wei et al. (2023) 的 symbol tuning 設置，已知這種重定義不太損害 VLM 既有能力，反而能提升 in-context learning。
- **Co-fine-tune 防遺忘**：若只用 robot data fine-tune，VLM 會 catastrophic forgetting 掉世界知識（消融顯示 co-FT 顯著勝出純 FT）。把 web data 留在 batch 裡，相當於「持續 anchor 在 pretraining manifold」，網路知識才能在新任務中被叫出來。

## Action tokenization 的意義

- **介面（interface）**：把連續控制信號轉成 256 個離散 token，是 LLM 路線與 RL/IL 路線最自然的接合點。任何沿著「LLM → 控制」方向的工作（如 [[OpenVLA|OpenVLA]]、π0、[[Octo|Octo]]）幾乎都採用變體。
- **解耦表達能力與輸出 head**：不用為動作另外設計 head（如高斯參數、Tanh-squashed Gaussian）、不用 BC loss 與 LM loss 分離；直接用 next-token CE。整個訓練管道、tokenizer、采樣策略、constrained decoding 都可以重用 LM 既有工具。
- **解析度與可控性**：256 bin / 維度的解析度足以做桌面 mobile manipulation；但對更精細任務（折毛巾、工具使用）可能不夠 — 這也對應論文的失敗案例與後續工作改用更高解析度或 continuous head 的 motivation。
- **與 multi-modal language 的同質性**：action token、image token、language token 三者同樣是 transformer 的離散輸入 / 輸出單位，意味著 chain-of-thought 可以「自然」把 plan 與 action 串在同一個序列裡，跨模態 reasoning 不需再加任何 architecture。
- **對 zero-shot / task spec 的啟示**：若 action 與 language 共享 token 空間，那「task embedding」也很容易被視為一段 prompt token；對使用者「文字→task spec/embedding 做 zero-shot」的主線而言，這暗示 task inference 模組的輸出最好能對齊到 LLM/VLM 的 token / embedding 空間。

## 為何說 RT-2 比較像 VLA（而不是傳統 VLM 或傳統 robot policy）

- 對比 VLM（如 PaLI-X / PaLM-E）：RT-2 額外保留 256 個 action token，並在 robot 任務中限制輸出 vocab，使其能直接執行閉迴路控制；同時仍能做 VQA / captioning，是「VLM 的超集」。
- 對比 RT-1 等傳統 robot policy：RT-2 不是從零學 visuo-language → action 映射，而是 reuse VLM 內部已對齊的語義空間；其動作預測沿用 next-token LM loss，與 BC 等價但能與 language 任務共享所有參數。
- 對比 Gato 等「generalist agent」：Gato 為動作另設 token 與 head，且未顯式對齊到網路尺度語義；RT-2 是更「乾淨」的 VLM 後接 action token，能直接繼承 VLM 的能力。
- 對比 SayCan / Code as Policies 等 LLM-as-planner：那些只在高階規劃用 LLM，低階仍用獨立 controller；RT-2 把規劃與執行合在同一網路，CoT 變得自然。

## 與本研究主線的關聯

使用者主線：robot play 資料 + 極少弱語言標註 + offline meta-RL + 用文字 → task spec / embedding 來做 zero-shot；關注 meta-learning × zero-shot 的並行方向。RT-2 是這條路上必須讀的「上界 baseline」與「設計參考」，其關聯與差異如下：

1. **它示範了 web-scale 語義知識的可遷移性**：當前 robot 領域對於「文字→task 表徵→新任務」的最大瓶頸是語義 grounding；RT-2 用 6k 次真機評估證明，只要 backbone 是 VLM，語義 zero-shot 幾乎是免費的（不需要為新指令收 demo）。這直接背書了你主線中「文字→task spec/embedding」這個介面的合理性。

2. **動作 tokenization 是你應該直接借用的介面**：使用者的 offline meta-RL 若要與 LLM/VLM 串接（無論是 task spec、reward signal、或 policy 表達），把 action 編成 LM token 是最簡單的方式。即使主線最後不用 transformer policy，至少 task spec / context 一定要走 LM token / embedding 路線，才能享受 VLM/LLM 的世界知識。

3. **RT-2 沒有解的是 task inference**：RT-2 假設 instruction 是顯式、準確、單一句的；它根本沒處理「instruction 含糊、弱、需從幾步互動推斷」的情境，也沒做 meta-learning（policy 不會在 episode 內快速適應）。使用者的差異化點正在此：robot play + 弱語言的設定下，需要一個 task inference / meta-RL 模組從少量上下文（影像、play、稀疏弱描述）推出 task embedding，再餵給 (RT-2-like) policy。所以使用者的「新意」應該放在 task inference module（meta-learned posterior over tasks，類似 [[VariBAD|VariBAD]] / [[PEARL|PEARL]]，但 task embedding 對齊到 LM 空間），而不是再做一次 RT-2。

4. **可當 baseline / oracle**：在使用者的實驗中，RT-2-like VLA policy（或開源的 OpenVLA）可以作為「強 instruction policy」的 oracle baseline — 給它完美 instruction 時的表現就是 zero-shot 上限；使用者要驗證的，是當 instruction 缺失或極弱時，用 task inference 從 play data 推出的 task embedding，能否逼近這個 oracle 的表現。

5. **Co-fine-tune 配方可移植**：使用者若要做「用文字描述當 task spec 來做 zero-shot 新任務」，幾乎一定要 co-fine-tune（保留 LM/VLM 的 pretrain mixture），否則語言空間會在少量 robot data 上塌掉，zero-shot 失敗。RT-2 的消融把這件事說得很清楚（純 fine-tune 比 co-FT 差約 10-20 個百分點）。

6. **限制提示研究空間**：RT-2 自承無法學新動作技能，這對使用者反而是好消息 — 「offline meta-RL on play data」原本的賣點之一就是能 stitch / combine 既有 skill。換言之，使用者方向（task inference + skill combination from play）正好補 RT-2 的缺口。

7. **Meta-learning × zero-shot 的銜接**：把 RT-2 視為「沒有 task uncertainty」的極限版 zero-shot，使用者主線可表述為「在 VLA 框架上加入 task posterior 與 belief 更新」。具體做法上：(i) 仍以 VLA 為 policy 骨幹；(ii) 在 VLA 的 prompt 中插入 meta-learned task embedding（連續 token）；(iii) 把 zero-shot 改成 few-shot adaptation — 用 episode 初期的 play 觀察更新 task posterior，再 decode action。這條路在實驗設計上有清楚 ablation：對比「給完美 instruction（RT-2 上限）」、「給弱 instruction」、「不給 instruction 但給 play context」。

## 一句話總結

RT-2 把動作離散化為 text token、用 co-fine-tune 把 web 知識保留在 VLM 中，第一次大規模證明「Vision-Language-Action 模型」這個範式能讓 internet-scale 語義直接驅動低階閉迴路控制，並湧現出符號、推理、多語、人物識別等 zero-shot 能力——但它沒解 task inference 與新動作技能，這正是使用者主線「offline meta-RL × text→task embedding」的切入點。
