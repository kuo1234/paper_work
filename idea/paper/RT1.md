---
type: paper-note
aliases:
  - "RT-1"
  - "RT1"
  - "Robotics Transformer"
year: 2022
stage: "5-zero-shot與generalist"
tags:
  - generalist-policy
  - transformer
  - scale
  - baseline
  - 5-zero-shot與generalist
summary: "大規模真機資料的generalist policy（USE+FiLM-EfficientNet+TokenLearner+Transformer→離散action）。"
---
# RT-1：Robotics Transformer for Real-World Control at Scale

> **標題**：RT-1: Robotics Transformer for Real-World Control at Scale
> **作者**：Anthony Brohan、Noah Brown、Chelsea Finn、Karol Hausman、Sergey Levine 等（Robotics at Google / Everyday Robots / Google Research, Brain Team；作者依字母順序排列）
> **出處與年份**：Robotics: Science and Systems (RSS) 2023, Daegu
> **主題**：以大規模、跨任務的真實機器人示範資料訓練單一 Transformer 策略，將自然語言指令、影像歷史與機器人動作三者放進同一個 sequence model，追求在新任務、新物件、新環境下的零樣本泛化以及對真實場景的穩健性。
> **整理目標**：以工程系統論文的角度精讀 RT-1：架構（USE 文字嵌入 + EfficientNet-B3 + FiLM 條件化 + TokenLearner + decoder-only Transformer + 離散化動作 tokens）、資料規模（13 台機器人、17 個月、約 130k 軌跡、700+ 指令）、實驗（seen / unseen / 干擾物 / 背景 / 長視角 SayCan / 跨機器人遷移）、以及它跟我自己研究主線（robot play + 弱語言 + offline meta-RL + 文字→task spec 做 zero-shot；以及 meta-learning × zero-shot 並行方向）的差別與可重用之處。

---

## 0. 閱讀總覽（白話）

RT-1 是 Google 在 2023 年提出的「機器人版 GPT-like 策略」。它的目標很直接：把 NLP、CV 在 2018-2022 學到的一個經驗——「資料量＋多樣性＋高容量模型 = 泛化」——搬到真實機器人控制上。

論文的核心主張可以濃縮成三句：

1. 機器人也吃「資料規模＋多樣性」這套：他們用 13 台 mobile manipulator 在 17 個月內收集了約 130k 條人類示範，涵蓋 700+ 條語言指令（picking、placing、開關抽屜、拿衛生紙、開玻璃罐等）。
2. 用 Transformer 把語言、影像、動作打成同一條 token 序列，但因為要在真實機器人上 3 Hz 即時推論，他們刻意設計一個「不大但夠用」的 35M 參數架構（FiLM-EfficientNet + TokenLearner 把 token 數壓到極少，再餵給 decoder-only Transformer 輸出離散化動作 token）。
3. 在 3000+ 次真實機器人試驗下，RT-1 在 seen tasks 達 97%、unseen 76%、干擾物 83%、背景 59%，且能吸收模擬資料、甚至吸收另一種型號（Kuka）的資料而不掉效能。資料多樣性比資料總量更重要。

對我來說重點不是看公式（這篇幾乎沒有理論推導），而是看：(a) 它怎麼設計「instruction → 動作」的編碼管線；(b) 它的資料規模與多樣性如何換算成泛化；(c) 它的「imitation + 離散動作分類」框架跟我關心的 offline meta-RL／zero-shot task embedding 差在哪。

一句話定位：RT-1 是 robotics 領域的「規模化 generalist policy」原型機，不關心 task inference／信念狀態這類 meta-RL 的議題，但是它做出的 backbone 與資料管線非常適合當我未來研究的 baseline 或 pretrained encoder。

---

## Abstract

論文主張：CV、NLP、語音等領域已經證明「task-agnostic 大規模預訓練 + 高容量架構」能帶來 zero-shot 或 few-shot 泛化，但 robotics 還沒有同等規模的範例。作者認為 robotics 要做到這件事的關鍵就是：(1) 開放式、任務無關的訓練；(2) 高容量、能把多樣化機器人資料「吸進去」的架構。本文提出 RT-1 模型族，並以大規模真實機器人實驗驗證它隨資料量、模型大小、資料多樣性的 scaling 行為，比較不同模型在泛化上的差異。

---

## I. Introduction（精讀）

### 核心問題

傳統 end-to-end 機器人學習（無論 imitation 或 RL）都是「為某個特定任務蒐特定資料、訓特定模型」。這跟 CV/NLP 早期一樣是 siloed 的小規模套路。CV/NLP 已經轉向「廣泛資料 + 大模型 + 預訓練」，那 robotics 為什麼還沒？因為機器人資料貴、難收，而且真實世界的多樣性高。作者問的核心問題是：

> **能不能訓練一個 single、能力強、多任務的 backbone，並像 NLP/CV 一樣享受 zero-shot 泛化？**

### 為什麼困難（兩個挑戰）

1. **資料挑戰**：要泛化就必須在「規模 × 廣度」上都夠，而且任務之間要有結構相連，模型才能在「相似 task 之間發現 pattern、再以新方式重組」。他們最後動用 13 台機器人 + 17 個月 + 約 130k 條 episode + 700+ 任務的資料蒐集規模。
2. **模型挑戰**：Transformer 有 capacity 但慢；機器人需要實時控制（這裡定為 3 Hz、< 100 ms 推論）。RT-1 的設計重點就在於把高維輸入輸出（影像、語言、馬達指令）都壓成 compact 的 token，讓 Transformer 可以即時跑。

### 核心貢獻

- 提出 RT-1 架構，把 image / instruction / action 都 tokenize，丟進 decoder-only Transformer。
- 在 700+ 真實指令、3000+ 真實試驗下，達到 97% seen / 76% unseen / 83% distractor / 59% background。
- 在 SayCan 框架下可執行 50 步的超長視角任務。
- 證明 RT-1 能吸收 simulation 資料、甚至吸收 Kuka 機器人（不同型態）資料，原任務效能幾乎不掉，且 bin-picking 評估翻倍提升（22% → 39%）。

### 為什麼這對 robotics 是個關鍵敘事

過去 robotics 一直有種「我們資料不夠多、所以我們需要 sample-efficient 的方法（meta-RL、model-based、sim2real…）」的論述。RT-1 正面挑戰這個論述：它說「不對，是我們資料還沒夠多、夠廣，補上之後 generalist policy 就會 work」。這是 robotics 從「巧妙演算法時代」往「資料 + scale 時代」轉軌的代表作之一，並為後續 [[RT2|RT-2]]、[[OpenX_Embodiment|Open X-Embodiment]] 等鋪路。

---

## II. Related Work（要點）

- **Transformer 策略**：許多先前工作用 Transformer 處理語言或多模態指令（CLIPort、PerceiverActor、VIMA、Gato 等）。RT-1 把這條路推到 real-world、大規模、即時控制。
- **語言條件化的機器人**：從 pipelined（語意解析 + 視覺 + 控制）到 end-to-end（BC-Z、Lynch & Sermanet 等）。RT-1 屬於 end-to-end，但任務數量與多樣性大幅超過先前工作。
- **大規模機器人資料集**：MIME、RoboNet、Bridge Data、BC-Z、Meta-World 等。RT-1 主張自己的資料集在 task 數、物件數、場景數、行為廣度上又上一個量級。
- **mobile manipulation**：過去多以強化學習為主，RT-1 證明大規模 imitation + Transformer 同樣可行。

關鍵差異化：Gato 雖然是「generalist agent」，但它真正的 real-world manipulation 只有單一 stacking 任務、且沒測泛化；RT-1 補上了大規模 real-world 任務寬度與泛化評估。

---

## III. RT-1 方法與架構（重點章節）

### A. 機器人與環境

- 平台：7-DOF 手臂、兩指夾爪、移動底座的 mobile manipulator。
- 場景：兩間真實 office kitchen（Kitchen1、Kitchen2）+ 一個仿造的「robot classroom」訓練場（部分流理台）。
- 評估時跨場景測效能與泛化（光線、背景、櫥櫃／抽屜配置都會變）。

### B. 模型架構（重點）

整體流程（圖 2，由上而下）：

`instruction(語言)` → USE 嵌入 → 透過 FiLM 注入 → 條件化的 EfficientNet-B3 → 影像 feature map → 展平成 81 個視覺 token → TokenLearner 壓到 8 個 token / 幀 → 6 幀 × 8 = 48 個 token 加上位置編碼 → decoder-only Transformer（8 self-attention 層、約 19M 參數） → 輸出離散化動作 token。

逐元件說明：

1. **語言編碼**：使用預訓練的 Universal Sentence Encoder（USE）把指令編成固定向量。RT-1 完全靠 USE 處理語言，不再對語言做進一步上下文建模。
2. **視覺編碼**：ImageNet-pretrained EfficientNet-B3，輸入 6 幀 300×300 影像，輸出 9×9×512 的 feature map。
3. **語言條件化（FiLM）**：把 USE 向量送進 identity-initialized FiLM 層，插入 EfficientNet 內部，讓「文字」能在早期就影響視覺特徵抽取（task-relevant features）。FiLM 初始化為單位映射，避免破壞 ImageNet 預訓練權重；訓練後逐漸偏離單位，把語言訊號融進去。FiLM-EfficientNet 加總約 16M 參數，輸出 81 個 token / 幀。
4. **TokenLearner**：一個 element-wise attention 模組，把 81 個視覺 token 軟性篩選成 8 個資訊密度高的 token，大幅減少 Transformer 要 attend 的長度，是 3 Hz 即時推論的關鍵。
5. **Transformer 主幹**：6 幀 × 8 token = 48 個 token（含位置編碼），餵入 decoder-only Transformer（8 層 self-attention，約 19M 參數），輸出動作 token。整體模型才 35M 參數，對 LLM 而言很小，但對 robot policy 而言已經是高容量。
6. **動作離散化**：動作維度共 11 維 = 7 手臂（x, y, z, roll, pitch, yaw, gripper 開合）+ 3 底盤（x, y, yaw）+ 1 mode（控制手臂／底盤／終止）。每個維度均勻離散成 256 個 bin。每一維被當成 categorical 分類問題輸出。
7. **損失函數**：標準 categorical cross-entropy + causal masking，等同 behavior cloning 損失。沒有 RL、沒有 reward、沒有任何 task-inference loss。

### 推論速度設計（很關鍵的工程點）

需求：人類完成相應指令大約 2-4 秒，所以 policy 要 ≥ 3 Hz、< 100 ms 推論。為此他們：

- **TokenLearner 壓 token**：81 → 8，推論加速 2.4x。
- **滑窗 token 重用**：相鄰時間窗口共用大部分歷史 token，只算新一幀，再加速 1.7x。

這兩個技巧讓 35M 參數模型在真機上能 3 Hz 跑起來，這是 RT-1 不是「跑不動的玩具」的關鍵。

### C. 資料（700+ 任務）

- 規模：約 130k 真實示範 episodes、13 台機器人、17 個月、若干 office kitchen「classroom」。
- 任務分組（skills，見論文 Table I）：
  - Pick Object：130
  - Move Object Near Object：337（最大宗）
  - Place Object Upright：8
  - Knock Object Over：8
  - Open / Close Drawer：各 3
  - Place Object into Receptacle：84
  - Pick from Receptacle and Place on Counter：162
  - 真實長視角任務專用技能（拉衛生紙、開玻璃罐、拿勺子等）：9
  - 總計：744 個 instruction。
- 「task」定義刻意以「instruction = 動詞 + 名詞片語」計，因為機器人領域沒有公認的 task 邊界定義。
- 蒐集流程刻意 modular：增任務就增 instruction、增資料，不需要架構或損失改動，便於規模化。

---

## IV. 資料與規模：為什麼「資料規模在 robotics 也開始變重要」

這篇文章是 robotics 史上把「資料量 × 多樣性」與「真實泛化」綁起來最嚴謹的實驗之一。重點：

1. **任務寬度 = 泛化的基礎**：BC-Z（≈100 任務）與 Gato（單一 stacking task）在新任務上明顯掉很多，RT-1（700+）才有 76% unseen。
2. **資料多樣性 > 資料數量**（IV-E 節的 ablation 圖 6 的核心結論）：
   - 把資料量砍到 51%／37%／22.5%（同樣 task 數），效能與泛化緩慢下降。
   - 但若保留 97% 資料、把 task 砍到 75%（即「資料多、但種類窄」），泛化掉得**比砍掉 49% 資料還快**。
   - 結論：對 robotics 來說，「再多收一個新 task」比「在既有 task 多收一些」更值錢。
3. **異質資料可吸收，但要有公共結構**：
   - 加入模擬資料：原任務效能不掉（92% → 90%），但「只在模擬看過的物件」效能從 23% → 87%（+64），跨領域遷移驚人。
   - 加入 Kuka QT-Opt 的 209k bin-picking 資料（不同機器人型態 + 不同動作分佈 + RL agent 蒐集）：原任務 92% → 90%，bin-picking 任務 22% → 39%（接近翻倍）。但若只用 Kuka 資料訓練在 MM 機器人上跑，是 0%。說明 RT-1 不是學了 Kuka 的動作，而是把兩者的共有結構抽出來、在 MM 的動作空間中重組。

這些結果為什麼意義重大：robotics 過去普遍認為「資料貴、樣本少」要靠 sample-efficient 演算法（meta-RL、model-based、bayes、causal 等），RT-1 提供強力對照證據：當你能弄到夠廣的資料，單純 imitation + Transformer 就能達到先前需要複雜演算法才能逼近的泛化。這是「scaling hypothesis」在 robotics 上的第一個正面案例之一，直接推動後續 RT-2、Open X-Embodiment、π0 等工作。

---

## V. 實驗與泛化結論

### 整體效能（Table II，與 baseline 比較）

| 模型 | Seen | Unseen | Distractors | Backgrounds |
|---|---|---|---|---|
| Gato | 65 | 52 | 43 | 35 |
| BC-Z | 72 | 19 | 47 | 41 |
| BC-Z XL | 56 | 43 | 23 | 35 |
| **RT-1** | **97** | **76** | **83** | **59** |

注意：所有 baseline 都用了 RT-1 的資料集，這樣才公平。RT-1 在每一項都顯著贏，且 Seen 接近飽和。

### 真實 kitchen 場景的多軸泛化（Table III）

把任務分成 L1（新檯面 + 光線）、L2（再加未見干擾物）、L3（再加全新任務設定、新物件或位置如水槽附近）。RT-1 在三個 level 仍是最強；尤其 L3 仍有 50%，而 Gato 是 0%、BC-Z 約 50%。

### 異質資料融合

- **模擬資料**（Table IV）：對僅在 sim 出現的物件，效能從 23% → 87%；對「sim 物件 + 新 skill」也從 7% → 33%。原 real 效能僅微降。
- **跨機器人**（Table V）：加入 Kuka bin-picking 後，標準 classroom 92% → 90%，bin-picking 22% → 39%。只用 Kuka 訓練是 0%（無法直接遷移），但混合資料能讓 RT-1 從 Kuka 的「視覺/狀態分佈」推斷出 MM 機器人應該採取什麼動作。

### 長視角任務 + SayCan（Table VI）

在 Kitchen1，RT-1 + SayCan 達 67% 執行成功率；Kitchen2 也維持 67%（其他方法在 Kitchen2 大多崩塌）。能拼到 50 步的超長序列。

### 主要結論（也是論文最重要的訊息）

1. 高容量 Transformer + 大量多樣資料 → robotics 也能像 NLP 那樣享受 zero-shot 泛化。
2. **資料多樣性 > 資料數量**。
3. RT-1 可以「吸收」異質資料（模擬、別種機器人），原任務不掉效能、新場景大幅獲益。
4. 35M 參數 + token 壓縮足以做到 3 Hz 真機推論。

### 論文自述的限制

- imitation learning 的天花板就在示範者；無法 surpass demonstrator。
- 對「全新動作」（沒在訓練中出現過任何形式）泛化還是有限；目前只能 recombine 已見概念。
- 仍是同一個機器人型態為主，跨型態仍受限。
- 任務並不極為精細靈巧（dexterous），未來需擴展。

---

## VI. 相關工作（補充重點）

- [[DecisionTransformer|Decision Transformer]]、Multi-Game DT、Trajectory Transformer、Gato：都是把控制當 sequence modelling 的前驅，RT-1 是把它做到「real-world + 大規模 + 即時」的代表。
- BC-Z：zero-shot task generalization via robotic imitation；是 RT-1 最直接的前作 baseline，但 task 數遠少。
- PerceiverActor、CLIPort、VIMA：多模態 prompt / multi-task transformer manipulator，多偏 sim 或 tabletop。
- SayCan：高階 LLM planner + 低階技能組合；RT-1 為它提供更可靠的低階執行層。

---

## 與本研究主線的關聯

我目前的研究主線（如 MEMORY 所載）是：

> robot play（不需大量標記）＋ 極少弱語言（不是完整 instruction）＋ offline meta-RL（從靜態軌跡學分佈內 adaptive policy）＋ 文字 → task spec/embedding 做 zero-shot；同時也並行關注 meta-learning × zero-shot 結構。

RT-1 與這條主線的關係可以分四點寫清楚。

### 1. RT-1 偏 generalist policy，不做 task inference，跟我主線的核心精神「不同」

RT-1 的整體框架是把「語言指令」當成 conditioning vector（USE 嵌入透過 FiLM 注進視覺，再 attention 過 Transformer），然後完全用 behavior cloning 訓練。它**沒有**：

- 沒有 task embedding 的後驗推論（不像 [[PEARL|PEARL]] 的 q(z|context)）。
- 沒有對 task 的不確定性建模（不像 [[VariBAD|VariBAD]] 的 belief b(m)）。
- 沒有 explore vs exploit 的 meta-objective。
- 沒有 offline meta-RL 中常見的 Bellman / reward / Q-function。

換句話說，RT-1 把「task 是什麼」完全外包給使用者（你告訴我 instruction），policy 只負責「執行」。這跟我關心的「在 play 資料上學 task 推論 → 對新弱語言做 zero-shot 適配」是兩個正交的方向。

### 2. 但 RT-1 的「instruction → action」管線就是我研究中可以重用的 backbone

我的設計目標是：給定弱語言（如「move blue thing」）或任務描述短句，把它編成一個 task embedding z，餵給 policy 去做 zero-shot 行為。RT-1 的「USE → FiLM → EfficientNet → TokenLearner → Transformer → 離散動作」這條管線可以幾乎原封不動拿來當：

- **z 的注入機制**：FiLM 是一個很乾淨、可訓練的條件化方式，我把 USE 換成自己的 task embedding 即可。
- **多幀視覺 + 條件化 + token 壓縮 + 即時推論的工程模板**。
- **動作離散化 + 分類損失** 的 imitation backbone，非常適合 offline meta-RL 中 actor 端的 base policy。

具體可重用設計：把 z（不論是 task embedding、belief vector、或 task descriptor 的嵌入）取代或補強 USE，透過 FiLM 注入到視覺，policy 結構就接上去了。

### 3. RT-1 對「資料規模 vs sample-efficient 演算法」的二元對立給了一個有用的對照

RT-1 的核心訊息是「資料多樣性最重要」，這對我來說有兩層意義：

- **負面壓力**：如果未來大家就靠資料規模硬幹，offline meta-RL / task embedding 這類「在資料稀少時提升 sample efficiency」的研究方向會被質疑「為何不直接收更多資料」。我必須在問題框架中強調：(a) 大量真實機器人資料**仍然貴**，play 資料只是「相對」便宜；(b) 弱語言場景下沒有強對齊的 instruction-action pair，RT-1 那套 supervised 訓練不適用；(c) 我關心的是「task spec 在 inference time 才出現」的 zero-shot 設定，這不是 RT-1 訓練範式直接覆蓋的場景。
- **正面啟發**：RT-1 證明「跨任務共享 backbone + 多樣性資料 → 自然湧現組合泛化」。這暗示我在 offline meta-RL 中也應該強調多樣的 task 分佈（很多 play、很多潛在子目標），而不是只在少量 task 上做花俏的 meta 演算法。

### 4. 把 RT-1 當 baseline / pretrained encoder 的具體用法

- **作為 baseline**：在我自己的設定下，把 RT-1 直接以 BC + instruction conditioning 跑一遍，比較它在「弱語言 / 缺乏對齊 instruction」場景下是否仍能 zero-shot。如果它崩掉，就證明 task inference 仍有不可取代的角色。
- **作為 visual / language encoder**：用 RT-1（或 RT-2、OXE pretrained）的 EfficientNet+FiLM+TokenLearner 部分當凍結特徵抽取器，在上面接我自己的 task embedding / meta 模組，可以大幅省下從零訓練的成本。
- **作為 offline 資料生產者**：RT-1 的 130k 軌跡也是潛在的 play-like dataset，可考慮（若公開或透過 OXE）拿來做 task inference 預訓練。

### 5. 跟 offline meta-RL 的差別（補強）

| 面向 | RT-1 | 我的主線（offline meta-RL + zero-shot） |
|---|---|---|
| 任務識別 | 由外部 instruction 直接給定 | 由 context / 軌跡推論出 task embedding |
| 訓練資料 | 大量人類示範（強監督，有指令對齊） | play / 弱語言（弱監督、無精準指令） |
| 損失 | BC 分類損失 | meta-objective、context encoder + policy |
| 推論時 | 給語言就執行 | 給文字 task spec → embedding → policy |
| 規模哲學 | 規模解決一切 | 結構 + 抽象解決資料稀少 |
| 評估泛化 | 新任務 = 已見動詞 + 名詞重組 | 新任務 = 訓練未見的 task embedding 區域 |

關鍵的概念差別：RT-1 把「task 識別」當 free（從 instruction 直接取），而 offline meta-RL 把「task 識別」當核心未知量。**這正好是我研究的 leverage 點**。

---

## 一句話總結

RT-1 證明了在 robotics 也能用「大規模多樣資料 + 條件化 Transformer + 離散動作 token + 即時推論工程」做出強泛化的 generalist policy，並把「資料多樣性比數量重要」這件事用 3000+ 次真實實驗講清楚；對我的研究而言，它是極佳的 backbone 模板與資料規模參照，但它把 task identification 完全外包給語言指令，這正是我以 offline meta-RL + 弱語言 task embedding 路線可以補上的部分。
