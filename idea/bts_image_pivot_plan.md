---
type: research-plan
aliases:
  - "BTS image-side pivot"
  - "CALVIN image perception roadmap"
tags:
  - 研究主線
  - BTS
  - CALVIN
  - image-policy
  - belief
  - binding
summary: "M1b-6 停止 3D-DA closed-loop reproduction 後，BTS 全面轉向 image/perception-side CALVIN 的設計注意事項、里程碑與閱讀清單。"
---

# BTS：全面轉向 image/perception-side CALVIN（2026-06-08）

## 0. 決策

M1b-6 3D-Diffuser-Actor closed-loop reproduction 停止。原因：GB10 100-seq 明顯低分；A6000 reference N=10 也沒有恢復論文級。繼續 debug 3D-DA public checkpoint / CALVIN env reproducibility 的邊際價值低。

新方向：**image/perception-side CALVIN**。

核心主張保留：

> structured belief improves visual-language manipulation under attribute-object binding ambiguity.

不是單純把 3D policy 換成 image policy；BTS 的 novelty 仍是 belief / object-attribute binding。

---

## 1. 轉圖片後最需要注意的事

### 1.1 不要只做 image encoder + action decoder

錯誤方向：

```text
RGB + language + proprio -> policy -> action
```

這只是 baseline，BTS novelty 會消失。

正確方向：

```text
RGB + language + proprio
  -> visual-language features / object evidence
  -> belief or binding module
  -> action policy
```

BTS 插入點應在 language-image fusion 後、action decoder 前。

### 1.2 Evaluation 要切 binding-sensitive subset

不要只看 CALVIN avg seq len。需要額外報：

- binding-sensitive subset success
- wrong-object error rate
- wrong-attribute error rate
- first-contact object accuracy
- target-object belief calibration
- OOD attribute recombination success

BTS 優勢應該主要出現在：

- 多物件
- 顏色/形狀/位置屬性混淆
- 指令指定 attribute-object pair
- shortcut 失效的初始狀態

### 1.3 資料問題是最大現實風險

目前已有正式 ABC→D validation partial extract，但 image policy training 需要 training data。

可選：

1. 下載/抽取 CALVIN training。
2. 先用 CALVIN debug dataset 或小 subset 做 feasibility。
3. 若 CALVIN training 太重，暫時轉 LIBERO / simpler image benchmark 做 BTS proof-of-concept。

### 1.4 Action representation 先不要大改

為了 isolating BTS effect，建議先沿用 CALVIN 常見：

```text
7D relative action = dx dy dz droll dpitch dyaw gripper
```

不要同時引入新 action space、3D point、oracle state。先做 image obs -> same action。

### 1.5 State labels 可作診斷，不要讓 policy 依賴 oracle

CALVIN simulator state 可用來產 pseudo labels：object position、color block、target object、first contact object。這些適合作：

- auxiliary binding loss
- probe labels
- diagnostic metrics

但主 policy input 應維持 RGB + language + proprio，避免變成 state-based oracle policy。

---

## 2. 建議架構版本

### 2.1 Baseline

```text
static RGB + gripper RGB + proprio + instruction
  -> image-language encoder
  -> action chunk decoder / diffusion decoder
```

目標：先跑通 image policy，不求 novelty。

### 2.2 BTS version

```text
static RGB + gripper RGB + proprio + instruction
  -> image-language encoder
  -> object/query tokens
  -> belief over object-attribute binding
  -> belief-conditioned action decoder
```

Belief module 輸出可為：

- belief tokens
- target-object distribution
- reweighted object query tokens
- uncertainty / entropy feature
- temporal belief update state

### 2.3 Ablations

必做：

1. baseline without belief
2. belief token only
3. belief + auxiliary binding loss
4. oracle target label upper bound（只作分析，不作主模型）
5. shuffled language / wrong attribute stress test

---

## 3. 里程碑

### M2-img-0：資料與 baseline feasibility

- 確認 CALVIN image training data 可讀。
- 建 minimal dataloader：static RGB、gripper RGB、proprio、language、relative action。
- 跑 offline BC / diffusion loss 下降。
- closed-loop single-task smoke 成功率 > random。

### M2-img-1：binding diagnostics

- 從 simulator state / task language 產 target-object pseudo label。
- 實作 first-contact object accuracy。
- 建 binding-sensitive task subset。

### M2-img-2：BTS belief module

- 加 belief/query tokens。
- 加 optional auxiliary binding loss。
- 比較 baseline vs BTS on binding subset。

### M2-img-3：正式 CALVIN ABC→D / 或替代 benchmark

- 若 CALVIN training/eval 太重，先用 LIBERO 或 smaller image benchmark 建 proof-of-concept。
- 若 CALVIN 可跑，報 full avg seq len + binding diagnostics。

---

## 4. 需要重新閱讀 / 新增閱讀的論文

### A. CALVIN / language-conditioned long-horizon manipulation

1. **CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks**
   - 必讀。重新確認 task split、ABC→D、evaluation protocol、annotations、state/task oracle。

2. **What Matters in Language Conditioned Robotic Imitation Learning over Unstructured Data** / HULC 相關 CALVIN baseline
   - 目的：理解 image-language CALVIN baseline 怎麼處理 language、history、multi-task。

3. **MCIL / language-conditioned imitation learning baselines for CALVIN**
   - 目的：找最小可跑 image baseline。

### B. Image visuomotor policy backbone

4. **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**
   - 必讀。action diffusion / action chunking 是 image route 最自然 baseline。

5. **ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**
   - 必讀。action chunking transformer，簡潔且 strong baseline。

6. **RT-1: Robotics Transformer for Real-World Control at Scale**
   - 目的：image-language-action tokenization / action discretization。

7. **Octo: An Open-Source Generalist Robot Policy**
   - 目的：現代 open backbone 候選；評估是否可接 CALVIN / 只借設計。

8. **OpenVLA: An Open-Source Vision-Language-Action Model**
   - 目的：VLA baseline / representation；但先不要被大模型工程拖住。

### C. Object-centric / binding / compositionality

9. **Object-Centric Learning with Slot Attention**
   - 必讀。object slots 作為 belief/binding carrier 的基礎。

10. **Perceiver / Perceiver IO**
   - 目的：latent query bottleneck；可作 belief tokens 設計參考。

11. **Transporter Networks / CLIPort**
   - 目的：language-conditioned manipulation 中 object grounding / spatial action map。

12. **PerAct: Perceiver-Actor for Multi-Task 3D Robot Manipulation**
   - 雖是 3D，但 object/language/action fusion 值得參考；不是主線依賴。

### D. VLM grounding / open-vocabulary object evidence

13. **CLIP**
   - 重新看 text-image alignment 限制；BTS 不能只假設 CLIP 自然解 binding。

14. **OWL-ViT / Grounding DINO / Segment Anything**
   - 只作 optional object proposal 工具，不建議一開始納入主 pipeline。

### E. BTS claim 需要對照的 reasoning / belief 文獻

15. **POMDP / belief state in robotics imitation or RL**
   - 目的：把 belief module 的理論語言寫穩，不只是 transformer memory。

16. **Compositional generalization / attribute-object binding in vision-language models**
   - 目的：支撐 BTS 的 scientific claim：錯在 binding，不只是 perception accuracy。

---

## 5. 第一批優先閱讀順序

只讀最小必要集，避免再次發散：

1. CALVIN benchmark paper
2. HULC / CALVIN image-language baseline
3. Diffusion Policy
4. ACT
5. Slot Attention
6. CLIPort
7. Octo 或 OpenVLA（二選一先讀 Octo）

讀完這 7 篇即可設計 M2-img-0/M2-img-1。

---

## 6. 下一步實作判斷

下一步不是直接訓練大模型，而是確認：

1. CALVIN training RGB data 是否可用 / 是否需要下載。
2. 最小 image dataloader 能否產出 `(static_rgb, gripper_rgb, proprio, language, rel_action)`。
3. 能否從 task + scene state 產 target-object diagnostic label。
4. baseline image BC 是否 loss 下降。

若這四個成立，再開始 BTS belief module。

---

## 7. Spark image data smoke（2026-06-08）

在 spark / container `bts_m1` 檢查目前 CALVIN `task_ABC_D` 資料。

### 7.1 現有資料

```text
/workspace/bts/calvin/dataset/task_ABC_D/training   exists: False
/workspace/bts/calvin/dataset/task_ABC_D/validation exists: True
validation npz_count: 99022
language windows: 1087
```

結論：目前只有 validation。可以做 dataloader / diagnostics / overfit smoke；不能作正式訓練資料。

### 7.2 單 frame 欄位

`episode_0000000.npz` keys：

```text
actions        (7,)  float64
rel_actions    (7,)  float64
robot_obs      (15,) float64
scene_obs      (24,) float64
rgb_static     (200, 200, 3) uint8
rgb_gripper    (84, 84, 3) uint8
rgb_tactile    (160, 120, 6) uint8
depth_static   (200, 200) float32
depth_gripper  (84, 84) float32
depth_tactile  (160, 120, 2) float32
```

Image-route minimal dataloader 所需欄位都存在：

```text
static RGB, gripper RGB, proprio(robot_obs), rel_actions, language annotation
```

### 7.3 Language window

`lang_annotations/auto_lang_ann.npy`：

```text
language.ann   list[str]
language.task  list[str]
language.emb   (1087, 1, 384)  # CALVIN embedding, not necessarily BTS/CLIP input
info.indx      list[(start, end)]
```

範例：

```text
(40, 104): open the drawer / open_drawer
(152, 216): toggle the button to turn on the led light / turn_on_led
(16413, 16456): take the red block and rotate it right / rotate_red_block_right
```

### 7.4 Immediate implication

M2-img-0 可以先做：

1. validation-only image dataloader smoke
2. tiny overfit experiment（明確標示不是正式訓練）
3. binding-sensitive window selection
4. target-object diagnostic parser prototype

正式 CALVIN training 需要補 `task_ABC_D/training` 或先抽 small training subset。


---

## 8. M2-img-0 smoke results（2026-06-08）

### 8.1 Minimal image dataloader passed

在 spark / `bts_m1` validation-only 上建 `CalvinImageWindowDataset` smoke，產出 batch：

```text
static_rgb  (8, 4, 3, 200, 200) torch.float32 [0,1]
gripper_rgb (8, 4, 3, 84, 84)   torch.float32 [0,1]
proprio     (8, 4, 7)            torch.float32
rel_action  (8, 4, 7)            torch.float32
frames      (8, 4)               torch.int64
language    list[str]
task        list[str]
```

First batch examples：

```text
language: ['open the drawer', 'grasp the handle of the drawer, then open it', 'toggle the button to turn on the led light']
task:     ['open_drawer', 'open_drawer', 'turn_on_led']
```

結論：image route 的最小資料欄位已確認可用。

### 8.2 Binding diagnostic parser v2 coverage

從 validation `language.ann` + `language.task` 做 rule-based target parser。v2 修正 slider vs drawer 的 `door handle` 誤判。

Coverage：

```text
windows: 1087
unparsed_target: 0
binding_sensitive_count: 782
color_block_count: 673
```

Target counts：

```text
blue_block:   234
pink_block:   221
red_block:    218
drawer:       140
sliding_door:  90
block:         64
led:           60
lightbulb:     60
```

這代表 CALVIN validation 本身有大量 attribute-object binding windows，可作 BTS diagnostic subset。

注意：action parser 目前只是輔助，仍會把一些 `grasp the door handle ... slide` parse 成 `lift`；target/binding parser 已足夠做第一版 diagnostic，action verb 後續再精修。

### 8.3 Immediate next technical step

1. 把 `/tmp/calvin_image_dataloader_smoke.py` 正式整理到 repo（或新 image prototype dir）。
2. 把 `/tmp/calvin_binding_parser_v2.py` 整理成 reusable diagnostic script。
3. 補 training data：目前 validation-only 不能訓練正式模型。
4. 在 validation-only 先做 tiny overfit sanity：確認 image encoder + small MLP/Transformer 能 overfit 16 windows 的 next-action prediction。


### 8.4 Tiny validation-only image overfit passed

目的：只驗證 image pipeline 能 end-to-end train，不作正式模型結果。資料仍是 validation-only，因此不能報為 training performance。

設定：

```text
samples: 32 frames from validation language windows
input: static RGB + gripper RGB downsampled to 64x64, proprio[:7]
model: small CNN + MLP
target: rel_actions (7D)
optimizer: AdamW lr=3e-3
steps: 200 epochs over tiny set
```

結果：

```text
step 0   mse 0.127945
step 25  mse 0.054843
step 50  mse 0.049779
step 75  mse 0.043883
step 100 mse 0.026017
step 125 mse 0.008852
step 150 mse 0.003859
step 175 mse 0.002249
step 200 mse 0.001562
ratio final/initial = 0.0122
```

結論：image batch format + action target + simple model training loop 全部可用。下一個 blocker 不是模型能否訓練，而是 **正式 training split / small training subset 的取得**。
