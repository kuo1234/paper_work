# BTS PoC Handoff for Claude Code

> ⚠️ **新 session 接手請先讀 `PROGRESS.md`**（single source of truth，含最新 v4 現況、SSH 環境、執行指令、完整演進史）。本檔是原始交接背景，現況已被 PROGRESS.md 取代。

## 你現在要接什麼

請接手 `C:\Users\kuo\Desktop\paperwork\bts-poc` 這個 PoC 專案，目標是：

1. 在可 SSH 到 GPU 機器的環境中，跑通這個 PoC
2. 修掉剩餘 runtime bug（若有）
3. 產出第一輪結果：
   - phenomenon 1：歧義規格 entropy > 精確規格 entropy
   - phenomenon 2：隨觀察 entropy 下降
   - phenomenon 3：belief / risk-aware 版 > single-point baseline
4. 若三現象成立，準備擴到 CALVIN；若不成立，診斷是哪個環節出問題

## 背景研究脈絡（請先理解再動手）

這個題目不是一般的 zero-shot language conditioning，而是：

> 把 meta-RL 的 task belief 推論機制，從「測試時要先互動」解放成「免互動的多模態規格（語言+圖片）」；規格作為 prior，觀察作為 likelihood，讓 policy 對 belief 做 risk-aware 行為。

目前完整方法與研究文件已經寫好：

- `idea/method_spec.md`：完整方法（BTS）
- `idea/poc_plan.md`：PoC 詳細實作計畫
- `idea/zero_shot_proposal.md`：高階提案
- `idea/related_work_checklist.md`：related work 查證與威脅判斷

請先讀：
1. `idea/method_spec.md`
2. `idea/poc_plan.md`
3. `idea/paper/T2DA.md`
4. `idea/paper/LUMOS.md`
5. `idea/paper/CALVIN.md`

## 目前 bts-poc/ 內已存在的程式碼

- `envs/gridworld.py`
  - Ambiguous-Spec GridWorld
  - 任務 = 顏色×形狀
  - 規格可精確或歧義
  - 物件採 2 colors × 2 shapes Cartesian product，保證歧義規格真的有意義
  - 有 oracle posterior / BFS expert / vectorization

- `data/generate.py`
  - 產生 offline dataset (jsonl)
  - 同一歧義規格會配到不同任務軌跡，逼出寬 belief
  - 已確認 tiny dataset 可成功生成

- `models/transformer.py`
  - Tiny in-context Transformer
  - 顯式 belief head（softmax over task classes）
  - policy head
  - single-point baseline wrapper（模擬 T2DA 式單點）

- `train.py`
  - `L_IC + lambda * L_belief`
  - teacher-forcing 訓練
  - 會存 `best.pt` 與 `metrics.jsonl`

- `eval/phenomena.py`
  - phenomenon 1：entropy gap
  - phenomenon 2：entropy decay
  - phenomenon 3：belief vs single rollout success（已加 baseline ckpt 介面）

- `run.sh`
  - belief 版 / single-point 版一鍵跑

- `README.md`
  - 執行方式與預期輸出

- `REVIEW_NOTES.md`
  - 我這邊做的 review 結論

- `requirements.txt`
  - 只有 `torch`、`matplotlib`

## 我這邊已驗證的事情（重要）

在 Cowork 沙箱內：

- 所有 `.py` 都通過 `py_compile`
- `data.generate` 可以成功生成 tiny dataset
- 歧義 support 分布合理，例如 tiny dataset 約為：
  - support counter ≈ `{2: 86, 1: 71, 4: 43}`
  - train/test ≈ `175/25`
- 我無法真正跑 train / eval 的唯一原因是：**這個沙箱沒有安裝 torch**
- 所以現在最大的未知數不是語法或路徑，而是：
  1. tensor shape / runtime bug
  2. 現象 2 與 3 是否真的會出來

## 目前我已知的風險 / 你需要優先檢查

### 1. phenomenon 2 可能太弱
目前 oracle posterior 的收斂規則是簡化版：
- 給規格得到候選集合 Z(c)
- 若 agent 站到某個候選物件上，就 posterior 收斂到該任務
- 否則維持相容任務集合

這足以先驗證想法，但可能導致 entropy 到後面才掉、曲線不夠漂亮。
如果第一輪結果不理想，優先改這裡。

### 2. phenomenon 3 雖然已實作，但沒在這邊真跑過
`eval/phenomena.py` 有 `--baseline-ckpt` 介面，但需要你真的把 belief / single 兩個版本各訓一個 ckpt 才能比較。

### 3. train.py 目前只監看 action loss / belief KL
還沒有在每個 epoch 直接算 rollout success。
PoC 初版可接受，但如果你要快調參，建議在 train loop 加一個小型 rollout eval。

## 建議你在 Claude Code 先做的事情（順序）

### Step 0: 進目錄
```bash
cd C:\Users\kuo\Desktop\paperwork\bts-poc
```

### Step 1: 建環境 / 安裝
若你用 conda：
```bash
conda create -n bts-poc python=3.10 -y
conda activate bts-poc
pip install -r requirements.txt
```

### Step 2: 先跑 tiny smoke test（不要直接上大）
```bash
python -m data.generate --out-dir data/tiny --n-episodes 200 --size 7 --horizon 10 --n-objects 4 --spec-mode mixed --test-holdout-tasks green_triangle

python train.py --train-jsonl data/tiny/train.jsonl --test-jsonl data/tiny/test.jsonl --out-dir runs/smoke_belief --epochs 1 --batch-size 32 --d-model 64 --layers 2 --heads 4 --lambda-belief 0.5

python train.py --train-jsonl data/tiny/train.jsonl --test-jsonl data/tiny/test.jsonl --out-dir runs/smoke_single --epochs 1 --batch-size 32 --d-model 64 --layers 2 --heads 4 --lambda-belief 0.5 --single-point-baseline

python eval/phenomena.py --ckpt runs/smoke_belief/best.pt --baseline-ckpt runs/smoke_single/best.pt --test-jsonl data/tiny/test.jsonl --out-dir runs/compare/figs
```

### Step 3: 若 smoke test 有 bug，先修到 tiny 能跑通
優先修：
- import / path
- tensor shapes
- attention mask
- single-point rollout compare

### Step 4: tiny 跑通後，再開 PoC 規模
```bash
bash run.sh
bash run.sh --single
python eval/phenomena.py --ckpt runs/bts_poc_belief/best.pt --baseline-ckpt runs/bts_poc_single/best.pt --test-jsonl data/gridworld_mixed/test.jsonl --out-dir runs/bts_poc_compare/figs
```

## 你要看的三個成功指標

1. `phenomenon1_entropy_gap.json`
   - `ambiguous_mean > exact_mean`
2. `phenomenon2_entropy_decay.png`
   - 平均 entropy 曲線往下
3. `phenomenon3_belief_vs_single.json`
   - `belief_success_rate > single_success_rate`

## 如果結果不好，怎麼判斷

### A. 現象1 不成立
代表資料設計沒逼出歧義，或 belief head collapse。
先檢查：
- `compatible_tasks_t0` 的 support 是否真的 >1
- `lambda_belief` 是否太小
- belief head 是否有學到任何東西

### B. 現象2 不成立
代表「觀察進來 belief 收斂」這件事不夠明顯。
先檢查：
- `compatible_tasks_given_state()` 是否太保守
- 模型是否真的利用 history token
- causal mask / 最後 token 抽取是否正確

### C. 現象3 不成立
可能原因：
- 單點 baseline 其實就夠好（環境太簡單）
- risk-aware policy 沒有真正利用 belief
- rollout 比較寫錯

這時可考慮：
- 加大歧義程度
- 增加物件數 / 任務組合
- 改更強的 risk aggregation（如 CVaR / worst-case over top-k tasks）

## 重要：不要現在就上 CALVIN
PoC 的精神是先用 toy env 證明三現象。只有當你在 toy env 上看到這三個現象成立,才值得把它搬到 CALVIN / 真實圖片 / 真 text encoder。

## 如果你要幫我做 code review，請特別看這些點

- `train.py` 是否有 teacher-forcing / target 對齊錯位
- `models/transformer.py` 取最後有效 token 的 index 是否正確
- `eval/phenomena.py` rollout compare 是否真的公平
- `envs/gridworld.py` 的 posterior 更新規則是否足夠支撐 phenomenon 2
- 是否需要把 train/test split 改成更嚴格的 unseen-task 組合切分

## 最後一句

現在這個 PoC 不是「已完成的研究系統」,而是「已經到可以在本地 GPU 上快速驗證核心假設的工程起點」。你的任務不是大改,而是：**先把 tiny 跑通、看到三個現象,再決定是否值得上 CALVIN。**
