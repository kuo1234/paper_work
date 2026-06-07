---
type: feasibility-study
aliases:
  - "CALVIN backbone 選型"
tags:
  - 研究主線
  - CALVIN
  - backbone
  - 可行性
summary: "進入 CALVIN 階段的 backbone 可行性比較：Octo / CALVIN 官方 baseline(HULC/MCIL) / Diffusion Policy(3D Diffuser Actor) 三條路，評估接 belief head 難度、multi-modal action、binding 解法、CALVIN 整合、社群、GB10 相容、與 method_spec 契合。結論：以 3D Diffuser Actor 為主候選。"
---

# CALVIN backbone 可行性比較（2026-06-07）

> 上游：[[method_spec]] §3（backbone 要求）｜[[poc_plan]]｜toy 已收斂（見 memory `bts-p3-derisk-plan` / `bts-calvin-onramp`）
> 目的：onramp 清單第一步——「先做 backbone 可行性比較再拍板，別一開始就押錯 codebase」。
> 範圍：純調研（讀論文+repo+README），未 clone、未動 code、未碰 GPU。

---

## 0. 為什麼這步重要（toy take-away 帶上來）

toy 三輪診斷把「P3 為何不過」定錨在**屬性-物件 binding**（flat 向量學不出符號式屬性綁定，neural 55% vs oracle 96%），且 policy 自由 rollout 崩壞需 rollout-穩定 + multi-modal action 的 backbone。所以選 backbone 的**硬需求**直接由 toy 結論導出：

| toy 導出的硬需求 | 為什麼 | 對應 backbone 特性 |
| --- | --- | --- |
| **entity/patch token + cross-attention** | 解 binding（toy flat 表徵的結構性極限） | 三個候選都是 transformer token-based，皆滿足 |
| **天生 multi-modal action head** | 解 BC 塌平均、greedy argmax 崩壞 | diffusion action head（Octo / 3D-DA 都有）；CALVIN 官方靠 latent-plan 較弱 |
| **rollout 穩定 + 大規模 play data** | 解 covariate shift / compounding error | 三者都吃 CALVIN play；VLA/diffusion 系更穩 |
| **可掛顯式 belief head + KL 監督接點** | method_spec §3.3 的核心新增 | 取決於架構——見下方逐項評估 |

method_spec §3 對 backbone 的要求：spec encoder（凍結 ViT+text enc）→ in-context transformer Θ → **顯式 belief head**（§3.3）+ **risk-aware policy head**（§3.4）。§4.3 需要 **oracle 相容任務分布 `p*(z|c,o)`** 當 belief KL 監督目標——這要靠 **CALVIN task detector**（官方提供，可對任意 frame 判定任務狀態）。

---

## 1. 三個候選的事實卡（皆已查證，附來源）

### A. Octo（generalist robot policy）
- **架構**：transformer-based diffusion policy；block-wise attention + readout token；task token（語言 / goal image）+ observation token + readout token。✅ token-based、有 cross-attention 結構 → 解 binding。
- **action head**：diffusion，action chunk=4，**天生 multi-modal**。✅
- **框架**：**JAX / Flax**（釘 `jax==0.4.20`，2023 末）。⚠️ 不是 PyTorch。
- **CALVIN**：**沒在 CALVIN 上評估**（訓練資料是 Open X-Embodiment，eval 在真機 WidowX + 模擬 Gym）。⚠️ 要自己接 CALVIN data loader + action space。
- **規模 / 硬體**：Octo-Small 27M、Octo-Base 93M；推論 1×4090 即可；預訓練用 TPUv4-128，資料 ~1.2TB。fine-tune「consumer GPU 幾小時」。
- **維護**：最後 release v1.5 = 2024-05，逾一年無新 release。⚠️ 半停更。
- **授權**：MIT。
- 來源：[Octo arXiv 2405.12213](https://arxiv.org/abs/2405.12213)、[octo-models.github.io](https://octo-models.github.io/)、[github.com/octo-models/octo](https://github.com/octo-models/octo)

### B. CALVIN 官方 baseline（MCIL / HULC）
- **架構**：MCIL（共享 latent goal，吃 image-goal + language-goal）；HULC 在其上加 hierarchical + transformer + multimodal latent plan + 對比學習。
- **action head**：latent-plan + 確定式 policy，**multi-modal 表達靠 latent plan 較弱**（非 diffusion）。⚠️ 正是 toy 崩壞的那類。
- **框架**：**PyTorch + PyTorch Lightning + Hydra**，Python **3.8**（老）。✅ 原生 CALVIN，但技術棧舊。
- **CALVIN**：**原生**——天生跑 CALVIN，提供 D_D 預訓練 checkpoint，data loader / EGL 渲染 / task indicator（=oracle）全內建。✅✅ 整合度最高。
- **效能**：HULC ABC→D 平均序列長 ~0.7（遠落後現代 SOTA ~3.0）。⚠️ 弱 baseline。
- **維護**：最後 commit 2023-02，無正式 release，49 open issues。⚠️ 已停更。
- **PyBullet**：EGL GPU 渲染（需 `EGL_VISIBLE_DEVICES`，舊版會 OOM）；setuptools 需 <58。
- **授權**：MIT。
- 來源：[github.com/mees/calvin](https://github.com/mees/calvin)、[CALVIN RA-L 2022]、HULC RA-L 2022

### C. Diffusion Policy — 具體選 **3D Diffuser Actor**（CALVIN 上的現代 SOTA 系）
- **架構**：3D scene token + **3D 相對/絕對 attention** + diffusion 去噪 end-effector pose；語言 token 與 3D 視覺 token 一起 attend。✅ token-based + cross-attention → 解 binding。
- **action head**：diffusion 去噪 3D pose trajectory，**天生 multi-modal**。✅
- **框架**：**PyTorch**（`diffusers["torch"]`, `dgl cu116`, `flash-attn`, `open3d`）。✅
- **CALVIN**：**有原生 CALVIN 整合 + SOTA 結果**（ABC→D zero-shot，比前 SOTA +9% 相對，2024-08 釋出 no-history CALVIN 模型）。✅
- **depth**：**需要 RGB-D / sensed depth**（3D 表徵從深度聚合）。⚠️ CALVIN 有深度，但這綁死了 obs 模態，且讓 method_spec 的「goal image spec」要多想（goal image 是否也要 depth）。
- **依賴重**：CUDA 11.6 + flash-attn + dgl + open3d + （RLBench 那套 CoppeliaSim 可略，只跑 CALVIN 的話）。⚠️ 安裝較痛。
- **維護**：29 commits，2024-08 有更新；活躍度中等。
- **授權**：MIT。
- 來源：[3D Diffuser Actor arXiv 2402.10885](https://arxiv.org/abs/2402.10885)、[github.com/nickgkan/3d_diffuser_actor](https://github.com/nickgkan/3d_diffuser_actor)

---

## 2. GB10 (DGX Spark) 硬體相容性（onramp「最花時間先確認」）

GB10 = Grace ARM CPU（**aarch64**）+ Blackwell GPU（SM 12.x，需 **CUDA 12.8+ / 13.x**）+ 統一記憶體（~128GB）。

| 項目 | PyTorch（B/C） | JAX（A=Octo） |
| --- | --- | --- |
| aarch64 CUDA wheel | ✅ NVIDIA 提供 ARM64 + Blackwell wheel（NGC 容器最穩）；標準 pytorch.org wheel 不適用 | ✅ 官方有 aarch64-linux CUDA wheel |
| Blackwell 支援 | ✅ 需 torch + CUDA 12.8/13.x（新版才有 sm_120/121） | ⚠️ JAX 文件 CUDA13「SM7.5 or newer」涵蓋，但 **Octo 釘死 jax 0.4.20（2023 末，無 Blackwell）→ 必須升 JAX，可能連帶改 Octo code** |
| 3D-DA 的 cu116 / flash-attn | ⚠️ cu116 太舊（無 Blackwell），**dgl/flash-attn 需找 aarch64 + CUDA12.8+ 版**，flash-attn 在 aarch64 編譯是已知痛點 | — |
| PyBullet + EGL（CALVIN 渲染） | ⚠️ 需確認 aarch64 上 PyBullet EGL headless 渲染可跑（x86 慣例，ARM 較少人踩過） | 同左（CALVIN 渲染與 backbone 框架無關，都要面對） |

**統一結論**：三條路在 GB10 上**都有 aarch64 + Blackwell 的安裝風險**，且**CALVIN 本身的 PyBullet/EGL 在 aarch64 是共同未知數**——這風險與選哪個 backbone 無關，必須最先用最小 split（D→D debug 1.3GB）跑一次「裝得起來 + 渲染得出來」的 smoke。**torch 系（B/C）的 Blackwell 支援比 JAX(Octo 釘舊版) 成熟**，這是排序上對 Octo 的扣分。

---

## 3. 選型矩陣

| 維度（權重） | A. Octo | B. CALVIN官方 HULC | C. 3D Diffuser Actor |
| --- | --- | --- | --- |
| 解 binding（cross-attn token）| ✅ | ✅(latent) | ✅ |
| **multi-modal action**（toy 痛點）| ✅ diffusion | ⚠️ 弱 | ✅ diffusion |
| **CALVIN 原生整合** | ❌ 要自接 | ✅✅ 原生 | ✅ 原生+SOTA |
| **接 belief head 難度** | 中（JAX，架構乾淨但要懂 Flax）| 中（latent-plan 處可掛，但 code 舊）| 中-高（diffusion conditioning 注 belief 要設計）|
| oracle 後驗監督接點（§4.3）| 需自接 CALVIN detector | ✅ detector 內建 | ✅ 用 CALVIN detector |
| **框架 / GB10 相容** | ⚠️ JAX 釘舊版，Blackwell 要升 | ✅ torch，但 py3.8 舊 | ✅ torch，但依賴重(flash-attn/dgl aarch64)|
| SOTA 強度（baseline 說服力）| 無 CALVIN 數 | ⚠️ 弱(~0.7) | ✅ 強(~3.0+) |
| 社群 / 維護 | ⚠️ 半停更 | ⚠️ 停更 | ◯ 中等 |
| obs 模態限制 | RGB | RGB | ⚠️ 需 RGB-**D** |
| 與 method_spec 契合 | ◯ in-context 但無 CALVIN | ◯ 雙 spec 原生但無 belief | ✅ 語言+3D，diffusion belief 可當 §3.3(c) |

---

## 4. 建議（待使用者拍板）

### 主推：**C. 3D Diffuser Actor 為 backbone 起點**
理由：(1) **CALVIN 原生 + 現代 SOTA**——baseline 一開始就強，審稿說服力夠；(2) **diffusion action head 天生 multi-modal**，直接解 toy 的崩壞痛點；(3) **PyTorch**，GB10 的 Blackwell 支援比 JAX 成熟；(4) diffusion belief 可直接當 method_spec §3.3(c) 的 ensemble/particle belief head，**契合度最高**。
代價：依賴重（flash-attn/dgl/open3d 在 aarch64 要花時間）、需 RGB-D（綁 obs 模態，goal-image spec 要重想）。

### 但第一里程碑不要先碰 belief（延續 toy 範圍紀律）
1. **M0（純可行性 smoke，最先做）**：在 GB10 上用 CALVIN **D→D debug split（1.3GB）** 把「裝得起來（torch+PyBullet+EGL aarch64）+ 渲染得出來 + 跑得動一次 eval」走通。**這步與 backbone 無關但風險最高，先排雷。**
2. **M1**：重現 3D-DA 在 CALVIN 的 baseline（能訓能 eval，對得上論文數字）。確認流程通。
3. **M2**：才加 belief head（§3.3）+ §4.1 歧義資料 + §4.3 oracle KL 監督（用 CALVIN task detector）。
4. **M3**：risk-aware（§3.4）+ calibration（§4.4）。

### 備援判斷
- 若 M0/M1 發現 3D-DA 依賴在 aarch64 裝不起來（flash-attn/dgl 卡死），**退回 B（CALVIN 官方 HULC）**做流程打通——它原生、torch、oracle 內建，雖弱但能先讓 belief 機制跑起來，之後再換強 backbone。
- **不建議 A（Octo）當起點**：無 CALVIN 整合 + JAX 釘舊版踩 Blackwell + 半停更，三重摩擦最大。Octo 的價值在「block-wise + readout token」的乾淨 in-context 設計，可當 method_spec §3.2 in-context 寫法的參考，不必當執行 backbone。

### 一個必須先確認的 method_spec 張力
3D-DA 需 **RGB-D**，但 method_spec §3.1 的 spec 是「凍結 ViT 出 goal **image** patch token」。要嘛 (a) goal image 也給 depth、要嘛 (b) belief 監督只用 oracle detector（不靠 goal image 重建），spec 仍走 2D。**M2 前要先定這個。**

---

## 5. 來源
- [Octo: arXiv 2405.12213](https://arxiv.org/abs/2405.12213)｜[octo-models.github.io](https://octo-models.github.io/)｜[github.com/octo-models/octo](https://github.com/octo-models/octo)
- [CALVIN: github.com/mees/calvin](https://github.com/mees/calvin)（RA-L 2022；MCIL/HULC baseline、task detector、PyBullet EGL）
- [3D Diffuser Actor: arXiv 2402.10885](https://arxiv.org/abs/2402.10885)｜[github.com/nickgkan/3d_diffuser_actor](https://github.com/nickgkan/3d_diffuser_actor)
- [JAX installation docs](https://docs.jax.dev/en/latest/installation.html)（aarch64 CUDA wheel、SM 版本支援）

> 注意：本報告中 CALVIN 序列長數字（HULC ~0.7、3D-DA ~3.0+）為從論文/leaderboard 記憶與 repo 描述整理，**M1 重現時須對齊官方確切數字**。
