---
type: feasibility-study
aliases:
  - "M0 環境實測"
tags:
  - 研究主線
  - CALVIN
  - GB10
  - 環境
summary: "M0 在遠端 spark-3994 (GB10, aarch64, CUDA13) 的環境實測結果：docker+GPU 通、torch 2.7 在 Blackwell sm121 實跑、PyBullet 能渲染、3D-DA 三個 aarch64 痛點(flash-attn/dgl/open3d)全化解。3D-DA 當 backbone 可行性從『要折騰』升級為『實測通暢』。"
---

# M0 環境實測報告（2026-06-07，遠端 spark-3994）

> 上游：[[calvin_backbone_comparison]]（M0 是其建議的第一里程碑）
> 機器：`p76141495@192.168.65.11`（spark-3994）｜實測非推測，皆在遠端真機跑過。
> 軟體路線：**docker + NGC 容器**（user 已加入 docker group）。

---

## 0. 環境基線（M0-0）✅

| 項目 | 實測值 |
| --- | --- |
| 架構 | **aarch64**（ARM64），kernel 6.17 NVIDIA |
| GPU | **NVIDIA GB10**，driver 580.159.03，**CUDA 13.0** |
| 算力 capability | **sm_121**（Blackwell） |
| 磁碟 | 3.7T，可用 **3.0T**（CALVIN 全 split ~100GB 綽綽有餘） |
| 統一記憶體 | **121 GB**（free ~100GB）+ 15G swap |
| CPU | 20 核 |
| 容器 | docker 29.2.1（已加入 docker group）；podman 4.9.3 也在 |
| Python | 系統 3.12.3，無 conda；venv+pip 可用 |

---

## 1. 踩過的坑與解法（給 M1 接手用）

### 坑 1：podman 4.9.3 跑不了 GPU（CDI 版本不相容）
- 系統 `/etc/cdi/nvidia.yaml` 是 CDI spec **0.7.0**，podman 4.9.3 內建 CDI library 只到 **0.6.0** → `unresolvable CDI devices`。
- podman rootless 只掃 root-owned 的 `/etc/cdi`、`/var/run/cdi`，**不讀 user dir**（`cdi_spec_dirs` 設定此版無效）；`--gpus all` 兼容層也是轉 CDI、同樣卡。
- **解法（已採用）**：改用 **docker**（`sudo usermod -aG docker p76141495`，使用者已跑）。docker 的 nvidia runtime 對 0.7.0 CDI 無問題。
- 備案（未用）：`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml && sudo sed -i 's/0.7.0/0.6.0/'` 可讓 podman 也通。

### 坑 2：PyBullet pip wheel 不帶 EGL plugin
- pip 裝的 pybullet 在 GB10 **能渲染出圖，但走 CPU TinyRenderer**（`eglRendererPlugin: cannot open shared object file`）。
- GPU EGL 加速需**源碼編譯 pybullet 帶 EGL**。容器內已驗證：build tools(cmake/gcc/g++) 齊、EGL dev 庫可 apt 裝（57 包無誤）、nvidia EGL 運行庫(`libEGL_nvidia.so`)已被 `--gpus` 注入。→ **M1 可解，但要做**。CALVIN 官方 install.sh 應會處理。

### 容器執行旗標（NVIDIA 建議，訓練時帶上）
`docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ...`

---

## 2. torch + Blackwell 實跑（M0-1）✅

NGC `nvcr.io/nvidia/pytorch:25.04-py3` 容器內：
- `torch 2.7.0a0+...nv25.04`、`cuda available True`、`device NVIDIA GB10`、**capability (12,1)**、**matmul 實跑成功**。
- 一個保守 warning「GB10 may not yet be supported in this version」，但實際運算正常 → 25.04 已能用 GB10。若日後遇怪事可試更新 tag。

---

## 3. CALVIN 渲染（M0-2）⚠️→可解
- ✅ PyBullet 在 GB10 aarch64 **能渲染出圖**（320×240 全幅有內容）。
- ⚠️ 但走 CPU fallback；GPU EGL 需源碼編 pybullet（見坑 2）。**「能不能動」=能；「能不能 GPU 加速」=要編譯。**

---

## 4. 3D Diffuser Actor 三個 aarch64 痛點 — 全化解（M0-3）✅✅

調研時擔心的三個依賴，實測結果**比預期好很多**：

| 依賴 | 調研預期 | **實測結果** |
| --- | --- | --- |
| **flash-attn** | 要 FA4 + ARM 編譯痛 | ✅ **NGC 容器已內建 `flash_attn 2.7.3`，在 GB10 sm121 實跑成功**（FA2 即可，免 FA4） |
| **dgl** | 無 ARM wheel | ✅ **有 aarch64 wheel**（`dgl-2.1.0-cp312-cp312-manylinux2014_aarch64.whl`），pip 直裝 |
| **open3d** | 無 aarch64 wheel（真痛點） | ⚠️ 確認無 wheel，**但 code 層證實 open3d 只在 RLBench 路徑(`utils/utils_with_rlbench.py`) import；CALVIN 路徑(`online_evaluation_calvin/`)只用 pybullet/calvin_env/hydra** → **純跑 CALVIN 不需 open3d，可繞過** |

**dgl 核心用途**：`diffuser_actor/utils/encoder.py` 用 `dgl.geometry` 做 FPS 下採樣（核心 encoder）→ dgl 必須，但已有 aarch64 wheel，無礙。

---

## 5. M0 結論：3D-DA 可行性升級

**[[calvin_backbone_comparison]] 對 3D-DA 的「依賴重、aarch64 要折騰」扣分，實測後大幅緩解**：
- GB10 環境基線全綠（CUDA13、3T 磁碟、121G 統一記憶體）。
- torch+Blackwell 實跑通。
- 3D-DA 三痛點：flash-attn 內建可跑、dgl 有 wheel、open3d 對 CALVIN 路徑可不裝。
- 唯一待做工：PyBullet EGL GPU 渲染需源碼編譯（CPU fallback 已能動，不阻塞流程驗證）。

**→ M0 排雷通過，可進 M1**（clone 3D-DA + CALVIN debug split，重現 baseline 能訓能 eval）。建議 M1 以 NGC `pytorch:25.04-py3` 為基底容器，內裝 calvin_env + 編 pybullet-EGL + pip dgl/diffusers，跳過 RLBench/open3d。

> 注意：本報告所有結論皆遠端真機實測。M1 真正 clone 後若 calvin_env 的 pybullet 版本與 EGL 編譯有衝突，需再排查。
