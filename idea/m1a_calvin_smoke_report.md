---
type: experiment-report
aliases:
  - "M1a CALVIN smoke"
  - "3D-DA CALVIN GB10 smoke"
tags:
  - 研究主線
  - CALVIN
  - 3D-Diffuser-Actor
  - GB10
  - 環境
summary: "M1a 在遠端 spark-3994 (GB10) 上把 3D Diffuser Actor + CALVIN debug split 的 closed-loop eval 跑通。記錄容器、依賴、EGL、dgl、pyhash、camera order 等所有 workaround，供 M1b/正式化重建。"
---

# M1a：3D Diffuser Actor + CALVIN closed-loop eval smoke（2026-06-07）

> 上游：[[calvin_backbone_comparison]]、[[m0_env_smoke_report]]
> 遠端：`p76141495@192.168.65.11`（spark-3994, GB10, aarch64, CUDA13）
> 容器：`bts_m1`，image 已 commit 為 `bts_m1_env:latest`
> 目標：不是對論文數字，而是確認 **3D-DA checkpoint + CALVIN env + GPU EGL + RGB-D point cloud + diffusion policy + closed-loop rollout** 整條技術鏈能跑。

---

## 0. 最終結果 ✅

在 CALVIN **debug split** + `NUM_SEQUENCES=1` 上，3D Diffuser Actor 的 online closed-loop eval 已跑完整個 60-step rollout：

```text
step: 59: 100%|██████████| 60/60
1/5 : 100.0% | 2/5 : 0.0% | 3/5 : 0.0% | 4/5 : 0.0% | 5/5 : 0.0% ||
Load 1/1000 episodes...
disconnecting id 0 from server
Destroy EGL OpenGL window.
```

**這代表實測通過：**
- CALVIN env 可初始化；
- GPU EGL render 可初始化（曾確認 `GL_RENDERER=NVIDIA GB10/PCIe`）；
- 3D-DA checkpoint 可載入；
- CLIP text encoder 可載入；
- RGB-D → point cloud → 3D encoder → diffusion policy forward 可跑；
- policy action 可送進 PyBullet closed-loop rollout；
- eval 結果可彙總。

**不要解讀數字**：此輪用 debug split + 1 sequence，只是技術 smoke，不是論文 ABC→D 1000-seq eval。

---

## 1. 遠端資料 / repo / checkpoint 狀態

```text
~/bts/3d_diffuser_actor/            # cloned from nickgkan/3d_diffuser_actor
~/bts/calvin/                       # cloned from mees/calvin --recurse-submodules
~/bts/calvin/dataset/calvin_debug_dataset/   # 1.3G debug split
~/bts/3d_diffuser_actor/train_logs/diffuser_actor_calvin.pth  # 184M HF checkpoint
~/bts/M1a_NOTES.md                  # 遠端補充筆記
```

Checkpoint source:
`https://huggingface.co/katefgroup/3d_diffuser_actor/resolve/main/diffuser_actor_calvin.pth`

---

## 2. 容器啟動方式

基底：`nvcr.io/nvidia/pytorch:25.04-py3`（torch 2.7, CUDA13, GB10 sm_121 實跑通）。

容器需帶 **graphics capability**，否則 CALVIN EGL 會失敗：

```bash
docker run -d --name bts_m1 --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYOPENGL_PLATFORM=egl \
  -e EGL_VISIBLE_DEVICES=0 \
  -v ~/bts:/workspace/bts -w /workspace/bts \
  bts_m1_env:latest sleep infinity
```

當前容器狀態已 `docker commit bts_m1 bts_m1_env:latest` 固化一次。

---

## 3. 安裝策略：保護 NGC torch，不跑 CALVIN install.sh

**不要直接跑 CALVIN `install.sh`**，因為 `calvin_models/requirements.txt` 釘死：
- `torch==1.13.1`
- `pytorch-lightning==1.8.6`
- `setuptools==57.5.0`

這會破壞 GB10 必需的 NGC torch 2.7。實測做法：

```bash
cd /workspace/bts/calvin
pip install -q wheel
(cd calvin_env/tacto && pip install -q --no-deps -e .)
(cd calvin_env && pip install -q --no-deps -e .)
(cd calvin_models && pip install -q --no-deps -e .)   # 只裝 code，不裝 torch1.13 pin
pip install -q hydra-core hydra-colorlog omegaconf gym numba numpy-quaternion \
  cloudpickle gitpython opencv-python-headless pandas rich pybullet
```

3D-DA 純 Python 依賴：

```bash
pip install -q typed-argument-parser einops blosc tqdm absl-py tensorboard \
  beautifulsoup4 defusedxml trimesh ftfy regex
pip install -q --no-deps git+https://github.com/openai/CLIP.git
pip install -q "transformers==4.44.2"   # 5.x 會讓 CLIPTextModel isinstance 檢查失敗
pip install -q torchdata==0.7.1 diffusers
pip install -q dgl
```

---

## 4. 已踩坑與 workaround

### 4.1 podman CDI 不相容 → 改 docker
- podman 4.9.3 讀不懂系統 `/etc/cdi/nvidia.yaml` 的 CDI spec 0.7.0 → `unresolvable CDI devices`。
- 使用者已跑 `sudo usermod -aG docker p76141495`，新 SSH session 後 docker 可用。
- docker `--gpus all` 可看到 GB10。

### 4.2 CALVIN GPU EGL 成功條件
初始錯誤：`failed to EGL with glad`。

解法：
1. 容器啟動時加：
   - `-e NVIDIA_DRIVER_CAPABILITIES=all`
   - `-e PYOPENGL_PLATFORM=egl`
   - `-e EGL_VISIBLE_DEVICES=0`
2. 容器內裝 header 並編 CALVIN EGL checker：

```bash
apt-get update -qq
apt-get install -y -qq libx11-dev libegl1-mesa-dev libgl1-mesa-dev
cd /workspace/bts/calvin/calvin_env/egl_check
bash build.sh
EGL_VISIBLE_DEVICE=0 ./EGL_options.o
```

成功訊號：

```text
Loaded EGL 1.5 after reload.
GL_VENDOR=NVIDIA Corporation
CUDA_DEVICE=0
GL_RENDERER=NVIDIA GB10/PCIe
```

### 4.3 `pyhash` 缺失 → 純 Python shim
CALVIN / 3D-DA 只用 `pyhash.fnv1_32()` 做 deterministic seed。Python 3.12 上原始 pyhash 不必硬裝，直接 shim：

```python
# /usr/local/lib/python3.12/dist-packages/pyhash.py
class fnv1_32:
    def __init__(self):
        self.offset = 2166136261
        self.prime = 16777619
    def __call__(self, value):
        if not isinstance(value, (bytes, bytearray)):
            value = str(value).encode('utf-8')
        h = self.offset
        for b in value:
            h = (h * self.prime) & 0xffffffff
            h = h ^ b
        return h
```

### 4.4 dgl / GraphBolt / FPS
dgl aarch64 wheel 可裝，但有兩個問題：

1. `graphbolt` 找不到 torch2.7 對應 C++ `.so`：
   `libgraphbolt_pytorch_2.7.0a0.so`。
   - 3D-DA 不用 graphbolt；只用 `dgl.geometry.farthest_point_sampler`。
   - workaround：把 `dgl/graphbolt/__init__.py` 末尾 `load_graphbolt()` 包成 try/except（non-fatal）。

2. `FarthestPointSampler does not support cuda device`：
   - aarch64 dgl wheel 的 FPS 算子是 CPU-only。
   - workaround：3D-DA `diffuser_actor/utils/encoder.py` 的 `run_fps` 改成 `.cpu()` 算 FPS，再 `.to(context_features.device)`。

M2 正式化建議：不要長期依賴 site-packages hot patch；要嘛自實作 FPS，要嘛編 dgl CUDA aarch64，要嘛把 patch 寫成清楚 repo diff。

### 4.5 tactile camera 不需要，debug split 要刪掉
CALVIN debug split 的 `validation/.hydra/merged_config.yaml` 有：

```yaml
cameras:
  static: ...
  gripper: ...
  tactile: calvin_env.camera.tactile_sensor.TactileSensor
```

3D-DA 只需要 static+gripper RGB-D。tactile 會引入 tacto/pyrender/X11 問題。workaround：刪掉 `tactile` key。

### 4.6 camera order assumption
3D-DA 原本寫死：

```python
static_cam = env.cameras[0]
gripper_cam = env.cameras[1]
```

debug/full config 可能順序不同，曾導致 `StaticCamera has no robot_uid`。workaround：按 class name 選：

```python
static_cam = next(c for c in env.cameras if c.__class__.__name__ == "StaticCamera")
gripper_cam = next(c for c in env.cameras if c.__class__.__name__ == "GripperCamera")
```

### 4.7 git dubious ownership
容器 root 讀 host-mounted repo 會觸發 gitpython：

```text
fatal: detected dubious ownership in repository
```

workaround：

```bash
git config --global --add safe.directory "*"
```

---

## 5. 最小 eval 命令（debug split）

此 smoke 將 `NUM_SEQUENCES` 暫改為 1，只驗流程通：

```bash
cd /workspace/bts/3d_diffuser_actor
export PYTHONPATH=$(pwd):$PYTHONPATH
export HYDRA_FULL_ERROR=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29555
sed -i 's/^NUM_SEQUENCES = 1000/NUM_SEQUENCES = 1/' online_evaluation_calvin/evaluate_policy.py
DS=/workspace/bts/calvin/dataset/calvin_debug_dataset
python online_evaluation_calvin/evaluate_policy.py \
  --calvin_dataset_path $DS \
  --calvin_model_path /workspace/bts/calvin/calvin_models \
  --text_encoder clip --text_max_length 16 --tasks A B C D \
  --backbone clip \
  --gripper_loc_bounds tasks/calvin_rel_traj_location_bounds_task_ABC_D.json \
  --gripper_loc_bounds_buffer 0.01 \
  --calvin_gripper_loc_bounds $DS/validation/statistics.yaml \
  --embedding_dim 192 --action_dim 7 --use_instruction 1 \
  --rotation_parametrization 6D --diffusion_timesteps 25 \
  --interpolation_length 20 --num_history 3 --relative_action 1 \
  --fps_subsampling_factor 3 --lang_enhanced 1 --save_video 0 \
  --base_log_dir /workspace/bts/eval_smoke_logs/ \
  --quaternion_format wxyz \
  --checkpoint /workspace/bts/3d_diffuser_actor/train_logs/diffuser_actor_calvin.pth
```

---

## 6. M1b 建議

M1a 已證明「技術鏈可跑」。下一步不要直接訓練，先做 **正式 eval 小批量**：

1. 固化上述 workaround 成 reproducible patch / Dockerfile（不要靠手動 hotfix）。
2. 下載正式 CALVIN 資料：理想先拿 `task_ABC_D/validation`；若官方只能整包，需評估是否下完整 `task_ABC_D.zip`（約數百 GB）。
3. 跑 10–50 sequences 的 ABC→D eval smoke，看速度與數字方向。
4. 再跑完整 1000 sequences 或進訓練。

**核心結論**：3D-DA 在 GB10 上已非紙上可行，而是 closed-loop 實測可跑。3D-DA 仍是 CALVIN backbone 主候選。
