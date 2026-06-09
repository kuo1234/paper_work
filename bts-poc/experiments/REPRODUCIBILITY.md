# BTS Image/VLA Diagnostics Reproducibility Manifest

This file lists the commands needed to reproduce the current image-route diagnostics.

## 0. Local controlled image-binding benchmark

Run from repo root:

```bash
PYTHONPATH=bts-poc python bts-poc/experiments/image_binding_v0.py --n 1000

PYTHONPATH=bts-poc python bts-poc/experiments/train_image_binding_v0.py \
  --n-train 5000 --n-eval 2000 --epochs 20 --cpu

PYTHONPATH=bts-poc python bts-poc/experiments/image_binding_detector_v0.py --n 1000

PYTHONPATH=bts-poc python bts-poc/experiments/image_binding_corruption_v0.py \
  --n 500 --seeds 5

PYTHONPATH=bts-poc python bts-poc/experiments/train_image_pixels_v0.py \
  --n-train 3000 --n-eval 1000 --epochs 10 --batch-size 128 --cpu

PYTHONPATH=bts-poc python bts-poc/experiments/train_patch_classifier_v0.py \
  --n-train-scenes 500 --n-eval-scenes 300 --epochs 6 --cpu

PYTHONPATH=bts-poc python bts-poc/experiments/learned_bts_policy_v0.py \
  --n-train-scenes 500 --n-eval 500 --epochs 6 --cpu
```

Reports:

```text
bts-poc/experiments/CONTROLLED_IMAGE_BINDING_REPORT.md
```

Expected headline:

```text
generic pixel_xy OOD success ≈ 0.244
structured BTS OOD success = 1.000
learned evidence + BTS OOD corruption success ≈ 0.972-1.000
```

---

## 1. Spark LIBERO Docker setup

On spark:

```bash
cd ~/HDD/bts/paperwork/docker/libero
bash build_and_run.sh
```

Expected:

```text
image: bts_libero:latest
container: bts_libero
```

Smoke:

```bash
docker exec bts_libero bash -lc 'python - <<PY
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
from libero.libero import benchmark
print("benchmark import ok")
PY'
```

Note: on spark GB10, `torch.cuda.is_available()` in this CUDA 11.8 container is expected to be false. Use it for LIBERO render/diagnostics, not OpenVLA GPU.

---

## 2. LIBERO metadata diagnostics

Copy updated scripts to spark if needed:

```bash
scp bts-poc/experiments/libero_object_diagnostics.py \
  p76141495@192.168.65.11:~/HDD/bts/libero_object_diagnostics.py
ssh p76141495@192.168.65.11 \
  'docker cp ~/HDD/bts/libero_object_diagnostics.py bts_libero:/workspace/libero_object_diagnostics.py'
```

LIBERO-Object:

```bash
ssh p76141495@192.168.65.11 'docker exec bts_libero bash -lc "
python /workspace/libero_object_diagnostics.py \
  --suite libero_object \
  --render-one \
  --out /workspace/bts/libero_object_diagnostics.json
"'
```

LIBERO-Spatial:

```bash
ssh p76141495@192.168.65.11 'docker exec bts_libero bash -lc "
python /workspace/libero_object_diagnostics.py \
  --suite libero_spatial \
  --render-one \
  --out /workspace/bts/libero_spatial_diagnostics_v2.json
"'
```

Expected LIBERO-Spatial headline:

```text
black_bowl + relation phrase -> goal akita_black_bowl_1 -> plate_1
obs has akita_black_bowl_1_pos and akita_black_bowl_2_pos
```

---

## 3. LIBERO rollout diagnostics

Copy updated rollout scripts:

```bash
scp bts-poc/experiments/libero_object_rollout_diagnostics.py \
  p76141495@192.168.65.11:~/HDD/bts/libero_object_rollout_diagnostics.py
scp bts-poc/experiments/summarize_libero_rollouts.py \
  p76141495@192.168.65.11:~/HDD/bts/summarize_libero_rollouts.py
ssh p76141495@192.168.65.11 '
  docker cp ~/HDD/bts/libero_object_rollout_diagnostics.py bts_libero:/workspace/libero_object_rollout_diagnostics.py
  docker cp ~/HDD/bts/summarize_libero_rollouts.py bts_libero:/workspace/summarize_libero_rollouts.py
'
```

LIBERO-Object policy comparison:

```bash
ssh p76141495@192.168.65.11 'docker exec bts_libero bash -lc "
for p in noop random target_reach target_reach_fast; do
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python /workspace/libero_object_rollout_diagnostics.py \
    --suite libero_object \
    --tasks 3 --inits 2 --steps 60 \
    --policy $p \
    --out /workspace/bts/libero_object_rollout_diag_${p}.json
done
"'
```

LIBERO-Spatial relation/instance diagnostics:

```bash
ssh p76141495@192.168.65.11 'docker exec bts_libero bash -lc "
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python /workspace/libero_object_rollout_diagnostics.py \
  --suite libero_spatial \
  --tasks 3 --inits 2 --steps 60 \
  --policy target_reach_fast \
  --out /workspace/bts/libero_spatial_rollout_diag_target_reach_fast_v2.json
"'
```

Long first-contact validation:

```bash
ssh p76141495@192.168.65.11 'docker exec bts_libero bash -lc "
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python /workspace/libero_object_rollout_diagnostics.py \
  --suite libero_spatial \
  --tasks 1 --inits 1 --steps 200 \
  --policy target_reach_fast \
  --out /workspace/bts/libero_spatial_contact_long.json
"'
```

Summarize:

```bash
ssh p76141495@192.168.65.11 'docker exec bts_libero bash -lc "
python /workspace/summarize_libero_rollouts.py \
  /workspace/bts/libero_object_rollout_diag_noop_v2.json \
  /workspace/bts/libero_object_rollout_diag_random_v2.json \
  /workspace/bts/libero_object_rollout_diag_target_reach_v2.json \
  /workspace/bts/libero_object_rollout_diag_target_reach_fast.json \
  /workspace/bts/libero_spatial_rollout_diag_target_reach_fast_v2.json \
  /workspace/bts/libero_spatial_contact_long.json \
  --out /workspace/bts/libero_rollout_summary.json
"'
```

Reports:

```text
bts-poc/experiments/LIBERO_OBJECT_DIAGNOSTICS_REPORT.md
bts-poc/experiments/LIBERO_SPATIAL_DIAGNOSTICS_REPORT.md
```

Expected headline:

```text
LIBERO-Spatial target_reach_fast:
  mean_nearest_target_fraction ≈ 0.4583
  first_nearest_target_rate ≈ 0.3333

LIBERO-Spatial long contact:
  first_contact_object = akita_black_bowl_1
  first_contact_is_target = True
```

---

## 3b. Spark-native OpenVLA GPU eval (spark-only route)

The CPU Docker (§1) is for LIBERO render/diagnostics only. OpenVLA GPU eval runs in a
**separate native spark venv** that keeps the only CUDA-capable aarch64 torch
(`2.12.0+cu130`) and installs the OpenVLA/LIBERO stack around it. See spec §34 for rationale.

One-time env build on spark:

```bash
ssh p76141495@192.168.65.11
python3 -m venv ~/openvla-spark/.venv
~/openvla-spark/.venv/bin/pip install --upgrade pip
~/openvla-spark/.venv/bin/pip install torch==2.12.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu130

mkdir -p ~/bts && cd ~/bts
git clone --depth 1 https://github.com/openvla/openvla.git
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git

# OpenVLA eval-critical deps (keep cu130 torch; skip training-only TF/dlimp)
cd ~/bts/openvla && ~/openvla-spark/.venv/bin/pip install \
  transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 \
  draccus==0.8.0 peft==0.11.1 "accelerate>=0.25.0" \
  einops huggingface_hub json-numpy jsonlines rich sentencepiece protobuf
~/openvla-spark/.venv/bin/pip install -e . --no-deps

# LIBERO runtime deps (skip pinned old transformers/numpy)
cd ~/bts/LIBERO && ~/openvla-spark/.venv/bin/pip install \
  robosuite==1.4.0 bddl==1.0.1 hydra-core==1.2.0 easydict gym==0.25.2 \
  cloudpickle future thop robomimic
~/openvla-spark/.venv/bin/pip install -e . --no-deps

# Editable finder for LIBERO does not resolve; add a plain .pth fallback
echo "/home/p76141495/bts/LIBERO" > \
  ~/openvla-spark/.venv/lib/python3.12/site-packages/libero_repo.pth

# Preseed ~/.libero/config.yaml (avoids interactive dataset-path prompt); see spec §34.2
```

Checkpoint download:

```bash
ssh p76141495@192.168.65.11 'export HF_HOME=~/bts/hf_cache; \
  ~/openvla-spark/.venv/bin/python -c "from huggingface_hub import snapshot_download; \
  print(snapshot_download(\"openvla/openvla-7b-finetuned-libero-spatial\"))"'
```

Load + inference smoke (decisive GB10 check):

```bash
ssh p76141495@192.168.65.11 'cd ~/bts && export HF_HOME=~/bts/hf_cache; \
  ~/openvla-spark/.venv/bin/python test_openvla_load.py'
# Expect: predict_action ~1.4s, 7D finite nonzero action, SMOKE_OK
```

OpenVLA LIBERO-Spatial eval through the existing diagnostics logger:

```bash
scp bts-poc/experiments/openvla_policy_adapter.py \
    bts-poc/experiments/libero_object_rollout_diagnostics.py \
    p76141495@192.168.65.11:~/bts-poc/experiments/

ssh p76141495@192.168.65.11 'cd ~/bts-poc/experiments && \
  export MUJOCO_GL=egl HF_HOME=~/bts/hf_cache PYTHONPATH=~/bts-poc/experiments && \
  ~/openvla-spark/.venv/bin/python libero_object_rollout_diagnostics.py \
    --suite libero_spatial --tasks 5 --inits 2 --steps 280 \
    --warmup-steps 10 --camera-size 256 \
    --policy external --policy-adapter openvla_policy_adapter:openvla_policy \
    --out runs/openvla_spatial_sweep5x2.json'
```

Critical fidelity details (without these OpenVLA looks broken — drift, success 0):

```text
agentview_image rotated 180 deg (img[::-1, ::-1]) before resize to 224
gripper: normalize [0,1]->[-1,+1] binarize, then invert sign
10 dummy no-op warmup steps (objects fall/settle) + env.seed(0)
center_crop True (finetuned checkpoints trained with image aug)
```

Expected single-rollout sanity (faithful preprocessing):

```text
success 1/1, target_dist_drop +0.317, nearest_target_fraction 0.537,
distractor_instance_contact_rate 0.0
(vs broken: drop -0.298, success 0 — before the preprocessing fix)
```


```text
e375631 docs: pivot BTS from 3D-DA to image route
dcef0b1 docs: record CALVIN image smoke tests
2ab4b2f docs: specify BTS image VLA route
c244814 feat: add controlled image binding benchmark
775a8d2 feat: train structured image binding baselines
f6be873 feat: ground image binding candidates in pixels
7853795 feat: add learned pixel binding baseline
332e845 test: add image binding corruption robustness
498e326 feat: train robust object evidence classifier
ba7b9d4 feat: integrate learned evidence with structured BTS
1bce931 docs: summarize controlled image binding results
f4c8254 chore: add isolated LIBERO Docker setup
c239ea7 fix: make LIBERO Docker build work on ARM64 spark
54f7317 feat: add LIBERO object diagnostics parser
06d393c feat: add LIBERO rollout diagnostics logger
597a06d feat: add LIBERO target-reaching diagnostic policy
1a49cc5 feat: add LIBERO nearest-object diagnostics
d7e0c2b feat: validate LIBERO nearest-object metric
109215a docs: summarize LIBERO object diagnostics
a9d14f3 feat: parse LIBERO spatial diagnostics
491ad15 feat: extend LIBERO rollout diagnostics to spatial
2660f77 feat: add LIBERO first-contact diagnostics
cb1a44c feat: add LIBERO rollout summarizer
```
