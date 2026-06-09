---
type: handoff
aliases:
  - "BTS image/VLA route session handoff"
  - "Image route autonomous implementation handoff"
tags:
  - BTS
  - image-policy
  - VLA
  - LIBERO
  - handoff
summary: "Handoff for the image/VLA route after pivot from 3D-DA: controlled image-binding benchmark, structured BTS evidence, LIBERO Docker/scaffold, Object/Spatial diagnostics, first-contact metrics, and reproducibility docs are implemented and committed."
---

# BTS Image/VLA Route — Session Handoff（2026-06-09）

## 0. Current status

Main branch/session direction:

```text
3D-DA/CALVIN closed-loop reproduction stopped.
BTS image/VLA route is now the active path.
OpenVLA real-VLA baseline now runs spark-only on GB10 (A6000 reserved for sharing).
```

Latest milestone — OpenVLA spark-only eval works and is faithful:

```text
OpenVLA-7B loads + runs on GB10 (torch 2.12+cu130) via pure HF trust_remote_code.
No silent numeric failure (predict_action ~1.4s, finite 7D action).
LIBERO-Spatial success 9/10 = 0.90 (matches published ~0.85) -> pipeline faithful.
```

Load-bearing research findings (reshape the BTS claim):

```text
LIBERO-Spatial: distractor-instance contact = 0.00 (OpenVLA does NOT mis-bind same-class).
Relation-stripped probe: success 0.90->0.20 but distractor still 0.00 (ambiguity = aborted
  grasp, not confident mis-binding).
LIBERO-Object: stable nonzero mis-binding -> wrong-object-type 3/30.
Known failures: cream_cheese->tomato_sauce, butter->basket, chocolate_pudding->orange_juice.
Targeted rerun --pairs 1:0,6:2,8:0 reproduces 3/3 wrong contacts.
OpenVLA+BTS oracle target-gate-until-contact fixes known failures: success 0/3->2/3, wrong 3/3->0/3.
Broader 10x1 gate check: wrong 2/10->0/10 but success 7/10->3/10, so next wrapper must be selective/evidence-based.
=> BTS real-benchmark binding claim should target OBJECT IDENTITY / attribute axis, not spatial.
```

Read these for the OpenVLA work:

```text
bts-poc/experiments/LIBERO_SPATIAL_OPENVLA_BASELINE_REPORT.md
idea/bts_image_vla_spec.md  (sections 34, 35, 35.1-35.3)
bts-poc/experiments/REPRODUCIBILITY.md  (section 3b: spark-native OpenVLA env)
```

Spark-only OpenVLA env (do not rebuild; see spec §34.2 / repro §3b):

```text
venv:  ~/openvla-spark/.venv (python3.12, torch 2.12.0+cu130, transformers 4.40.1)
repos: ~/bts/openvla, ~/bts/LIBERO (editable --no-deps)
ckpts: HF_HOME=~/bts/hf_cache (libero-spatial + libero-object 7B, ~15G each)
run:   MUJOCO_GL=egl HF_HOME=~/bts/hf_cache PYTHONPATH=~/bts-poc/experiments
       OPENVLA_CHECKPOINT=<...> to switch checkpoints
```

Core claim now supported by controlled experiments:

```text
Generic image-language policies exploit spurious shortcuts and fail OOD object-attribute binding.
Generic belief heads can also fail by memorizing atomic seen compositions.
Structured object-attribute belief fixes OOD wrong-object errors.
```

Real benchmark scaffold:

```text
LIBERO Docker builds on spark (CPU diagnostics).
LIBERO native GPU env on spark for OpenVLA eval (the active route).
LIBERO-Object/Spatial diagnostics work; wrong-instance + wrong-object-type metrics in place.
First-contact object extraction works for spatial target instance.
```

---

## 1. Key docs to read first

```text
bts-poc/PROGRESS.md
bts-poc/README.md
idea/bts_image_vla_spec.md
bts-poc/experiments/REPRODUCIBILITY.md
bts-poc/experiments/CONTROLLED_IMAGE_BINDING_REPORT.md
bts-poc/experiments/LIBERO_OBJECT_DIAGNOSTICS_REPORT.md
bts-poc/experiments/LIBERO_SPATIAL_DIAGNOSTICS_REPORT.md
docs/decisions/ADR-001-bts-image-libero-openvla.md
```

---

## 2. Main implemented artifacts

### Controlled synthetic benchmark

```text
bts-poc/envs/image_binding.py
bts-poc/experiments/image_binding_v0.py
bts-poc/experiments/train_image_binding_v0.py
bts-poc/experiments/train_image_pixels_v0.py
bts-poc/experiments/image_binding_detector_v0.py
bts-poc/experiments/image_binding_corruption_v0.py
bts-poc/experiments/train_patch_classifier_v0.py
bts-poc/experiments/learned_bts_policy_v0.py
```

Headline results:

```text
shortcut OOD success ≈ 0.258, wrong_object ≈ 0.742
generic_belief OOD success ≈ 0.080, wrong_object ≈ 0.920
pixel_xy OOD success ≈ 0.244, wrong_object ≈ 0.756
bts_structured OOD success = 1.000, wrong_object = 0.000
learned evidence + BTS under corruption success ≈ 0.972-1.000
```

### LIBERO setup and diagnostics

```text
docker/libero/Dockerfile
docker/libero/build_and_run.sh
docker/libero/README.md
bts-poc/experiments/libero_object_diagnostics.py
bts-poc/experiments/libero_object_rollout_diagnostics.py
bts-poc/experiments/summarize_libero_rollouts.py
```

Spark container:

```text
image: bts_libero:latest
container: bts_libero
```

Important caveat:

```text
torch.cuda.is_available() == False inside bts_libero.
Use this container for LIBERO render/diagnostics, not OpenVLA GPU.
```

---

## 3. LIBERO findings

### LIBERO-Object

Works for object identity binding:

```text
alphabet_soup -> basket
cream_cheese -> basket
...
```

Diagnostics available:

```text
target object from language
BDDL verification
object-state positions
target distance to EEF
nearest object to EEF
first-contact object
```

### LIBERO-Spatial

Better for BTS real-benchmark claim.

All tasks are relation-conditioned black-bowl instance binding:

```text
language target: black_bowl
scene contains: akita_black_bowl_1 and akita_black_bowl_2
BDDL goal target: akita_black_bowl_1
```

Example tasks:

```text
black bowl between the plate and the ramekin -> akita_black_bowl_1
black bowl next to the ramekin -> akita_black_bowl_1
black bowl from table center -> akita_black_bowl_1
```

Spatial rollout diagnostics:

```text
target_reach_fast, tasks 0-2, 2 init states, 60 steps:
mean_target_dist_drop = 0.1659
mean_nearest_target_fraction = 0.4583
first_nearest_target_rate = 0.3333
```

Long contact validation:

```text
task 0 init 0, 200 steps:
first_contact_object = akita_black_bowl_1
first_contact_is_target = True
first_contact around t=90
nearest_target_fraction = 0.805
target_dist_final ≈ 0.021
```

---

## 4. Recent commits from this session

```text
17010c8 docs: index image VLA diagnostics progress
cabcb8a docs: add image route reproducibility manifest
cb1a44c feat: add LIBERO rollout summarizer
2660f77 feat: add LIBERO first-contact diagnostics
36c7b55 docs: summarize LIBERO spatial diagnostics
491ad15 feat: extend LIBERO rollout diagnostics to spatial
a9d14f3 feat: parse LIBERO spatial diagnostics
109215a docs: summarize LIBERO object diagnostics
d7e0c2b feat: validate LIBERO nearest-object metric
1a49cc5 feat: add LIBERO nearest-object diagnostics
597a06d feat: add LIBERO target-reaching diagnostic policy
06d393c feat: add LIBERO rollout diagnostics logger
54f7317 feat: add LIBERO object diagnostics parser
c239ea7 fix: make LIBERO Docker build work on ARM64 spark
f4c8254 chore: add isolated LIBERO Docker setup
1bce931 docs: summarize controlled image binding results
ba7b9d4 feat: integrate learned evidence with structured BTS
498e326 feat: train robust object evidence classifier
332e845 test: add image binding corruption robustness
7853795 feat: add learned pixel binding baseline
f6be873 feat: ground image binding candidates in pixels
775a8d2 feat: train structured image binding baselines
c244814 feat: add controlled image binding benchmark
2ab4b2f docs: specify BTS image VLA route
```

---

## 5. Working tree warning

Current git status still shows many unrelated modified/untracked files outside the image/VLA commits, e.g. old `bts-poc` toy files, `varibad_learning_staging`, Obsidian files, and earlier experiment artifacts.

Do **not** bulk commit all dirty files.

During this session, commits were scoped to image/VLA route files only.

---

## 6. Safe next actions

Best next engineering target:

```text
Build a generic policy-eval wrapper for LIBERO diagnostics.
```

Purpose:

```text
policy(obs, instruction) -> 7D action
logger records target/nearest/contact metrics
```

This wrapper should support:

```text
heuristic policies
future learned BTS policy
future OpenVLA policy output
```

Next research target:

```text
OpenVLA eval stack on GPU-capable environment, likely lab A6000 rather than spark bts_libero.
```

Reason:

```text
bts_libero works for LIBERO CPU/EGL diagnostics, but torch.cuda is false on GB10/CUDA11.8 container.
```

---

## 7. Do not redo

Already done:

```text
3D-DA debug stop decision
CALVIN image validation smoke
controlled image-binding benchmark
structured BTS baseline
learned object evidence integration
LIBERO Docker build
LIBERO-Object diagnostics
LIBERO-Spatial diagnostics
first-contact extraction
reproducibility manifest
```
