---
type: research-spec
aliases:
  - "BTS image VLA spec"
  - "Belief over object-attribute binding for image-language manipulation"
tags:
  - 研究主線
  - BTS
  - image-policy
  - VLA
  - OpenVLA
  - LIBERO
  - binding
summary: "BTS 全面轉向 image route 的完整規格：先做 controlled image-binding benchmark，再接 LIBERO，最後與 OpenVLA 比較；核心是 structured belief over object-attribute bindings。"
---

# BTS Image/VLA Route — Complete Spec（2026-06-08）

## 0. One-line objective

Build and evaluate **BTS as a structured belief module for object-attribute binding in image-language manipulation**, then compare against strong image policies and an open VLA baseline (**OpenVLA**).

Core claim:

> Explicit belief over object-attribute bindings improves robustness and reduces wrong-object / wrong-attribute errors in image-language manipulation, including when attached to modern VLA-style backbones.

---

## 1. Why pivot from 3D-DA/CALVIN to image/VLA

M1b-6 showed 3D-Diffuser-Actor CALVIN closed-loop reproduction is not reliable enough to be a foundation for BTS:

- GB10 100-seq old checkpoint: avg seq len ≈ 0.57.
- A6000 N=10 reference: `50 / 30 / 10 / 10 / 0`, avg=1.0.
- Extensive checks ruled out checkpoint load, flags, validation extraction, FPS fallback, precision, CPU-vs-GPU forward mismatch, scheduler, language truncation, render/camera obvious issues.
- Continuing 3D-DA reproduction has low marginal value.

Image route preserves the scientific point while avoiding point cloud/FPS/reproduction bottlenecks.

---

## 2. Benchmark strategy

Do **not** make CALVIN image the only path. Use three layers.

### Stage A — Controlled image-binding benchmark（main mechanism benchmark）

Purpose: prove BTS mechanism cleanly.

Requirements:

- RGB observation.
- Multiple objects.
- Attribute-object language, e.g. `push the red cube`, `pick the blue sphere`.
- Spurious shortcut train distribution.
- OOD attribute-object recombination test.
- Direct wrong-object / wrong-attribute metrics.
- Fast training/eval.

This can be a synthetic tabletop/gridworld-image environment first. It should extend the existing toy/P3 insight into pixels.

### Stage B — LIBERO（main real benchmark）

Purpose: demonstrate BTS on a modern language-conditioned manipulation benchmark.

Why LIBERO over CALVIN as primary real benchmark:

- More aligned with modern VLA / imitation learning ecosystem.
- Better for language-conditioned generalization suites.
- Less tied to old CALVIN reproduction issues.
- More suitable for OpenVLA-style comparison.

Need to verify:

- Which LIBERO suites have object/attribute binding.
- Availability of image observations and language tasks.
- Whether state labels can provide target-object diagnostics.
- Training/eval cost on spark/lab.

### Stage C — CALVIN image（secondary / optional benchmark）

Purpose: retain continuity with prior CALVIN work.

Current status:

- Validation image data works.
- Minimal dataloader passed.
- Binding parser coverage good.
- Tiny validation-only image overfit passed.
- Training split is absent locally.

Use CALVIN only if small/large training extraction becomes practical. Do not let CALVIN training data block the image/VLA route.

---

## 3. Baseline stack

### Level 0 — Minimal image BC baseline（debug baseline）

Goal: fast sanity.

Input:

```text
static RGB + gripper RGB + proprio + language/task embedding
```

Model:

```text
small CNN/ResNet + language embedding + MLP/Transformer action head
```

Output:

```text
7D relative action or short action chunk
```

Use for:

- dataloader validation
- tiny overfit
- diagnostic plumbing
- first closed-loop smoke

### Level 1 — Strong non-VLA baseline

Use one of:

1. **ACT-style action chunking**
2. **Diffusion Policy-style action decoder**

Recommendation: start with ACT-style chunking before diffusion.

Reason:

- Easier to implement/debug.
- More stable for small data.
- Action chunking fits manipulation.
- Diffusion adds sampling complexity; use later for stronger baseline.

### Level 2 — VLA comparison

Primary VLA:

```text
OpenVLA
```

Secondary design reference:

```text
Octo
```

Do not start with RT-1/RT-2/pi0 as experimental baselines because reproducibility/access is weaker.

OpenVLA comparison modes, in increasing cost:

1. frozen OpenVLA features + lightweight action head
2. OpenVLA LoRA fine-tuning
3. OpenVLA + BTS belief module + action head
4. full fine-tune only if resources allow

---

## 4. BTS model design

### 4.1 Abstract architecture

```text
RGB observation + language + proprio
  -> visual-language encoder
  -> object/query evidence tokens
  -> BTS belief module over object-attribute bindings
  -> belief-conditioned action decoder
```

### 4.2 Belief state

Belief should represent uncertainty over bindings, not just be another hidden token.

Candidate variables:

```text
B_t(object, attribute, role)
```

Example:

```text
P(target = red_block | image, instruction, history)
P(target = blue_block | image, instruction, history)
P(target = drawer | image, instruction, history)
```

Roles:

- target object
- source object
- receptacle / destination
- interactable affordance（drawer handle, button, switch）

### 4.3 Implementation variants

#### Variant A — Belief tokens（fastest）

```text
visual-language tokens
  -> K learned belief query tokens via cross-attention
  -> action decoder
```

Pros: easy, differentiable, minimal labels needed.  
Cons: less interpretable unless probed.

#### Variant B — Explicit target distribution（diagnostic-friendly）

```text
object/query tokens
  -> target classifier over candidate objects
  -> belief embedding = weighted object tokens
  -> action decoder
```

Pros: clear wrong-object metrics, auxiliary loss possible.  
Cons: needs object candidates / pseudo labels.

#### Variant C — Temporal Bayesian-style belief update（most BTS-like）

```text
B_{t+1} = Update(B_t, visual evidence_t, language, action history)
```

Pros: closest to BTS novelty.  
Cons: more engineering; should be Stage 2 after A/B.

Recommended first implementation:

```text
Variant B-lite:
  object/query tokens from CNN/ViT feature map
  target distribution supervised by pseudo labels when available
  weighted belief token conditions ACT-style action decoder
```

---

## 5. Diagnostics and metrics

Core metrics beyond success:

### 5.1 Binding metrics

```text
target object accuracy
wrong-object error rate
wrong-attribute error rate
first-contact object accuracy
belief entropy / calibration
OOD attribute-object recombination success
```

### 5.2 Policy metrics

```text
offline action MSE / L1
chunk prediction loss
closed-loop success
long-horizon avg seq len（only for CALVIN-like eval）
```

### 5.3 Required ablations

```text
baseline image policy
+ belief tokens only
+ explicit target distribution
+ auxiliary binding loss
+ shuffled-language stress test
+ spurious shortcut split
+ oracle target upper bound（analysis only）
```

---

## 6. Dataset and label plan

### 6.1 Controlled image-binding benchmark

Generate labels directly:

```text
target object id
object attributes
first contact object
success/failure
wrong object / wrong attribute
```

This is the cleanest proof of mechanism.

### 6.2 LIBERO

Need investigate:

- available simulator state
- object names / task metadata
- whether target object can be extracted from language/task definitions
- whether first-contact object can be logged

### 6.3 CALVIN

Already verified validation parser:

```text
windows: 1087
unparsed_target: 0
binding_sensitive_count: 782
color_block_count: 673
```

Target counts:

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

Need training split before formal training.

---

## 7. OpenVLA comparison design

### 7.1 Why OpenVLA

OpenVLA is the primary VLA comparison because:

- open-source
- representative modern VLA baseline
- reviewer-recognizable
- supports fine-tuning / adaptation workflows
- lets BTS claim be tested against a strong image-language-action backbone

Comparison statement:

> BTS is not a replacement for VLA scale; it is a structured belief layer that can complement VLA representations and reduce binding errors.

### 7.2 OpenVLA experiment ladder

#### OpenVLA-Frozen

```text
image + language -> frozen OpenVLA representations -> action head
```

Use as first VLA-level sanity.

#### OpenVLA-LoRA

```text
LoRA fine-tune OpenVLA on benchmark demonstrations
```

Use if resources allow.

#### OpenVLA+BTS

```text
OpenVLA visual-language tokens
  -> BTS target/binding belief module
  -> action decoder
```

Evaluate whether BTS reduces binding-specific failures compared with OpenVLA-only.

### 7.3 What not to claim

Do not claim BTS beats all VLAs globally unless evaluated at that scale.

Claim should be narrower:

```text
BTS improves binding-sensitive tasks and diagnostic errors under matched data/backbone settings.
```

---

## 8. Execution milestones

### M2-0 — Spec + smoke（current）

Done / in progress:

- CALVIN image dataloader smoke passed.
- CALVIN binding parser v2 passed.
- CALVIN tiny validation-only overfit passed.
- This spec created.

Exit condition:

```text
spec written, committed, next implementation path clear
```

### M2-1 — Controlled image-binding benchmark

Deliverables:

- environment generator
- RGB renderer
- language generator
- train/OOD split
- wrong-object diagnostics
- minimal BC baseline

Exit condition:

```text
baseline learns ID split; fails more on OOD binding split
```

### M2-2 — BTS on controlled benchmark

Deliverables:

- belief module v1
- target distribution diagnostic
- BTS vs baseline result

Exit condition:

```text
BTS improves OOD binding and reduces wrong-object errors
```

### M2-3 — LIBERO feasibility

Deliverables:

- install/run LIBERO smoke
- inspect data/state/task metadata
- identify binding-sensitive tasks
- choose first suite

Exit condition:

```text
LIBERO dataloader/eval feasible, binding diagnostics possible
```

### M2-4 — Strong baseline

Deliverables:

- ACT-style or Diffusion Policy-style baseline
- matched data split
- diagnostic metrics

Exit condition:

```text
strong non-VLA baseline established
```

### M2-5 — OpenVLA comparison

Deliverables:

- OpenVLA frozen or LoRA baseline
- OpenVLA+BTS integration
- binding diagnostic comparison

Exit condition:

```text
BTS effect tested against VLA baseline
```

---

## 9. Immediate next actions

### Action 1 — Build controlled image-binding benchmark spec-to-code

Start with a simple synthetic image environment:

```text
64x64 or 128x128 RGB
3-5 colored objects
language instruction target color/object/action
simple 2D action or discrete action first
train distribution with spurious shortcuts
OOD split swaps color/object/location correlations
```

This does not need robot physics at first. It is mechanism validation.

### Action 2 — Keep CALVIN as diagnostic reference

Do not download full CALVIN training yet. Keep current validation tools for:

- parser testing
- diagnostic design
- future benchmark continuity

### Action 3 — Research LIBERO setup

Check:

- installation cost
- dataset size
- image/action format
- object metadata
- OpenVLA compatibility

### Action 4 — Prepare reading pack

Priority reading:

1. OpenVLA
2. LIBERO
3. ACT
4. Diffusion Policy
5. Slot Attention
6. CLIPort
7. Octo

---

## 10. Main risks

### Risk A — benchmark too toy-like

Mitigation: controlled benchmark is Stage A only; real benchmark is LIBERO/OpenVLA.

### Risk B — OpenVLA too expensive

Mitigation: start frozen features + action head; LoRA only if needed.

### Risk C — belief module becomes generic attention

Mitigation: require explicit binding diagnostics and target distribution/probe.

### Risk D — no clean object labels in real benchmark

Mitigation: use language/task metadata + simulator state where available; otherwise report first-contact object with approximate object regions.

---

## 11. Success criteria for the image route

Minimum publishable evidence should include:

1. Controlled benchmark where BTS reduces wrong-object errors under OOD binding.
2. Real benchmark evidence on LIBERO or CALVIN showing same trend on binding-sensitive tasks.
3. Comparison to strong non-VLA policy baseline.
4. At least one VLA comparison, preferably OpenVLA frozen/LoRA, showing BTS complements VLA representations.

Target story:

```text
Large image-language policies can still bind the right attribute to the wrong object.
BTS introduces an explicit, temporally maintained belief over bindings.
This reduces wrong-object and wrong-attribute errors, especially under compositional shifts.
```


---

## 12. LIBERO/OpenVLA research update（2026-06-08）

This update is recorded as `docs/decisions/ADR-001-bts-image-libero-openvla.md`.

### 12.1 Benchmark choice

Recommended real-benchmark start:

```text
1. LIBERO-Object
2. LIBERO-Spatial
3. LIBERO-10 / Long only later
```

Reason:

- LIBERO-Object is closest to object identity / object-binding diagnostics.
- LIBERO-Spatial is the next step for relational binding.
- LIBERO-10/Long introduce horizon/confounding before binding diagnostics are stable.
- LIBERO-Goal stresses goal variation more than pure object-attribute binding.

### 12.2 Practical setup notes

LIBERO native setup is older/simulator-heavy:

```bash
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

Datasets:

```bash
python benchmark_scripts/download_libero_datasets.py
# or
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

Suites:

```text
libero_spatial
libero_object
libero_goal
libero_10
libero_90
libero_100
```

Simulator uses MuJoCo / robosuite-style offscreen rendering. Expect EGL friction (`MUJOCO_EGL_DEVICE_ID`, `CUDA_VISIBLE_DEVICES`).

### 12.3 OpenVLA setup notes

OpenVLA stack is newer:

```text
Python 3.10
PyTorch 2.2.x
transformers==4.40.1
flash-attn==2.5.5
A100 recommended for full reproduction
LoRA smaller but still ~27GB+ VRAM
```

OpenVLA has a LIBERO eval path:

```bash
pip install -r experiments/robot/libero/libero_requirements.txt
python experiments/robot/libero/run_libero_eval.py   --model_family openvla   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object   --task_suite_name libero_object   --center_crop True
```

Default eval is 500 trials = 10 tasks × 50 episodes. Start with much smaller trial count.

Available checkpoints include:

```text
openvla/openvla-7b-finetuned-libero-spatial
openvla/openvla-7b-finetuned-libero-object
openvla/openvla-7b-finetuned-libero-goal
openvla/openvla-7b-finetuned-libero-10
```

Modified LIBERO RLDS data from OpenVLA is about 10GB total:

```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

### 12.4 Image/action compatibility

LIBERO native env example:

```python
from libero.libero.envs import OffScreenRenderEnv

env = OffScreenRenderEnv(
    bddl_file_name=task_bddl_file,
    camera_heights=128,
    camera_widths=128,
)
obs, reward, done, info = env.step([0.] * 7)
```

Expected:

```text
image: RGB camera frames, commonly 128x128
action: 7D continuous
reward: sparse, success gives +1
```

OpenVLA inference expects:

```text
PIL image
prompt: "In: What action should the robot take to {instruction}?
Out:"
output: 7-DoF continuous action via predict_action(...)
```

OpenVLA-OFT is worth checking later because it uses two images + proprio:

```text
full_image
wrist_image
state
task_description
num_images_in_input = 2
use_proprio = True
```

### 12.5 Diagnostics feasibility

LIBERO exposes task metadata:

```text
task.name
task.language
task.problem_folder
task.bddl_file
task_suite.get_task_init_states(task_id)
```

BDDL path construction:

```python
task_bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file,
)
```

BTS diagnostics likely path:

1. Parse task language.
2. Parse BDDL goal predicates.
3. Group by object names / attributes / relations.
4. Log success by object-pair, distractor, relation, template, init state.
5. Inspect robosuite object names / simulator state for first-contact or nearest-object proxies.

Caveat: object metadata is feasible but not plug-and-play; expect custom BDDL/simulator-state inspection.

### 12.6 Updated execution order

```text
M2-1: Controlled image-binding benchmark v0
M2-2: BTS on controlled benchmark
M2-3a: LIBERO-Object install/eval smoke
M2-3b: LIBERO BDDL diagnostic parser
M2-4: ACT/Diffusion strong baseline
M2-5: OpenVLA LIBERO-Object small eval
M2-6: OpenVLA+BTS integration
```

Do not start with full OpenVLA fine-tuning. First prove mechanism, then evaluate existing OpenVLA checkpoint on small LIBERO-Object runs.


---

## 13. Controlled image-binding benchmark v0 result（2026-06-08）

Implemented:

```text
bts-poc/envs/image_binding.py
bts-poc/experiments/image_binding_v0.py
```

Design:

- 96×96 RGB scenes.
- 4 colored objects per scene.
- Attributes: color × shape.
- Instruction: `pick the {color} {shape}`.
- Action abstraction: discrete object selection. This isolates binding before robot dynamics.
- Train/ID split includes a deliberate color-location shortcut.
- OOD split holds out selected color-shape target pairs and breaks the location shortcut.
- Direct diagnostics: success, wrong_object, wrong_color, wrong_shape.

Held-out OOD target pairs:

```text
red triangle
blue square
green circle
```

Smoke command:

```bash
PYTHONPATH=bts-poc python bts-poc/experiments/image_binding_v0.py --n 1000 --save-images
```

Result:

```text
location_prior train success=0.989 wrong_object=0.011 wrong_color=0.006 wrong_shape=0.006
location_prior id    success=0.982 wrong_object=0.018 wrong_color=0.012 wrong_shape=0.011
location_prior ood   success=0.256 wrong_object=0.744 wrong_color=0.454 wrong_shape=0.454

oracle_language train success=1.000 wrong_object=0.000 wrong_color=0.000 wrong_shape=0.000
oracle_language id    success=1.000 wrong_object=0.000 wrong_color=0.000 wrong_shape=0.000
oracle_language ood   success=1.000 wrong_object=0.000 wrong_color=0.000 wrong_shape=0.000
```

Interpretation:

- The benchmark successfully creates a spurious shortcut: location prior works in train/ID and collapses on OOD.
- OOD failure is mostly wrong-object with both wrong-color and wrong-shape components.
- Oracle language binding solves OOD, proving the task is solvable when object-attribute binding is correct.

Next implementation step:

```text
Train a small image policy baseline on this benchmark, then add BTS target-belief supervision and compare OOD wrong-object rate.
```


---

## 14. Trainable controlled image-binding baselines（2026-06-08）

Implemented:

```text
bts-poc/experiments/train_image_binding_v0.py
```

This trains three policies on the controlled benchmark:

1. `shortcut`: instruction-only xy regressor. It learns train-time color-location shortcut.
2. `generic_belief`: generic MLP scorer over instruction-object features. It can memorize seen color-shape pairs and treat held-out pairs as negatives.
3. `bts_structured`: decomposed structured belief scorer with separate color-match and shape-match terms. It cannot memorize atomic color-shape pairs, so it generalizes to held-out bindings.

Command:

```bash
PYTHONPATH=bts-poc python bts-poc/experiments/train_image_binding_v0.py   --n-train 5000 --n-eval 2000 --epochs 20 --cpu
```

Result after first attempt:

```text
shortcut ood success≈0.27 wrong_object≈0.73
generic_belief ood success≈0.08 wrong_object≈0.92
```

Diagnosis: generic MLP belief is not enough. It memorizes training compositions and fails worse than the shortcut on held-out color-shape pairs. This is exactly the failure mode BTS must avoid: belief must be **structured**, not just another attention/MLP head.

Fixed with `bts_structured` decomposed scorer:

```text
shortcut train success=0.986 wrong_object=0.014 wrong_color=0.010 wrong_shape=0.008
shortcut id    success=0.988 wrong_object=0.012 wrong_color=0.006 wrong_shape=0.009
shortcut ood   success=0.258 wrong_object=0.742 wrong_color=0.475 wrong_shape=0.422

generic_belief train success=1.000 wrong_object=0.000 wrong_color=0.000 wrong_shape=0.000
generic_belief id    success=1.000 wrong_object=0.000 wrong_color=0.000 wrong_shape=0.000
generic_belief ood   success=0.080 wrong_object=0.920 wrong_color=0.641 wrong_shape=0.465

bts_structured train success=1.000 wrong_object=0.000 wrong_color=0.000 wrong_shape=0.000
bts_structured id    success=1.000 wrong_object=0.000 wrong_color=0.000 wrong_shape=0.000
bts_structured ood   success=1.000 wrong_object=0.000 wrong_color=0.000 wrong_shape=0.000
```

Key finding:

> A generic learned belief head is insufficient under compositional OOD. The benefit comes from explicit factorization of object-attribute binding, not from adding capacity.

This is strong support for the image-route BTS story.

Next fix:

```text
Move from metadata object candidates to image-derived candidates or rendered object patches, so the structured belief remains image-grounded rather than oracle-metadata-grounded.
```


---

## 15. Image-grounded candidate extraction v0（2026-06-08）

Implemented:

```text
bts-poc/envs/image_binding.py::extract_image_candidates
bts-poc/experiments/image_binding_detector_v0.py
```

Purpose: remove oracle metadata from the structured belief result. v0 uses deterministic pixel segmentation of the synthetic rendered objects:

- color by nearest known RGB palette
- connected components by color mask
- centroid from component pixels
- shape by component fill ratio
  - square ≈ high fill
  - circle ≈ medium fill
  - triangle ≈ low fill

Command:

```bash
PYTHONPATH=bts-poc python bts-poc/experiments/image_binding_detector_v0.py --n 1000
```

Result:

```text
train success=1.000 wrong_object=0.000 wrong_color=0.000 wrong_shape=0.000 detected_all=1.000 avg_candidates=4.00
id    success=1.000 wrong_object=0.000 wrong_color=0.000 wrong_shape=0.000 detected_all=1.000 avg_candidates=4.00
ood   success=1.000 wrong_object=0.000 wrong_color=0.000 wrong_shape=0.000 detected_all=1.000 avg_candidates=4.00
```

Interpretation:

- Structured binding now works from image-derived candidates, not ground-truth object metadata.
- This is still a deterministic synthetic detector, not a learned perception model.
- It validates the staged path: image → candidates → structured belief → action.

Next fix:

```text
Add a learned pixel baseline / learned candidate classifier so the comparison is not only deterministic parser vs shortcut.
```


---

## 16. Learned pixel baseline v0（2026-06-08）

Implemented:

```text
bts-poc/experiments/train_image_pixels_v0.py
```

Model:

```text
RGB image + instruction one-hot -> CNN -> target xy regression -> nearest object diagnostic
```

Command:

```bash
PYTHONPATH=bts-poc python bts-poc/experiments/train_image_pixels_v0.py   --n-train 3000 --n-eval 1000 --epochs 10 --batch-size 128 --cpu
```

Result:

```text
pixel_xy train success=0.983 wrong_object=0.017 wrong_color=0.011 wrong_shape=0.011 loss=0.0050
pixel_xy id    success=0.989 wrong_object=0.011 wrong_color=0.007 wrong_shape=0.009 loss=0.0049
pixel_xy ood   success=0.244 wrong_object=0.756 wrong_color=0.474 wrong_shape=0.454 loss=0.1051
```

Interpretation:

- A generic learned image+instruction policy also learns the spurious color-location shortcut.
- It performs near-perfectly on train/ID but collapses on held-out binding OOD, like the handcrafted location-prior baseline.
- This strengthens the BTS motivation: image grounding alone is insufficient when the training distribution supports a shortcut.

Current controlled benchmark comparison:

```text
pixel_xy generic policy OOD:      success 0.244, wrong_object 0.756
shortcut prior OOD:               success 0.258, wrong_object 0.742
generic MLP belief OOD:           success 0.080, wrong_object 0.920
structured BTS belief OOD:        success 1.000, wrong_object 0.000
image-derived structured binding: success 1.000, wrong_object 0.000
```

Next fix:

```text
Replace deterministic image candidate extractor with a learned/soft candidate module, or add noise/occlusion to make v0 less trivially segmentable.
```


---

## 17. Image-grounded binding corruption robustness（2026-06-08）

Implemented:

```text
bts-poc/experiments/image_binding_corruption_v0.py
```

Initial stress test found a real bug: the deterministic color threshold in `extract_image_candidates` was too tight (`dist < 40`), so Gaussian noise destroyed detections:

```text
noise20 success≈0.279 detected_all≈0.008
noise40 success≈0.254 detected_all≈0.000
```

Fix:

```text
increase palette distance threshold to dist < 140
```

Rerun command:

```bash
PYTHONPATH=bts-poc python bts-poc/experiments/image_binding_corruption_v0.py --n 500 --seeds 5
```

Post-fix OOD robustness:

```text
clean          success=1.000±0.000 detected_all=1.000 avg_candidates=4.00
noise20        success=1.000±0.000 detected_all=1.000 avg_candidates=4.00
noise40        success=0.856±0.011 detected_all=1.000 avg_candidates=4.00
occ12          success=0.954±0.008 detected_all=0.998 avg_candidates=4.00
noise20_occ12  success=0.959±0.008 detected_all=0.999 avg_candidates=4.00
```

Interpretation:

- The image-grounded structured binding path is robust to moderate noise and occlusion after threshold fix.
- Severe noise (`std=40`) mostly hurts shape classification, not object detection.
- This gives a concrete next perception target: learned/soft shape evidence should replace brittle fill-ratio classification.


---

## 18. Learned object evidence classifier v0（2026-06-08）

Implemented:

```text
bts-poc/experiments/train_patch_classifier_v0.py
```

Purpose: replace brittle deterministic fill-ratio evidence with learned color/shape patch evidence.

Initial attempt trained only on `noise20+occ8`, which over-specialized to that corruption regime:

```text
clean shape≈0.708
occ12 shape≈0.674
noise20 shape=1.000
```

Fix: mixed-corruption training with clean, noise, occlusion, and noise+occlusion datasets.

Command:

```bash
PYTHONPATH=bts-poc python bts-poc/experiments/train_patch_classifier_v0.py   --n-train-scenes 500 --n-eval-scenes 300 --epochs 6 --cpu
```

Post-fix result:

```text
clean          color=1.000 shape=1.000 both=1.000 loss=0.0645
noise20        color=1.000 shape=0.999 both=0.999 loss=0.0311
noise40        color=1.000 shape=0.989 both=0.989 loss=0.0533
occ12          color=1.000 shape=0.968 both=0.968 loss=0.2335
noise20_occ12  color=1.000 shape=0.973 both=0.973 loss=0.1571
```

Interpretation:

- Learned object evidence is robust across clean/noise/occlusion when trained on a mixed corruption distribution.
- This is the next bridge from synthetic deterministic parsing to VLA-compatible learned perception tokens.
- Remaining gap: integrate learned patch evidence into the structured BTS belief policy end-to-end.


---

## 19. Learned evidence + structured BTS policy v0（2026-06-08）

Implemented:

```text
bts-poc/experiments/learned_bts_policy_v0.py
```

Pipeline:

```text
RGB image
  -> deterministic color connected components for candidate locations
  -> learned patch classifier for color/shape evidence
  -> structured BTS score = log P(color target) + log P(shape target)
  -> select object
```

This integrates learned object evidence with structured binding. It is no longer using ground-truth object metadata or deterministic shape fill-ratio for the final decision.

Command:

```bash
PYTHONPATH=bts-poc python bts-poc/experiments/learned_bts_policy_v0.py   --n-train-scenes 500 --n-eval 500 --epochs 6 --cpu
```

OOD result:

```text
clean          success=1.000 wrong_object=0.000 wrong_color=0.000 wrong_shape=0.000 detected_all=1.000
noise20        success=1.000 wrong_object=0.000 wrong_color=0.000 wrong_shape=0.000 detected_all=1.000
noise40        success=0.998 wrong_object=0.002 wrong_color=0.000 wrong_shape=0.002 detected_all=1.000
occ12          success=0.980 wrong_object=0.020 wrong_color=0.002 wrong_shape=0.018 detected_all=0.996
noise20_occ12  success=0.972 wrong_object=0.028 wrong_color=0.000 wrong_shape=0.028 detected_all=1.000
```

Comparison against generic pixel policy:

```text
generic pixel_xy OOD clean: success=0.244, wrong_object=0.756
learned evidence + BTS OOD clean: success=1.000, wrong_object=0.000
learned evidence + BTS OOD noise/occlusion: success≈0.972-1.000
```

Interpretation:

- Learned perception plus structured belief preserves OOD binding performance under corruption.
- Failure under occlusion is now small and mostly shape-related.
- This is the strongest controlled image-route evidence so far.

Next fix:

```text
Add a multi-seed report script that runs the key controlled benchmark experiments and emits a compact table/JSON for paper-style reporting.
```


---

## 20. Spark LIBERO setup feasibility check（2026-06-09）

Checked spark host:

```text
host: spark-3994
python3: /usr/bin/python3, Python 3.12.3
GPU: NVIDIA GB10
storage: /dev/nvme0n1p2 3.7T total, 2.9T available
available env managers: no conda, no micromamba, no uv, no python3.8/3.9/3.10
Docker: available
existing container: bts_m1
```

Checked `bts_m1` container:

```text
Python 3.12.3
no micromamba/conda/uv/python3.8 visible
```

Decision:

- Do **not** install LIBERO into `bts_m1`; it risks damaging the working CALVIN/3D-DA environment and LIBERO wants older Python-style dependencies.
- Use a separate Docker image/container for LIBERO, or first install micromamba/conda on host.
- Given Docker is available and disk is ample, preferred route is a new isolated `bts_libero` Docker image.

Next concrete setup plan:

```text
1. Create Dockerfile/libero or scripted docker run.
2. Base image with Python 3.8/3.10-compatible MuJoCo/robosuite stack.
3. Install LIBERO in isolation.
4. Run LIBERO-Object minimal import + OffScreenRenderEnv smoke.
5. Only after native LIBERO works, separately test OpenVLA eval stack.
```

Open question:

```text
Use native LIBERO stack first, or use OpenVLA's LIBERO requirements first?
```

Current recommendation: native LIBERO first for diagnostics/BDDL inspection; OpenVLA stack second for VLA comparison.


---

## 21. LIBERO Docker build and render smoke passed（2026-06-09）

Build retry result:

- First build failed because `torch==2.1.2` is unavailable for ARM64 CUDA 11.8 index.
- Fixed to ARM64-compatible pins:

```text
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
```

Docker image built:

```text
image: bts_libero:latest
image id: 66dfb4bba6b5
size: 11.8GB
container: bts_libero
status: Up
```

Import smoke initially hit LIBERO's interactive config prompt. Runtime fix:

```text
/root/.libero/config.yaml
```

Preseeded paths:

```yaml
assets: /workspace/LIBERO/libero/libero/assets
bddl_files: /workspace/LIBERO/libero/libero/bddl_files
benchmark_root: /workspace/LIBERO/libero/libero
datasets: /workspace/LIBERO/libero/datasets
init_states: /workspace/LIBERO/libero/libero/init_files
```

Dockerfile was updated to create this config during build.

### LIBERO-Object metadata smoke

```text
suite: libero_object
n_tasks: 10
```

First tasks:

```text
0 pick_up_the_alphabet_soup_and_place_it_in_the_basket
1 pick_up_the_cream_cheese_and_place_it_in_the_basket
2 pick_up_the_salad_dressing_and_place_it_in_the_basket
```

Each task has fixed init states:

```text
init_states shape: (50, 110)
```

### OffScreenRenderEnv smoke

LIBERO render/env smoke passed in `bts_libero`:

```text
env = OffScreenRenderEnv(... camera_heights=128, camera_widths=128)
obs = env.reset()
obs, reward, done, info = env.step([0.0] * 7)
```

Observed image keys:

```text
agentview_image             (128, 128, 3) uint8
robot0_eye_in_hand_image    (128, 128, 3) uint8
```

State/diagnostic keys include object states:

```text
alphabet_soup_1_pos / quat / to_robot0_eef_*
basket_1_pos / quat / to_robot0_eef_*
cream_cheese_1_pos / ...
object-state                (98,)
robot0_proprio-state        (39,)
```

Step smoke:

```text
reward=0.0
done=False
info={}
```

Important caveat:

```text
torch.cuda.is_available() == False inside bts_libero
```

Likely due GB10 / CUDA runtime compatibility with CUDA 11.8 container. Native LIBERO CPU/EGL smoke still works. For OpenVLA/GPU, use a separate newer CUDA/torch stack or lab A6000.

Next fix:

```text
Build LIBERO diagnostic parser over task.language + BDDL + object-state keys.
```


---

## 22. LIBERO-Object diagnostic parser v0（2026-06-09）

Implemented:

```text
bts-poc/experiments/libero_object_diagnostics.py
```

Ran in `bts_libero`:

```bash
python /workspace/libero_object_diagnostics.py   --suite libero_object   --render-one   --out /workspace/bts/libero_object_diagnostics.json
```

Result:

```text
suite libero_object
n_tasks 10
```

Parsed task targets:

```text
0 alphabet_soup      -> basket | pick up the alphabet soup and place it in the basket
1 cream_cheese       -> basket | pick up the cream cheese and place it in the basket
2 salad_dressing     -> basket | pick up the salad dressing and place it in the basket
3 bbq_sauce          -> basket | pick up the bbq sauce and place it in the basket
4 ketchup            -> basket | pick up the ketchup and place it in the basket
5 tomato_sauce       -> basket | pick up the tomato sauce and place it in the basket
6 butter             -> basket | pick up the butter and place it in the basket
7 milk               -> basket | pick up the milk and place it in the basket
8 chocolate_pudding  -> basket | pick up the chocolate pudding and place it in the basket
9 orange_juice       -> basket | pick up the orange juice and place it in the basket
```

For every task:

```text
bddl_contains_target = True
bddl_contains_receptacle = True
init_states_shape = (50, 110)
```

Task-0 rendered obs exposes object diagnostic keys:

```text
alphabet_soup_1_pos
basket_1_pos
butter_1_pos
cream_cheese_1_pos
milk_1_pos
salad_dressing_1_pos
tomato_sauce_1_pos
object-state
robot0_eef_pos
...
```

Interpretation:

- LIBERO-Object is diagnostic-friendly for BTS: target object and receptacle can be parsed from language and verified in BDDL.
- Observation exposes per-object positions, enabling first-contact / nearest-object / target-distance diagnostics.
- LIBERO-Object is object-identity binding rather than color-shape attribute binding, but it is the correct first real benchmark.

Next fix:

```text
Build a small LIBERO-Object random/noop rollout logger that records target distance, nearest object to gripper, and success over init states.
```


---

## 23. LIBERO-Object rollout diagnostics v0（2026-06-09）

Implemented:

```text
bts-poc/experiments/libero_object_rollout_diagnostics.py
```

Ran in `bts_libero`:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python /workspace/libero_object_rollout_diagnostics.py   --tasks 3 --inits 2 --steps 10 --policy noop   --out /workspace/bts/libero_object_rollout_diag_noop.json

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python /workspace/libero_object_rollout_diagnostics.py   --tasks 3 --inits 2 --steps 10 --policy random   --out /workspace/bts/libero_object_rollout_diag_random.json
```

Both no-op and small random rollout diagnostics completed:

```text
n_rollouts: 6
success_count: 0
```

Example initial diagnostics:

```text
task 0 target=alphabet_soup
nearest0=cream_cheese_1, nearest_dist≈0.251
target_dist_to_eef≈0.326

task 1 target=cream_cheese
nearest0=milk_1, nearest_dist≈0.250
target_dist_to_eef≈0.345

task 2 target=salad_dressing
nearest0=tomato_sauce_1, nearest_dist≈0.247
target_dist_to_eef≈0.318
```

What the logger records per step:

```text
target_dist_to_eef
target_dist_to_receptacle
nearest_object to gripper
agentview_image shape
wrist image shape
success_seen
```

Interpretation:

- LIBERO-Object is now connected to BTS-style diagnostics, not just import/render smoke.
- The target object is often not the nearest object to the gripper initially, so wrong-object / nearest-distractor diagnostics are meaningful.
- This scaffold can evaluate a real policy by logging whether first-contact / nearest approach goes to the correct target object.

Next fix:

```text
Add a heuristic target-reaching policy using object-state positions to verify diagnostic success path and distance reduction before integrating learned policies.
```


---

## 24. LIBERO target-reaching diagnostic heuristic（2026-06-09）

Updated:

```text
bts-poc/experiments/libero_object_rollout_diagnostics.py
```

Added policy:

```text
target_reach
```

Policy logic:

```text
parse target object from language
find target object position in obs
find robot0_eef_pos
apply clipped proportional delta action toward target
```

This is diagnostic-only, not a task-solving policy. It tests whether the logger can detect behavior that intentionally approaches the correct target.

Command:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python /workspace/libero_object_rollout_diagnostics.py   --tasks 3 --inits 2 --steps 25 --policy target_reach   --out /workspace/bts/libero_object_rollout_diag_target_reach.json
```

Distance-drop comparison over 6 rollouts:

```text
noop drops         [0.0103, 0.0156, 0.0082, 0.0090, 0.0261, 0.0284]
noop mean_drop     0.0163
noop mean_final    0.3194

random drops       [0.0146, 0.0141, 0.0073, 0.0075, 0.0258, 0.0268]
random mean_drop   0.0160
random mean_final  0.3197

target_reach drops      [0.0465, 0.0509, 0.0479, 0.0492, 0.0659, 0.0678]
target_reach mean_drop  0.0547
target_reach mean_final 0.2810
```

Interpretation:

- The diagnostic metric is behavior-sensitive: target-reaching reduces target-to-EEF distance about 3.4× more than noop/random.
- This validates LIBERO-Object as a real-benchmark scaffold for measuring whether policies approach the correct target object.
- It still does not solve the full task; gripper/orientation/receptacle placement are not handled.

Next engineering target:

```text
Add first-approach / nearest-object-over-time summary and wrong-nearest-object metric for any policy trace.
```


---

## 25. LIBERO nearest-object metrics v0（2026-06-09）

Updated:

```text
bts-poc/experiments/libero_object_rollout_diagnostics.py
```

Added per-rollout summary:

```text
first_nearest_object
first_nearest_is_target
nearest_target_fraction
first_target_nearest_t
target_dist_initial
target_dist_final
target_dist_drop
```

Aggregate summary:

```text
mean_target_dist_drop
mean_nearest_target_fraction
first_nearest_target_rate
```

Ran three policies over 3 tasks × 2 init states × 25 steps:

```bash
for p in noop random target_reach; do
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python /workspace/libero_object_rollout_diagnostics.py     --tasks 3 --inits 2 --steps 25 --policy $p     --out /workspace/bts/libero_object_rollout_diag_${p}_v2.json
done
```

Results:

```text
noop:
  mean_target_dist_drop = 0.0163
  mean_nearest_target_fraction = 0.0
  first_nearest_target_rate = 0.0

random:
  mean_target_dist_drop = 0.0164
  mean_nearest_target_fraction = 0.0
  first_nearest_target_rate = 0.0

target_reach:
  mean_target_dist_drop = 0.0547
  mean_nearest_target_fraction = 0.0
  first_nearest_target_rate = 0.0
```

Interpretation:

- `target_dist_drop` is behavior-sensitive and clearly separates target_reach from noop/random.
- `nearest_target_fraction` remains 0 for all three because target_reach moves toward the parsed target but does not get close enough within 25 steps to make the target object nearest to the gripper.
- This is useful: nearest-object is a stricter diagnostic than target-distance reduction and probably corresponds more closely to first-contact / object-selection behavior.

Recommended diagnostic hierarchy for real policies:

```text
1. target_dist_drop          # weak but smooth signal
2. first_target_nearest_t    # stronger approach signal
3. nearest_target_fraction   # sustained target focus
4. first-contact object      # strongest wrong-object metric, needs contact extraction
5. success                   # full task metric
```

Next fix:

```text
Extend target_reach horizon/action scale or add target_nearest heuristic to validate the stricter nearest-object metric.
```


---

## 26. LIBERO strict nearest-object metric validated（2026-06-09）

Updated:

```text
bts-poc/experiments/libero_object_rollout_diagnostics.py
```

Added policy:

```text
target_reach_fast
```

Difference from `target_reach`:

```text
gain: 2.0 -> 5.0
clip: 0.08 -> 0.20
steps: tested with 60
```

Command:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python /workspace/libero_object_rollout_diagnostics.py   --tasks 3 --inits 2 --steps 60 --policy target_reach_fast   --out /workspace/bts/libero_object_rollout_diag_target_reach_fast.json
```

Result over 6 rollouts:

```text
success_count = 0
mean_target_dist_drop = 0.2437
mean_nearest_target_fraction = 0.3722
first_nearest_target_rate = 0.0
```

Per-rollout nearest target fraction:

```text
alphabet_soup init0:     0.2667, drop 0.2278
alphabet_soup init1:     0.2167, drop 0.2331
cream_cheese init0:      0.3500, drop 0.2361
cream_cheese init1:      0.2833, drop 0.2467
salad_dressing init0:    0.5833, drop 0.2586
salad_dressing init1:    0.5333, drop 0.2599
```

Interpretation:

- The stricter nearest-object metric is now validated: when the policy approaches the target aggressively enough, the target becomes nearest for a substantial fraction of the trajectory.
- `first_nearest_target_rate` remains 0 because all tested rollouts start with a distractor closer than the target, which is expected and useful for wrong-object diagnostics.
- Full task success remains 0 because this heuristic still does not grasp/place.

Diagnostic hierarchy is now empirically validated:

```text
weak approach:      target_dist_drop
strong approach:    nearest_target_fraction
initial ambiguity:  first_nearest_target_rate
full task:          success
```

Next engineering target:

```text
Add a compact LIBERO diagnostics report comparing noop/random/target_reach/target_reach_fast.
```


---

## 27. LIBERO-Spatial diagnostic parser v0（2026-06-09）

Updated:

```text
bts-poc/experiments/libero_object_diagnostics.py
```

The parser now supports spatial language forms:

```text
pick up the {object} {relation phrase} and place it on/in the {receptacle}
```

and BDDL goal extraction:

```text
(:goal (And (On {goal_object_instance} {goal_receptacle_instance})))
```

Ran in `bts_libero`:

```bash
python /workspace/libero_object_diagnostics.py   --suite libero_spatial   --render-one   --out /workspace/bts/libero_spatial_diagnostics_v2.json
```

Result:

```text
suite libero_spatial
n_tasks 10
```

Parsed tasks:

```text
0 black_bowl -> plate | relation between the plate and the ramekin | goal akita_black_bowl_1 -> plate_1
1 black_bowl -> plate | relation next to the ramekin | goal akita_black_bowl_1 -> plate_1
2 black_bowl -> plate | relation from table center | goal akita_black_bowl_1 -> plate_1
3 black_bowl -> plate | relation on the cookie box | goal akita_black_bowl_1 -> plate_1
4 black_bowl -> plate | relation in the top drawer of the wooden cabinet | goal akita_black_bowl_1 -> plate_1
5 black_bowl -> plate | relation on the ramekin | goal akita_black_bowl_1 -> plate_1
6 black_bowl -> plate | relation next to the cookie box | goal akita_black_bowl_1 -> plate_1
7 black_bowl -> plate | relation on the stove | goal akita_black_bowl_1 -> plate_1
8 black_bowl -> plate | relation next to the plate | goal akita_black_bowl_1 -> plate_1
9 black_bowl -> plate | relation on the wooden cabinet | goal akita_black_bowl_1 -> plate_1
```

For all tasks:

```text
bddl_contains_target = True
bddl_contains_receptacle = True
goal object instance = akita_black_bowl_1
goal receptacle instance = plate_1
```

Rendered obs exposes both ambiguous bowl instances:

```text
akita_black_bowl_1_pos
akita_black_bowl_2_pos
plate_1_pos
glazed_rim_porcelain_ramekin_1_pos
cookies_1_pos
...
```

Interpretation:

- LIBERO-Spatial is more BTS-relevant than LIBERO-Object because the language disambiguates between two visually similar black bowls by relation/location.
- The target instance is available from BDDL goal, making wrong-bowl diagnostics feasible.
- Relation phrases are parseable from language and can be cross-checked against BDDL init regions.

This becomes the primary real-benchmark diagnostic target for relation/object binding.

Next fix:

```text
Extend rollout diagnostics from LIBERO-Object to LIBERO-Spatial using BDDL goal_object_instance as target key, then run target_reach_fast to verify wrong-bowl nearest metrics.
```


---

## 28. LIBERO-Spatial rollout diagnostics v0（2026-06-09）

Updated:

```text
bts-poc/experiments/libero_object_rollout_diagnostics.py
```

Changes:

- `--suite` argument added, supports `libero_object` and `libero_spatial`.
- BDDL goal object instance is parsed and preferred as target key.
- This disambiguates `akita_black_bowl_1` vs `akita_black_bowl_2` in LIBERO-Spatial.
- Fixed nearest-object target matching to handle exact instance names (`name == target`) as well as prefix names (`name.startswith(target + "_")`).
- Spatial language parser added to rollout logger.

Command:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python /workspace/libero_object_rollout_diagnostics.py   --suite libero_spatial   --tasks 3 --inits 2 --steps 60   --policy target_reach_fast   --out /workspace/bts/libero_spatial_rollout_diag_target_reach_fast_v2.json
```

Result over 6 rollouts:

```text
success_count = 0
mean_target_dist_drop = 0.1659
mean_nearest_target_fraction = 0.4583
first_nearest_target_rate = 0.3333
```

Per-task pattern:

```text
task 0 relation=between plate and ramekin:
  nearest_target_fraction = 0.35 / 0.40

task 1 relation=next to ramekin:
  nearest_target_fraction = 0.00 / 0.00

task 2 relation=from table center:
  first_nearest_target = True / True
  nearest_target_fraction = 1.00 / 1.00
```

Interpretation:

- LIBERO-Spatial rollout diagnostics now use BDDL target instances correctly.
- The metrics detect relation-dependent ambiguity: some spatial relations start with the goal bowl nearest, others have distractors closer.
- This is directly relevant to BTS: relation-conditioned object-instance binding can be evaluated by `nearest_target_fraction`, `first_nearest_target_rate`, and future first-contact metrics.

Important caveat:

```text
success remains 0 because target_reach_fast only approaches; it does not grasp/place.
```

Next fix:

```text
Add a relation-aware diagnostic report for LIBERO-Spatial and compare Object vs Spatial as BTS real-benchmark targets.
```


---

## 29. LIBERO first-contact diagnostics v0（2026-06-09）

Updated:

```text
bts-poc/experiments/libero_object_rollout_diagnostics.py
```

Added MuJoCo contact extraction:

```text
contact_object(env, object_names)
```

It scans `env.sim.data.contact` and maps robot/gripper collision geoms to object instances.

Robot/contact geoms observed:

```text
gripper0_hand_collision
gripper0_finger1_collision
gripper0_finger2_collision
robot0_link*_collision
```

Object geoms use instance prefixes, e.g.:

```text
akita_black_bowl_1_g22
akita_black_bowl_1_g29
```

Added per-step:

```text
contact_object
```

Added per-rollout:

```text
first_contact_object
first_contact_is_target
```

### Short rollout result

With `target_reach_fast`, 60 steps over 3 LIBERO-Spatial tasks × 2 init states produced no contacts. This is expected: the heuristic approaches but often does not physically touch within 60 steps.

### Long single rollout validation

Command:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python /workspace/libero_object_rollout_diagnostics.py   --suite libero_spatial --tasks 1 --inits 1 --steps 200   --policy target_reach_fast   --out /workspace/bts/libero_spatial_contact_long.json
```

Result for task 0 init 0:

```text
nearest_target_fraction = 0.805
first_target_nearest_t = 39
first_contact_object = akita_black_bowl_1
first_contact_is_target = True
target_dist_initial = 0.3335
target_dist_final = 0.0210
target_dist_drop = 0.3125
contact_count = 108
first contact around t=90
```

Example contact pair:

```text
akita_black_bowl_1_g22 <-> gripper0_finger1_collision
```

Interpretation:

- First-contact extraction works.
- It correctly identifies contact with the BDDL target instance, not just the generic class `black_bowl`.
- This completes the key BTS real-benchmark diagnostic chain:

```text
language relation
  -> BDDL target instance
  -> target-distance approach
  -> nearest-target fraction
  -> first-contact object
  -> wrong-object / wrong-instance metric
```

This is a major milestone for the image/VLA route.


---

## 30. LIBERO rollout summarizer v0（2026-06-09）

Implemented:

```text
bts-poc/experiments/summarize_libero_rollouts.py
```

Purpose: compare rollout diagnostic JSONs across policies/suites.

Command run in `bts_libero`:

```bash
python /workspace/summarize_libero_rollouts.py   /workspace/bts/libero_object_rollout_diag_noop_v2.json   /workspace/bts/libero_object_rollout_diag_random_v2.json   /workspace/bts/libero_object_rollout_diag_target_reach_v2.json   /workspace/bts/libero_object_rollout_diag_target_reach_fast.json   /workspace/bts/libero_spatial_rollout_diag_target_reach_fast_v2.json   /workspace/bts/libero_spatial_contact_long.json   --out /workspace/bts/libero_rollout_summary.json
```

Summary table:

```text
file                                                suite           policy             n  success  drop    nearest_frac  first_nearest  contacts  contact_target  final_dist
libero_object_rollout_diag_noop_v2.json            None            noop               6  0        0.0163  0.0000        0.0000         0         NA              0.3194
libero_object_rollout_diag_random_v2.json          None            random             6  0        0.0164  0.0000        0.0000         0         NA              0.3193
libero_object_rollout_diag_target_reach_v2.json    None            target_reach       6  0        0.0547  0.0000        0.0000         0         NA              0.2810
libero_object_rollout_diag_target_reach_fast.json  None            target_reach_fast  6  0        0.2437  0.3722        0.0000         0         NA              0.0920
libero_spatial_rollout_diag_target_reach_fast_v2   libero_spatial  target_reach_fast  6  0        0.1659  0.4583        0.3333         0         NA              0.1523
libero_spatial_contact_long.json                   libero_spatial  target_reach_fast  1  0        0.3125  0.8050        0.0000         1         1.0000          0.0210
```

Interpretation:

- Summarizer now gives one compact table for policy diagnostics.
- Long LIBERO-Spatial run validates first-contact target correctness (`contact_target=1.0`).
- This tool is ready to wrap future learned/OpenVLA policies.


---

## 31. External LIBERO policy adapter interface（2026-06-09）

Updated:

```text
bts-poc/experiments/libero_object_rollout_diagnostics.py
```

Added:

```text
--policy external
--policy-adapter module:function
```

Adapter signature:

```python
def policy(obs: dict, context: dict) -> list[float]:
    return action_7d
```

Context contains:

```text
suite
task_id
init_id
language
target_key
receptacle_key
target_object
receptacle
relation
goal_target_instance
goal_receptacle_instance
step
trace_so_far
```

Added smoke adapters:

```text
bts-poc/experiments/libero_policy_adapters.py
  zero_policy
  target_reach_policy
```

Smoke command in `bts_libero`:

```bash
cd /workspace
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python /workspace/libero_object_rollout_diagnostics.py   --suite libero_spatial   --tasks 1 --inits 1 --steps 20   --policy external   --policy-adapter libero_policy_adapters:target_reach_policy   --out /workspace/bts/libero_external_adapter_smoke.json
```

Result:

```text
n_rollouts = 1
success_count = 0
mean_drop = 0.0361
```

Interpretation:

- External adapter path works.
- Future OpenVLA/BTS policy wrappers can plug into diagnostics without editing logger internals.
- The diagnostic logger is now policy-agnostic enough for the next phase.


---

## 32. OpenVLA adapter skeleton（2026-06-09）

Implemented:

```text
bts-poc/experiments/openvla_policy_adapter.py
```

Purpose: define the integration contract between future OpenVLA runtime and the existing LIBERO diagnostics logger.

Adapter entrypoint:

```python
def openvla_policy(obs: dict, context: dict) -> list[float]
```

Expected diagnostic invocation:

```bash
python libero_object_rollout_diagnostics.py   --suite libero_spatial   --policy external   --policy-adapter openvla_policy_adapter:openvla_policy
```

The skeleton intentionally does not import OpenVLA at module import time because `bts_libero` is not an OpenVLA GPU environment. It provides:

```text
OpenVLAAdapterConfig
configure(...)
make_prompt(language)
image conversion to PIL
openvla_policy skeleton
zero_policy smoke adapter
```

Syntax check passed:

```bash
python -m py_compile bts-poc/experiments/openvla_policy_adapter.py
```

Next actual OpenVLA work should happen in a GPU-capable OpenVLA environment, likely lab A6000 or a separate CUDA stack, not spark `bts_libero`.


---

## 33. Lab A6000 OpenVLA feasibility check（2026-06-09）

Checked `ssh lab`:

```text
host: USCC
GPU: NVIDIA RTX A6000, 46068 MiB
Driver: 596.59
nvidia-smi path: /usr/lib/wsl/lib/nvidia-smi
system python: /usr/bin/python3, Python 3.12.3
no conda/micromamba/uv/pipx/docker on PATH
HDD free: 2.6T available at /mnt/wsl/HDD
root fs free: 32G only
```

Existing env:

```text
~/HDD/envs/bts_a6000
Python 3.12.13 conda-forge
Torch 2.7.0+cu128
CUDA available: True
GPU: NVIDIA RTX A6000
```

Implications:

- Lab A6000 is the right machine for OpenVLA GPU eval, but not with the current `bts_a6000` env if OpenVLA strictly needs Python 3.10 / torch 2.2 / transformers 4.40 / flash-attn pins.
- No env manager is on PATH, despite `bts_a6000` being conda-style. Need either:
  1. install micromamba under `~/HDD/tools`, or
  2. use Python 3.12 existing env and test whether OpenVLA dependencies tolerate it, or
  3. create a container another way (Docker absent, so not preferred).
- Use `~/HDD` for all OpenVLA files; root fs is nearly full.

Recommended next action:

```text
Install micromamba locally under ~/HDD/tools on lab, create isolated openvla env with Python 3.10, then clone OpenVLA and run import/checkpoint metadata smoke.
```
