---
type: experiment-report
aliases:
  - "BTS controlled image binding v0 report"
  - "Synthetic image-binding benchmark results"
tags:
  - BTS
  - image-binding
  - controlled-benchmark
  - experiment-report
summary: "Controlled image-binding v0 results: generic image policies exploit spurious shortcuts and fail OOD; structured BTS binding with learned object evidence succeeds under OOD and corruption."
---

# Controlled Image-Binding Benchmark v0 Report（2026-06-08）

## 1. Purpose

After pivoting BTS from 3D-Diffuser-Actor/CALVIN reproduction to image-language manipulation, we need a fast controlled benchmark that isolates the core failure mode:

> A policy may bind the right attribute to the wrong object when the training distribution supports spurious shortcuts.

The benchmark tests whether structured belief over object-attribute bindings improves OOD robustness compared with generic image/language policies and generic belief heads.

---

## 2. Benchmark

Implementation:

```text
bts-poc/envs/image_binding.py
bts-poc/experiments/image_binding_v0.py
```

Scene:

```text
96x96 RGB
4 colored objects
attributes: color × shape
instruction: "pick the {color} {shape}"
action abstraction: select target object
```

OOD held-out target pairs:

```text
red triangle
blue square
green circle
```

Train/ID includes a color-location shortcut. OOD breaks the shortcut and tests held-out color-shape recombination.

Metrics:

```text
success
wrong_object
wrong_color
wrong_shape
```

---

## 3. Main results

### 3.1 Shortcut and oracle sanity

Command:

```bash
PYTHONPATH=bts-poc python bts-poc/experiments/image_binding_v0.py --n 1000
```

| Policy | Train success | ID success | OOD success | OOD wrong-object |
|---|---:|---:|---:|---:|
| location prior | 0.989 | 0.982 | 0.256 | 0.744 |
| oracle language binding | 1.000 | 1.000 | 1.000 | 0.000 |

Interpretation: the benchmark is solvable, but shortcut policies collapse on OOD.

### 3.2 Trainable policies

Implementation:

```text
bts-poc/experiments/train_image_binding_v0.py
bts-poc/experiments/train_image_pixels_v0.py
```

| Policy | Description | Train success | ID success | OOD success | OOD wrong-object |
|---|---|---:|---:|---:|---:|
| shortcut | instruction-only xy regressor | 0.986 | 0.988 | 0.258 | 0.742 |
| generic_belief | generic MLP over instruction-object features | 1.000 | 1.000 | 0.080 | 0.920 |
| pixel_xy | CNN image + instruction -> xy | 0.983 | 0.989 | 0.244 | 0.756 |
| bts_structured | decomposed color/shape structured belief | 1.000 | 1.000 | 1.000 | 0.000 |

Key finding:

> A generic belief head is not enough. It memorizes train-time color-shape compositions and fails worse than the shortcut baseline. The gain comes from explicit object-attribute factorization.

### 3.3 Image-grounded structured binding

Implementation:

```text
bts-poc/experiments/image_binding_detector_v0.py
bts-poc/experiments/learned_bts_policy_v0.py
```

Deterministic image-derived candidates:

| Setting | OOD success | detected_all |
|---|---:|---:|
| clean | 1.000 | 1.000 |
| noise20 | 1.000 | 1.000 |
| noise40 | 0.856 | 1.000 |
| occ12 | 0.954 | 0.998 |
| noise20+occ12 | 0.959 | 0.999 |

Learned patch evidence + structured BTS:

| Setting | OOD success | OOD wrong-object | detected_all |
|---|---:|---:|---:|
| clean | 1.000 | 0.000 | 1.000 |
| noise20 | 1.000 | 0.000 | 1.000 |
| noise40 | 0.998 | 0.002 | 1.000 |
| occ12 | 0.980 | 0.020 | 0.996 |
| noise20+occ12 | 0.972 | 0.028 | 1.000 |

Interpretation:

- BTS can be grounded in image-derived candidates.
- Learned object evidence remains robust under noise/occlusion.
- Remaining failures are small and mostly shape/occlusion-related.

---

## 4. Scientific takeaway

The controlled benchmark now supports the core BTS image-route claim:

```text
Generic image-language policies can exploit spurious shortcuts and fail OOD binding.
Generic belief heads can also fail by memorizing atomic seen compositions.
Structured belief over factorized object attributes restores OOD compositional binding.
```

Most important sentence for paper framing:

> The benefit is not from adding capacity or an unconstrained belief head; it comes from enforcing a structured object-attribute factorization in the belief state.

---

## 5. Current limitations

1. The benchmark is synthetic and still simple.
2. Object locations are extracted by deterministic color connected components.
3. Action is abstract object selection, not robot control.
4. Language is templated, not free-form.
5. Learned evidence uses cropped object patches, not end-to-end detection.

---

## 6. Next steps

1. Add harder visual variation:
   - random backgrounds
   - object scale variation
   - partial overlap
   - distractor colors closer in RGB
2. Replace deterministic connected components with learned/soft candidate proposals.
3. Add temporal belief update across multiple observations.
4. Port diagnostics to LIBERO-Object via BDDL/task-language parsing.
5. Compare against OpenVLA on LIBERO-Object after small eval smoke.
