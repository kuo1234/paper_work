# OpenVLA LIBERO-Spatial Baseline Report

**Date:** 2026-06-09
**Env:** spark GB10 (aarch64), `~/openvla-spark/.venv`, torch 2.12.0+cu130, transformers 4.40.1
**Checkpoint:** `openvla/openvla-7b-finetuned-libero-spatial`
**Eval harness:** existing `libero_object_rollout_diagnostics.py` external-policy path
**Adapter:** `openvla_policy_adapter:openvla_policy` (faithful preprocessing — see below)

This is the first real VLA baseline on the image route. It runs entirely on spark
(spark-only decision: A6000 reserved for shared use), proving the GB10 stack is faithful
to OpenVLA's official LIBERO eval.

---

## 1. Setup faithfulness

Critical preprocessing replicated from OpenVLA `experiments/robot/libero/run_libero_eval.py`
(without these, OpenVLA drifts and scores 0 — verified):

```text
agentview_image rotated 180 deg (img[::-1, ::-1]) before resize to 224 (lanczos)
center_crop True (finetuned checkpoints trained with image augmentation), 90% crop+resize
gripper: normalize [0,1] -> [-1,+1] (binarized), then invert sign (env: -1=open, +1=close)
10 dummy no-op warmup steps to settle objects + env.seed(0)
camera resolution 256
```

Sanity that the fix matters (single rollout, task 0 init 0):

```text
before preprocessing fix: success 0, target_dist_drop -0.298 (EEF moves AWAY)
after  preprocessing fix: success 1, target_dist_drop +0.317 (EEF moves TOWARD target)
```

---

## 2. Baseline result (5 tasks x 2 inits = 10 rollouts, 280 steps each)

Artifact: `runs/openvla_spatial_sweep5x2.json`

| Metric | Value |
|---|---|
| **Success rate** | **9 / 10 = 0.90** |
| any_target_contact_rate | 1.00 |
| **any_distractor_instance_contact_rate** | **0.00** |
| first_contact_distractor_rate | 0.00 |
| mean_nearest_target_fraction | 0.753 |
| mean_target_dist_drop | +0.299 |

Per-rollout:

```text
t0i0 succ=1 distractor=0 near_frac=0.54   t0i1 succ=0 distractor=0 near_frac=0.07
t1i0 succ=1 distractor=0 near_frac=0.75   t1i1 succ=1 distractor=0 near_frac=0.77
t2i0 succ=1 distractor=0 near_frac=0.84   t2i1 succ=1 distractor=0 near_frac=0.92
t3i0 succ=1 distractor=0 near_frac=0.91   t3i1 succ=1 distractor=0 near_frac=0.76
t4i0 succ=1 distractor=0 near_frac=0.99   t4i1 succ=1 distractor=0 near_frac=0.98
```

The single failure (t0i1) still contacted the correct bowl (`any_target_contact=1`,
`distractor=0`) but with low nearest-fraction (0.07) — a grasp/manipulation failure, not
a binding/instance failure.

The 0.90 success rate is consistent with OpenVLA's published ~0.85 on libero_spatial,
which independently corroborates that the spark-only pipeline is faithful.

---

## 3. Key implication for BTS (this reshapes the narrative)

**On vanilla LIBERO-Spatial, OpenVLA does NOT mis-bind same-class instances.**
Distractor-instance contact rate is 0.00 across all 10 rollouts: when the language says
"the black bowl between the plate and the ramekin", OpenVLA reliably grasps the BDDL
target instance (`akita_black_bowl_1`), not the distractor (`akita_black_bowl_2`).

Consequence: the naive BTS pitch — "generic VLAs mis-bind object instances on standard
benchmarks, structured belief fixes it" — is **not** supported on stock LIBERO-Spatial.
A 7B VLA finetuned on this exact suite has already learned the relation-conditioned binding.

This is an honest, load-bearing negative result. It does NOT invalidate the controlled-
benchmark finding (structured belief beats generic belief OOD); it bounds where the BTS
gap is demonstrable. The binding-failure gap must be elicited under harder conditions the
finetuned VLA did not see in-distribution:

```text
1. OOD spatial relations / rephrased instructions not in the libero_spatial training mix
2. Added same-class distractors or perturbed init states (more bowls, ambiguous placement)
3. Attribute binding (color/shape) rather than spatial relation — closer to the controlled
   benchmark where the gap was large (generic 0.080 vs structured 1.000 OOD)
4. Zero-shot / non-finetuned VLA, where in-distribution binding hasn't been memorized
```

Recommended next experiment: probe OpenVLA under (1)/(2) to find a regime with nonzero
distractor-contact rate. If none exists on LIBERO, pivot the real-benchmark binding claim
to a controlled-distractor variant or to the attribute-binding axis where BTS already wins.

---

## 4. Reproduce

See `REPRODUCIBILITY.md` §3b for the full env build and eval command.
