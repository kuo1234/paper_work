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

---

## 5. Probe: relation-stripped (ambiguous) instructions

Artifact: `runs/openvla_spatial_ambiguous5x2.json`
Command: baseline sweep + `--language-override "pick up the black bowl and place it on the plate"`
(the disambiguating spatial relation is removed; scoring still uses the BDDL target instance).

| Metric | With relation (baseline) | Relation stripped |
|---|---|---|
| Success rate | 0.90 | **0.20** |
| any_target_contact_rate | 1.00 | **0.70** |
| any_distractor_instance_contact_rate | 0.00 | **0.00** |
| mean_nearest_target_fraction | 0.753 | 0.294 |
| mean_target_dist_drop | +0.299 | +0.163 |

Interpretation (load-bearing nuance):

```text
OpenVLA clearly USES the relation phrase: removing it collapses success 0.90 -> 0.20.
BUT it still does NOT grasp the distractor bowl (distractor rate stays 0.00).
The failure mode is degraded / aborted grasp (target-contact 1.0 -> 0.7,
nearest-fraction 0.75 -> 0.29), i.e. ambiguity -> hesitation, NOT confident mis-binding.
```

So neither vanilla nor relation-stripped LIBERO-Spatial produces *confident* same-class
mis-binding for OpenVLA. The "resolve which instance" BTS story is therefore not directly
demonstrated on this VLA+suite by either probe. To find a confident-mis-binding regime, the
remaining levers are:

```text
- add a SECOND same-class bowl placed where a shortcut policy would prefer it (BDDL edit)
- attribute binding (color/shape) instead of spatial relation (controlled benchmark axis)
- a non-finetuned / zero-shot VLA that hasn't memorized in-distribution binding
```

---

## 6. LIBERO-Object: stabilized cross-object mis-binding signal

Artifacts:

```text
runs/openvla_object5x2.json   (initial 10 rollouts)
runs/openvla_object10x3.json  (larger 30-rollout sweep; headline below)
```

Checkpoint: `openvla/openvla-7b-finetuned-libero-object`.
Suite has 10 DISTINCT object types -> basket, so the binding axis is object identity, not
same-class instance (there are no same-class pairs; distractor-instance rate is trivially 0).
The relevant signal is **wrong-object-TYPE contact** (target stem != contacted stem).

### 6.1 Larger sweep result (10 tasks x 3 inits = 30 rollouts)

| Metric | Value |
|---|---|
| Success rate | 23/30 = 0.767 |
| any_target_contact_rate | 0.90 |
| mean_nearest_target_fraction | 0.783 |
| mean_target_dist_drop | +0.233 |
| **wrong_object_type_first_rate** | **0.10** |
| **wrong_object_type_any_rate** | **0.10** |

Wrong-object contact cases:

```text
t1i0: target=cream_cheese        first_contact=tomato_sauce
t6i2: target=butter              first_contact=basket
t8i0: target=chocolate_pudding   first_contact=orange_juice
```

The initial 5x2 sweep already showed 1/10 wrong-object type (`cream_cheese -> tomato_sauce`).
The larger 10x3 sweep stabilizes the rate at **3/30 = 0.10**.

This is the **first stable nonzero mis-binding signal** in the OpenVLA study: unlike spatial
(0.00 either way), the object-IDENTITY axis does produce wrong-object contacts. It points the
BTS real-benchmark binding claim toward **object identity / attribute binding** rather than
spatial-relation disambiguation — consistent with the controlled benchmark, where the large
BTS gap was on object-attribute binding (generic 0.080 vs structured 1.000 OOD).

Caveat: wrong-object rate is not huge (10%), but it is real and directly relevant. Next step is
not to keep broadening vanilla LIBERO blindly; instead build a BTS/OpenVLA intervention for
these object-identity failures or design a controlled-distractor/attribute variant that makes
the error denser.

Fast targeted rerun command for the known failures (new `--pairs` option):

```bash
OPENVLA_CHECKPOINT=openvla/openvla-7b-finetuned-libero-object \
MUJOCO_GL=egl HF_HOME=~/bts/hf_cache PYTHONPATH=~/bts-poc/experiments \
~/openvla-spark/.venv/bin/python libero_object_rollout_diagnostics.py \
  --suite libero_object --pairs 1:0,6:2,8:0 \
  --steps 280 --warmup-steps 10 --camera-size 256 \
  --policy external --policy-adapter openvla_policy_adapter:openvla_policy \
  --out runs/openvla_object_known_failures.json
```

Use this for fast BTS/OpenVLA intervention tests instead of rerunning a full 30-rollout sweep.

### 6.2 Targeted rerun confirms the known failures are stable

Artifact: `runs/openvla_object_known_failures_rerun.json`

Reran only the three known wrong-object cases with `--pairs 1:0,6:2,8:0`.
All three reproduced as wrong-object contacts:

```text
n_rollouts 3, success_count 0, wrong_type_contact_rate 1.0

1:0 cream_cheese       first_contact tomato_sauce_1      wrong_type=True
6:2 butter             first_contact basket_1            wrong_type=True
8:0 chocolate_pudding  first_contact orange_juice_1      wrong_type=True
```

This gives a cheap fixed failure set for the next intervention step: a BTS/OpenVLA module only
needs to be tested first on these 3 rollouts before running the full 30-rollout sweep.

### 6.3 Oracle target-gate intervention smoke

Artifact: `runs/openvla_object_known_failures_target_gate.json`
Adapter: `openvla_policy_adapter:openvla_target_gate_policy`

This is a diagnostic upper-bound, not the final learned BTS method. It keeps OpenVLA rotation +
gripper, but replaces translation with a proportional controller toward the BDDL target instance
(`context['target_key']`). It asks: if structured binding selects the correct target, can we
eliminate the known wrong-first-contact failures?

Result on the fixed 3 failure cases:

```text
first_contact_wrong_type: 3/3 -> 0/3  (fixed)
first contacts:
  1:0 cream_cheese       tomato_sauce_1     -> cream_cheese_1
  6:2 butter             basket_1           -> butter_1
  8:0 chocolate_pudding  orange_juice_1     -> chocolate_pudding_1
any_wrong_type_contact:  3/3 -> 1/3  (cream_cheese later brushed tomato_sauce)
success_count: 0/3       (binding fixed; full task completion not fixed)
```

Interpretation: structured target binding is sufficient to fix the **first wrong-object contact**
on the known failures, but it is not by itself sufficient for task success. The next BTS/OpenVLA
prototype should therefore separate two claims:

```text
binding metric: first_contact_wrong_type / any_wrong_type_contact
full manipulation metric: LIBERO success
```

BTS should first reduce wrong-contact metrics, then later address placement/success.


