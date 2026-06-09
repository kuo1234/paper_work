# BTS/OpenVLA Intervention Report — LIBERO-Object Binding Failures

**Date:** 2026-06-09
**Env:** spark-only GB10, `~/openvla-spark/.venv`
**Base VLA:** `openvla/openvla-7b-finetuned-libero-object`
**Suite:** LIBERO-Object

This report focuses only on the real-benchmark BTS/OpenVLA intervention evidence after the
OpenVLA baseline established a stable object-identity mis-binding rate.

---

## 1. Baseline failure set

Large baseline sweep:

```text
10 tasks x 3 inits = 30 rollouts
success = 23/30 = 0.767
wrong_object_type = 3/30 = 0.10
```

Stable wrong-object cases:

```text
1:0  cream_cheese        -> tomato_sauce
6:2  butter              -> basket
8:0  chocolate_pudding   -> orange_juice
```

Targeted rerun (`--pairs 1:0,6:2,8:0`) reproduced all three:

```text
success = 0/3
first_wrong_type = 3/3
any_wrong_type = 3/3
```

This is the fixed failure set for fast intervention testing.

---

## 2. Intervention variants tested

### 2.1 Always target-gate oracle

Adapter: `openvla_target_gate_policy`

Mechanism: preserve OpenVLA rotation + gripper, replace translation with a proportional controller
toward the BDDL target for the entire rollout.

Known failures:

```text
success = 0/3
first_wrong_type = 0/3
any_wrong_type = 1/3
```

Interpretation: fixes first wrong-object contact, but continuous pulling toward target harms full
task completion and can still brush a wrong object later.

---

### 2.2 Target-gate until contact

Adapter: `openvla_target_gate_until_contact_policy`

Mechanism: gate translation toward target until the target is contacted, then hand control back to
pure OpenVLA.

Known failures:

```text
success = 2/3
first_wrong_type = 0/3
any_wrong_type = 0/3
```

Per-case:

```text
cream_cheese       success=0, first_contact=cream_cheese_1
butter             success=1, first_contact=butter_1
chocolate_pudding  success=1, first_contact=chocolate_pudding_1
```

This is the strongest positive result: a structured target-binding acquisition gate fixes the
wrong-object contact and recovers task success on 2/3 known failures.

---

### 2.3 Broader 10x1 check

Compare baseline init-0 slice vs target-gate-until-contact:

| Metric | OpenVLA baseline init0 | target-gate-until-contact init0 |
|---|---:|---:|
| success | 7/10 | 3/10 |
| any_wrong_type | 2/10 | **0/10** |
| any_target_contact | 8/10 | **10/10** |
| mean_nearest_target_fraction | 0.758 | 0.712 |

Interpretation: the acquisition gate is excellent for binding metrics but too blunt as an always-on
wrapper. It fixes wrong contacts but regresses otherwise-good rollouts.

---

### 2.4 Low-gain target gate

Adapter: target-gate-until-contact with:

```text
BTS_TARGET_GATE_GAIN=2.0
BTS_TARGET_GATE_CLIP=0.08
```

Known failures:

```text
success = 0/3
first_wrong_type = 0/3
any_wrong_type = 0/3
```

Interpretation: weaker translation still fixes binding, but loses the 2/3 success recovery. Success
requires decisive acquisition + good handoff, not just gentler motion.

---

### 2.5 Directional endpoint gate

Adapter: `openvla_directional_bts_gate_policy`

Mechanism: before target contact, compute `endpoint = EEF + OpenVLA_action[:3]`; gate only if the
endpoint is nearer a non-target object than the BDDL target.

Known failures:

```text
success = 1/3
first_wrong_type = 0/3
any_wrong_type = 0/3
```

Broader init-0:

```text
success = 3/10
first_wrong_type = 0/10
any_wrong_type = 1/10
target_contact = 10/10
```

Interpretation: geometry-only selectivity is not enough. It removes first wrong contacts but still
regresses success.

---

## 3. Why geometry-only selectivity fails

Action/eef/object traces were added to diagnose this. Mixed pure-OpenVLA traces show endpoint-nearest
wrong-object rates are high for both bad and good rollouts:

```text
cream_cheese wrong case:        endpoint_wrong_rate = 0.625
alphabet_soup good case:        endpoint_wrong_rate = 0.579
salad_dressing good case:       endpoint_wrong_rate = 0.863
orange_juice good case:         endpoint_wrong_rate = 0.795
```

Therefore raw endpoint-nearest geometry cannot reliably decide when to intervene. It produces too
many false positives.

---

## 4. Current conclusion

The real-benchmark BTS/OpenVLA story is now sharper:

```text
OpenVLA has stable object-identity mis-binding on LIBERO-Object (3/30 = 0.10).
A structured target-binding acquisition gate can eliminate the wrong contacts.
If the gate is used until target contact, it recovers success on 2/3 known failures.
But always-on/geometric gating regresses otherwise-good rollouts.
```

So the next method must be:

```text
evidence/belief-selective target acquisition
```

not:

```text
always-on oracle target control
endpoint-nearest geometry heuristic
```

This fits the BTS claim: belief should decide **when** to override a VLA's implicit binding, not
blindly replace the VLA policy.

---

## 5. Next concrete experiment

Build a small visual/object-evidence module for LIBERO-Object:

```text
input: agentview image + language target object name
output: target evidence / confidence over visible object candidates
trigger: activate target-gate only when confidence is low or top evidence disagrees with VLA motion/contact trend
metric: wrong_object_type first/any + LIBERO success on fixed failures and 10x1/10x3 slices
```

First dataset substrate collected:

```text
script: bts-poc/experiments/collect_libero_object_evidence_data.py
remote artifact: ~/bts-poc/experiments/runs/libero_object_evidence_v0
metadata: runs/libero_object_evidence_v0/metadata.jsonl
n = 50 frames (libero_object 10 tasks x 5 inits, 256px, OpenVLA-rotated agentview)
labels: language, target object, object_positions, target_pos, receptacle_pos
```

Next milestone should be a diagnostic evidence oracle or detector, not full policy learning.
