---
type: experiment-report
aliases:
  - "BTS LIBERO Spatial diagnostics report"
  - "LIBERO Object vs Spatial benchmark comparison"
tags:
  - BTS
  - LIBERO
  - spatial-binding
  - diagnostics
summary: "LIBERO-Spatial diagnostics: relation-conditioned black-bowl instance binding is parseable via language + BDDL goal instance; rollout metrics detect target-instance approach. LIBERO-Spatial is the better BTS real-benchmark target than LIBERO-Object."
---

# LIBERO-Spatial Diagnostics Report（2026-06-09）

## 1. Purpose

Evaluate whether **LIBERO-Spatial** is a better real-benchmark target than LIBERO-Object for BTS.

BTS needs tasks where the policy must bind language to the correct object instance under ambiguity. LIBERO-Object tests object identity; LIBERO-Spatial tests relation-conditioned instance selection.

---

## 2. Parser status

Script:

```text
bts-poc/experiments/libero_object_diagnostics.py
```

Supported now:

```text
LIBERO-Object:
  pick up the {object} and place it in the {receptacle}

LIBERO-Spatial:
  pick up the {object} {relation phrase} and place it on/in the {receptacle}

BDDL goal:
  (:goal (And (On {goal_object_instance} {goal_receptacle_instance})))
```

LIBERO-Spatial command:

```bash
python /workspace/libero_object_diagnostics.py \
  --suite libero_spatial \
  --render-one \
  --out /workspace/bts/libero_spatial_diagnostics_v2.json
```

---

## 3. LIBERO-Spatial parsed tasks

All 10 tasks parse successfully.

| task | language target | relation | receptacle | BDDL target instance | BDDL receptacle |
|---:|---|---|---|---|---|
| 0 | black_bowl | between the plate and the ramekin | plate | akita_black_bowl_1 | plate_1 |
| 1 | black_bowl | next to the ramekin | plate | akita_black_bowl_1 | plate_1 |
| 2 | black_bowl | from table center | plate | akita_black_bowl_1 | plate_1 |
| 3 | black_bowl | on the cookie box | plate | akita_black_bowl_1 | plate_1 |
| 4 | black_bowl | in the top drawer of the wooden cabinet | plate | akita_black_bowl_1 | plate_1 |
| 5 | black_bowl | on the ramekin | plate | akita_black_bowl_1 | plate_1 |
| 6 | black_bowl | next to the cookie box | plate | akita_black_bowl_1 | plate_1 |
| 7 | black_bowl | on the stove | plate | akita_black_bowl_1 | plate_1 |
| 8 | black_bowl | next to the plate | plate | akita_black_bowl_1 | plate_1 |
| 9 | black_bowl | on the wooden cabinet | plate | akita_black_bowl_1 | plate_1 |

Important observation:

```text
language says "black bowl"
scene contains akita_black_bowl_1 and akita_black_bowl_2
BDDL goal identifies akita_black_bowl_1 as target
```

This is exactly relation-conditioned object-instance binding.

---

## 4. Render/state keys

Rendered task 0 exposes both candidate bowls:

```text
akita_black_bowl_1_pos
akita_black_bowl_2_pos
plate_1_pos
glazed_rim_porcelain_ramekin_1_pos
cookies_1_pos
object-state
robot0_eef_pos
```

So diagnostics can measure:

```text
correct bowl vs wrong bowl
nearest bowl to gripper
target-instance distance to gripper
target-instance distance to receptacle
relation-dependent ambiguity
```

---

## 5. Rollout diagnostics

Script:

```text
bts-poc/experiments/libero_object_rollout_diagnostics.py
```

LIBERO-Spatial command:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python /workspace/libero_object_rollout_diagnostics.py \
  --suite libero_spatial \
  --tasks 3 --inits 2 --steps 60 \
  --policy target_reach_fast \
  --out /workspace/bts/libero_spatial_rollout_diag_target_reach_fast_v2.json
```

Result over 6 rollouts:

| metric | value |
|---|---:|
| success_count | 0 / 6 |
| mean_target_dist_drop | 0.1659 |
| mean_nearest_target_fraction | 0.4583 |
| first_nearest_target_rate | 0.3333 |

Per-relation pattern:

| task | relation | first nearest target? | nearest target fraction | target dist drop |
|---:|---|---:|---:|---:|
| 0 | between plate and ramekin | 0/2 | 0.35 / 0.40 | 0.200 / 0.202 |
| 1 | next to ramekin | 0/2 | 0.00 / 0.00 | 0.172 / 0.169 |
| 2 | from table center | 2/2 | 1.00 / 1.00 | 0.129 / 0.124 |

Interpretation:

- The rollout logger correctly uses BDDL target instance (`akita_black_bowl_1`) instead of ambiguous language target (`black_bowl`).
- Relation condition changes initial ambiguity and nearest-target behavior.
- `target_reach_fast` moves toward the correct target instance enough to activate strict nearest-target metrics in many cases.
- Task success remains 0 because reach-only does not grasp/place.

---

## 6. LIBERO-Object vs LIBERO-Spatial for BTS

| Criterion | LIBERO-Object | LIBERO-Spatial | Better |
|---|---|---|---|
| Easy parser | yes | yes after BDDL goal parsing | tie |
| Image/state availability | yes | yes | tie |
| Target object ambiguity | moderate | high | Spatial |
| Multiple similar target-class objects | no / less central | yes: two black bowls | Spatial |
| Relation-conditioned binding | weak | strong | Spatial |
| Wrong-object diagnostics | object identity | instance + relation | Spatial |
| First smoke simplicity | easier | slightly harder | Object |
| BTS novelty alignment | medium | high | Spatial |

Conclusion:

```text
LIBERO-Object is the right first smoke.
LIBERO-Spatial is the better real benchmark for BTS claims.
```

---

## 7. Recommended benchmark role

Use:

```text
LIBERO-Object:
  environment smoke
  OpenVLA checkpoint smoke
  basic object-target diagnostics

LIBERO-Spatial:
  primary BTS real-benchmark diagnostic
  relation-conditioned wrong-object / wrong-instance metrics
  OpenVLA+BTS comparison target
```

---

## 8. Next engineering steps

1. Add first-contact/contact-object extraction if available from MuJoCo contacts.
2. Build LIBERO-Spatial policy-eval wrapper that can call any action policy and log BTS diagnostics.
3. Run OpenVLA LIBERO-Object checkpoint smoke on a GPU-capable stack.
4. If OpenVLA works, port the same diagnostics to OpenVLA outputs.
5. Later: train/fine-tune BTS module on LIBERO-Spatial demonstrations.
