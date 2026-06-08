---
type: experiment-report
aliases:
  - "BTS LIBERO Object diagnostics report"
  - "LIBERO real benchmark scaffold report"
tags:
  - BTS
  - LIBERO
  - diagnostics
  - real-benchmark
summary: "LIBERO-Object scaffold status: isolated Docker builds on spark, render works, task/BDDL parser works, rollout diagnostics distinguish no-op/random/target-reaching behavior via target-distance and nearest-object metrics."
---

# LIBERO-Object Diagnostics Scaffold Report（2026-06-09）

## 1. Purpose

Build the first real-benchmark scaffold for the BTS image/VLA route.

Goal:

```text
LIBERO-Object -> parse target object -> render image/state -> log whether policy approaches the correct object
```

This is not yet a learned policy benchmark. It verifies that LIBERO can support BTS wrong-object / binding diagnostics.

---

## 2. Environment

Isolated Docker setup:

```text
docker/libero/Dockerfile
docker/libero/build_and_run.sh
docker/libero/README.md
```

Spark result:

```text
image: bts_libero:latest
image id: 66dfb4bba6b5
size: 11.8GB
container: bts_libero
status: Up
```

Important caveat:

```text
torch.cuda.is_available() == False inside bts_libero
```

Use this container for LIBERO metadata/render/diagnostics. Use a different stack/machine for OpenVLA GPU.

---

## 3. LIBERO-Object metadata

Script:

```text
bts-poc/experiments/libero_object_diagnostics.py
```

Result:

```text
suite: libero_object
n_tasks: 10
init states per task: (50, 110)
```

Parsed tasks:

| task | target | receptacle | language |
|---:|---|---|---|
| 0 | alphabet_soup | basket | pick up the alphabet soup and place it in the basket |
| 1 | cream_cheese | basket | pick up the cream cheese and place it in the basket |
| 2 | salad_dressing | basket | pick up the salad dressing and place it in the basket |
| 3 | bbq_sauce | basket | pick up the bbq sauce and place it in the basket |
| 4 | ketchup | basket | pick up the ketchup and place it in the basket |
| 5 | tomato_sauce | basket | pick up the tomato sauce and place it in the basket |
| 6 | butter | basket | pick up the butter and place it in the basket |
| 7 | milk | basket | pick up the milk and place it in the basket |
| 8 | chocolate_pudding | basket | pick up the chocolate pudding and place it in the basket |
| 9 | orange_juice | basket | pick up the orange juice and place it in the basket |

All targets/receptacles are found in BDDL.

---

## 4. Render/state smoke

LIBERO `OffScreenRenderEnv` works with EGL in the isolated container.

Observation image keys:

```text
agentview_image             (128, 128, 3) uint8
robot0_eye_in_hand_image    (128, 128, 3) uint8
```

Diagnostic state keys include:

```text
robot0_eef_pos
object-state
{object}_1_pos
{object}_1_quat
{object}_1_to_robot0_eef_pos
```

This is enough for:

```text
target distance to gripper
nearest object to gripper
target-to-receptacle distance
first-nearest / sustained-nearest target metrics
```

---

## 5. Rollout diagnostics

Script:

```text
bts-poc/experiments/libero_object_rollout_diagnostics.py
```

Policies tested:

```text
noop
tiny random
target_reach
target_reach_fast
```

Rollout setting:

```text
LIBERO-Object first 3 tasks
2 init states each
6 rollouts total
```

### 5.1 Metrics

Per rollout:

```text
first_nearest_object
first_nearest_is_target
nearest_target_fraction
first_target_nearest_t
target_dist_initial
target_dist_final
target_dist_drop
success_seen
```

Aggregate:

```text
mean_target_dist_drop
mean_nearest_target_fraction
first_nearest_target_rate
success_count
```

### 5.2 Results

| policy | steps | success_count | mean target dist drop | mean nearest-target fraction | first-nearest target rate |
|---|---:|---:|---:|---:|---:|
| noop | 25 | 0 / 6 | 0.0163 | 0.0000 | 0.0000 |
| random | 25 | 0 / 6 | 0.0164 | 0.0000 | 0.0000 |
| target_reach | 25 | 0 / 6 | 0.0547 | 0.0000 | 0.0000 |
| target_reach_fast | 60 | 0 / 6 | 0.2437 | 0.3722 | 0.0000 |

Interpretation:

- `target_dist_drop` separates weak target approach from no-op/random.
- `nearest_target_fraction` is stricter and only activates when the policy gets close enough.
- `first_nearest_target_rate` is 0 because all tested rollouts start with a distractor closer than the target. This is useful for BTS: the benchmark has real wrong-object ambiguity.
- None of the heuristic policies solve the task because they do not grasp/place.

---

## 6. Why this matters for BTS

LIBERO-Object now supports the real-benchmark diagnostic path:

```text
language target object
  -> BDDL verification
  -> object-state positions
  -> per-step approach/nearest-object diagnostics
  -> future wrong-object / first-contact metrics
```

This creates a bridge from the controlled synthetic result to a real manipulation benchmark.

Current BTS story alignment:

```text
Controlled benchmark: structured binding fixes OOD wrong-object errors.
LIBERO-Object scaffold: real tasks expose target/distractor state needed to measure wrong-object behavior.
```

---

## 7. Remaining gaps

1. Need first-contact/contact-object extraction, not only nearest-object.
2. Need a learned or pretrained policy to evaluate, not heuristic reachers.
3. Need OpenVLA eval on GPU-capable stack.
4. Need LIBERO-Spatial parser for relational binding.
5. Need dataset/demo path for imitation learning.

---

## 8. Recommended next step

Do **not** jump to OpenVLA fine-tuning yet.

Next best engineering step:

```text
LIBERO-Spatial diagnostic parser
```

Reason: BTS is about binding; LIBERO-Object is object identity, while LIBERO-Spatial adds relational binding. Parsing it before training will show whether the real benchmark can test the richer claim.
