# Image Evidence Module — Feasibility Probes (2026-06-10)

These scripts test whether a **deployable image-based evidence module** can replace the oracle
BDDL target gate used in `BTS_OPENVLA_INTERVENTION_REPORT.md`. The module must answer, from an
agentview image + the language target name, **which visible object is the target** — this is the
precondition for a belief-selective intervention gate (decide *when* to override OpenVLA's binding).

Substrate: 50 LIBERO-Object evidence frames (10 tasks × 5 inits, 256px OpenVLA-rotated agentview),
`runs/libero_object_evidence_v0/`. Object world positions are projected to pixels with the
robosuite camera matrix (`get_camera_transform_matrix` + `project_points_from_world_to_camera`,
then 180° flip to match the OpenVLA-rotated image).

Pass bar (from the stop-loss plan, Phase 3): fixed-failure top-1 ≥ 2/3, broader top-2 = 3/3.

## Result: the route does not stabilize. This is a negative result.

| Probe | Script | top-1 | fixed failures (cream/butter/choc) |
|---|---|---:|---|
| Projection self-consistency | `clip_diag.py` | 0.28 (vs 0.14 random) | — projection is correct, crops land on objects |
| CLIP-crop zero-shot | `clip_evidence.py` | 0.10–0.12 | ≈ random |
| CLIP-crop trained head (LOTO) | `clip_train.py` | 0.04 | worse than zero-shot |
| OWL-ViT (OWLv2) target-only | `owl_evidence.py` | 0.40 | **0 / 1 / 1 = 2/3** |
| OWL-ViT prompt × radius sweep | `owl_sweep.py` | 0.40 (locked) | locked at 0/1/1 across all configs |
| OWL-ViT score normalization (3 schemes) | `owl_norm.py` | 0.10 / 0.28 | did not recover CONFUSED |

## Root cause (the load-bearing finding): `owl_errattr.py`

Error attribution over 50 frames at the best config:

```
HIT 18 / CONFUSED 32 / MISSED 0      (detect_rate = 1.00 for every object)
```

Zero misses — OWLv2 always detects the target. Every error is CONFUSED: the target is detected but
a **distractor outscores it**, because OWLv2 raw scores carry a strong, target-independent
**per-object prior**:

```
high: orange_juice 0.20, milk 0.13, ketchup 0.12, butter 0.12
low : tomato_sauce 0.008, chocolate_pudding 0.014, cream_cheese 0.017, alphabet_soup 0.021
```

So `cream_cheese` (target) always loses to a co-present `milk`, regardless of the query. This is why:
- tuning prompt template / projection radius does nothing (`owl_sweep.py`: bias is a per-object constant);
- per-object normalization (self-name contrast, softmax-over-names) fails to recover it (`owl_norm.py`);
- results are implementation-sensitive (raw scoring swings 0.20↔0.40 on near-box detail) — the signal
  sits at the noise floor.

CLIP-crop dies outright because CLIP cannot recognize small objects in 256px simulator renders.

## Conclusion

The diagnosis is clean and publishable as a limitation: **off-the-shelf zero-shot open-vocabulary
detectors have an uncorrectable per-object score bias on LIBERO simulator renders**, so they cannot
serve as a stable deployable evidence module without detector fine-tuning (which violates the
low-training-cost discipline). The oracle-evidence intervention results in
`BTS_OPENVLA_INTERVENTION_REPORT.md` remain valid (Level B: diagnosis + oracle intervention).

## Repro (on spark)

```
ssh -i nvsync.key p76141495@192.168.65.11   # see memory/spark-ssh-access
cd ~/bts-poc/experiments
MUJOCO_GL=egl HF_HOME=~/bts/hf_cache PYTHONPATH=~/bts-poc/experiments \
  ~/openvla-spark/.venv/bin/python <probe>.py
# OWLv2 weights: google/owlv2-base-patch16-ensemble; CLIP: openai/clip-vit-base-patch32
```

Logs from the runs that produced the numbers above are in `logs/`.
