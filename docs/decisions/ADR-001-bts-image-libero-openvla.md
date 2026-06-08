# ADR-001: Use controlled image-binding benchmark first, LIBERO-Object as primary real benchmark, and OpenVLA as VLA comparison

## Status
Accepted

## Date
2026-06-08

## Context
BTS pivoted from 3D-Diffuser-Actor/CALVIN closed-loop reproduction to image-language manipulation. The project needs a benchmark and VLA comparison path that preserves the core scientific claim: structured belief helps object-attribute binding under ambiguity.

Constraints:
- 3D-DA CALVIN public checkpoint reproduction remained low on GB10 and A6000, so CALVIN/3D-DA cannot be the sole foundation.
- The first image route must produce fast evidence for the binding mechanism, not be blocked by simulator/data setup.
- A real manipulation benchmark is still needed for external validity.
- The comparison should include a recognizable open VLA baseline.
- Training/eval resources are limited; first experiments must avoid full-scale VLA fine-tuning.

## Decision
Use a three-stage benchmark and comparison plan:

1. **Controlled image-binding benchmark first** for mechanism validation.
   - Synthetic or lightweight image environment.
   - Direct labels for target object, attributes, first contact, wrong-object/wrong-attribute errors.
   - Explicit train/OOD splits for object-attribute recombination and spurious shortcut removal.

2. **LIBERO-Object as the first real benchmark**, followed by **LIBERO-Spatial**.
   - LIBERO-Object is the closest first real suite for object identity / object-binding diagnostics.
   - LIBERO-Spatial is next for relational binding.
   - Avoid starting with LIBERO-10/Long because long-horizon failures confound binding diagnostics.
   - Avoid starting with LIBERO-Goal for pure object-attribute binding because it stresses goal variation more than object identity.

3. **OpenVLA as the primary VLA comparison**.
   - Start with existing LIBERO fine-tuned checkpoints and small eval runs.
   - Prefer frozen features + action head or LoRA before full fine-tuning.
   - Compare OpenVLA-only vs OpenVLA+BTS on binding-sensitive diagnostics.

CALVIN image remains a secondary/optional benchmark. Current CALVIN validation tools are useful for diagnostics, but full CALVIN training extraction should not block progress.

## Alternatives Considered

### CALVIN image as primary benchmark
- Pros: Existing env/data/eval context; validation image smoke already works; many binding-sensitive windows.
- Cons: Training split absent locally; full training data is large; prior CALVIN reproduction exposed environment/debug risk; object diagnostics require custom parsing.
- Rejected as primary: keep as secondary continuity target, not the main route.

### LIBERO-10 / LIBERO-Long first
- Pros: More impressive long-horizon benchmark.
- Cons: Long-horizon confounds make it hard to isolate object-attribute binding; slower debug cycle.
- Rejected for first benchmark: use after object/spatial diagnostics work.

### OpenVLA full fine-tuning first
- Pros: Strongest direct VLA comparison.
- Cons: Heavy compute; setup complexity; may distract from mechanism validation.
- Rejected for first implementation: start with existing checkpoints, frozen features, or LoRA.

### Octo as primary VLA comparison
- Pros: Strong open robot policy; useful design reference.
- Cons: OpenVLA has clearer VLA identity and existing LIBERO checkpoint/eval path.
- Rejected as primary: keep as secondary/design reference.

### Only controlled synthetic benchmark
- Pros: Cleanest mechanism evidence and fastest iteration.
- Cons: Too toy-like alone; limited external validity.
- Rejected as complete plan: accepted only as Stage A, followed by LIBERO/OpenVLA.

## Consequences

Positive:
- BTS can first prove the binding mechanism under controlled conditions.
- LIBERO provides a practical real benchmark aligned with VLA workflows.
- OpenVLA gives a clear, recognizable VLA comparison.
- The plan avoids being blocked by CALVIN training data or full VLA fine-tuning.

Negative / risks:
- A custom controlled benchmark may be seen as toy-like unless followed by LIBERO.
- LIBERO object metadata/diagnostics are feasible but not plug-and-play; BDDL and simulator state inspection are needed.
- OpenVLA setup may still require significant environment work and VRAM.

Mitigations:
- Treat controlled benchmark as mechanism evidence only.
- Use LIBERO-Object and LIBERO-Spatial for real benchmark evidence.
- Start OpenVLA with existing LIBERO fine-tuned checkpoints and small trial counts.
- Keep diagnostics narrow: target object, wrong-object/wrong-attribute, first-contact, binding-sensitive success.

## Implementation Notes

LIBERO setup likely needs a Python 3.8-style environment for native LIBERO. OpenVLA uses a newer Python/PyTorch stack, so native LIBERO and OpenVLA may require separate environments.

Known useful OpenVLA LIBERO command shape:

```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True
```

Start with small trial counts before full 10 tasks × 50 episodes.

## Follow-up Tasks

1. Implement controlled image-binding benchmark v0.
2. Research/install LIBERO-Object smoke in isolated environment.
3. Inspect LIBERO BDDL/task metadata for target-object diagnostics.
4. Evaluate existing OpenVLA LIBERO-Object checkpoint on a small run.
5. Design OpenVLA+BTS integration only after baseline eval works.
