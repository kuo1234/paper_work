from __future__ import annotations

"""OpenVLA adapter skeleton for LIBERO diagnostics.

This file intentionally avoids importing OpenVLA at module import time. The `bts_libero`
container is for LIBERO diagnostics and has no GPU-capable OpenVLA stack. Use this as the
contract for a separate OpenVLA environment (e.g. lab A6000 or an OpenVLA container).

Expected external use:

    python libero_object_rollout_diagnostics.py \
      --suite libero_spatial \
      --policy external \
      --policy-adapter openvla_policy_adapter:openvla_policy

Adapter contract:

    def openvla_policy(obs: dict, context: dict) -> list[float]

OpenVLA expects a PIL image and prompt, then returns 7D action after unnormalization.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class OpenVLAAdapterConfig:
    checkpoint: str = "openvla/openvla-7b-finetuned-libero-spatial"
    image_key: str = "agentview_image"
    wrist_image_key: str = "robot0_eye_in_hand_image"
    center_crop: bool = True
    unnorm_key: str | None = None
    use_wrist: bool = False
    use_proprio: bool = False


_MODEL = None
_PROCESSOR = None
_CONFIG = OpenVLAAdapterConfig()


def configure(**kwargs) -> None:
    """Configure adapter before first use.

    Example:
        configure(checkpoint="openvla/openvla-7b-finetuned-libero-spatial", use_wrist=False)
    """
    global _CONFIG
    data = _CONFIG.__dict__.copy()
    data.update(kwargs)
    _CONFIG = OpenVLAAdapterConfig(**data)


def _lazy_load():
    """Load OpenVLA via pure HuggingFace trust_remote_code.

    The published checkpoints (e.g. openvla/openvla-7b-finetuned-libero-spatial) ship
    their own configuration_prismatic.py / modeling_prismatic.py via the config auto_map.
    Loading with trust_remote_code=True is therefore self-contained: it does NOT require
    the local `prismatic` package, `dlimp`, or TensorFlow. This is what makes a spark-only
    (aarch64 GB10, torch 2.12+cu130) eval feasible despite OpenVLA pinning torch 2.2+cu121.
    """
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR

    import os
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    # Allow per-run checkpoint override without editing code (useful in the eval loop):
    #   OPENVLA_CHECKPOINT=openvla/openvla-7b-finetuned-libero-object
    ckpt = os.environ.get("OPENVLA_CHECKPOINT", _CONFIG.checkpoint)
    if ckpt != _CONFIG.checkpoint:
        configure(checkpoint=ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    _PROCESSOR = AutoProcessor.from_pretrained(_CONFIG.checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        _CONFIG.checkpoint,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    _MODEL = model.to(device).eval()
    return _MODEL, _PROCESSOR


def make_prompt(language: str) -> str:
    return f"In: What action should the robot take to {language}?\nOut:"


def _to_pil(image: np.ndarray):
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PIL is required for OpenVLA image conversion") from exc
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)


def openvla_policy(obs: Dict[str, Any], context: Dict[str, Any]) -> List[float]:
    """External adapter entrypoint for LIBERO diagnostics.

    Faithfully mirrors OpenVLA's official run_libero_eval preprocessing:
      1. agentview_image rotated 180 deg (img[::-1, ::-1]) to match training.
      2. resized to 224 (lanczos), then 90% center-crop+resize if center_crop (train aug).
      3. predict_action via HF, then gripper normalize [0,1]->[-1,+1] (binarized) and inverted.

    Returns a 7D action [dx, dy, dz, droll, dpitch, dyaw, gripper] for LIBERO OSC_POSE.
    """
    import numpy as np
    import torch
    from PIL import Image

    model, processor = _lazy_load()

    raw = np.asarray(obs[_CONFIG.image_key])
    raw = raw[::-1, ::-1]  # rotate 180 deg to match OpenVLA training preprocessing
    if raw.dtype != np.uint8:
        raw = np.clip(raw, 0, 255).astype(np.uint8)
    image = Image.fromarray(np.ascontiguousarray(raw)).resize((224, 224), Image.LANCZOS)
    if _CONFIG.center_crop:
        image = _center_crop(image, scale=0.9)

    language = context.get("language") or "complete the task"
    prompt = make_prompt(language)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = processor(prompt, image).to(device, dtype=dtype)

    unnorm_key = _CONFIG.unnorm_key or _default_unnorm_key(_CONFIG.checkpoint)
    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    action = np.asarray(action, dtype=float).reshape(-1)[:7]
    action = _postprocess_gripper(action)
    return action.tolist()


def _postprocess_gripper(action):
    """Normalize gripper [0,1]->[-1,+1] (binarized) then invert sign, per OpenVLA LIBERO eval.

    normalize_gripper_action(binarize=True): y = 2x - 1, then sign() -> {-1,+1}
    invert_gripper_action: y = -y  (env uses -1=open, +1=close)
    """
    import numpy as np

    g = 2.0 * (float(action[-1]) - 0.0) / (1.0 - 0.0) - 1.0  # [0,1] -> [-1,+1]
    g = float(np.sign(g)) if g != 0 else -1.0  # binarize
    g = -g  # invert
    action[-1] = g
    return action


def _center_crop(image, scale: float = 0.9):
    """Center crop matching OpenVLA LIBERO eval (crop to 90% then resize back)."""
    w, h = image.size
    cw, ch = int(round(w * scale)), int(round(h * scale))
    left = (w - cw) // 2
    top = (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch)).resize((w, h))


def _default_unnorm_key(checkpoint: str) -> str:
    # Checkpoint names use hyphens (finetuned-libero-object); norm_stats keys use
    # underscores (libero_object). Normalize before matching.
    name = checkpoint.lower().replace("-", "_")
    for key in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        if key in name:
            return key
    return "libero_spatial"


def _apply_target_translation_gate(action, obs: Dict[str, Any], context: Dict[str, Any]):
    """Replace action[:3] with clipped translation toward context['target_key']."""
    import os
    import numpy as np

    action = np.asarray(action, dtype=float).reshape(-1)[:7].copy()
    target_key = context.get("target_key")
    if target_key and target_key in obs and "robot0_eef_pos" in obs:
        gain = float(os.environ.get("BTS_TARGET_GATE_GAIN", "5.0"))
        clip = float(os.environ.get("BTS_TARGET_GATE_CLIP", "0.20"))
        delta = np.asarray(obs[target_key], dtype=float) - np.asarray(obs["robot0_eef_pos"], dtype=float)
        action[:3] = np.clip(gain * delta, -clip, clip)
    return action


def openvla_target_gate_policy(obs: Dict[str, Any], context: Dict[str, Any]) -> List[float]:
    """Diagnostic BTS/OpenVLA upper-bound: gate translation toward the BDDL target.

    This is NOT the final learned BTS method. It is an oracle diagnostic that asks:
    if structured binding selects the correct target instance/object, is wrong-object
    contact eliminated on the fixed OpenVLA failure cases?

    It preserves OpenVLA's rotation + gripper output, but replaces action[:3] with a
    clipped proportional controller toward context['target_key'] (e.g. cream_cheese_1_pos).
    """
    action = openvla_policy(obs, context)
    return _apply_target_translation_gate(action, obs, context).tolist()


def openvla_target_gate_until_contact_policy(obs: Dict[str, Any], context: Dict[str, Any]) -> List[float]:
    """Gate translation only until the target has been contacted, then release to OpenVLA.

    The always-gated oracle fixes first-contact binding but can hurt placement/success because it
    keeps pulling toward the object. This variant tests a more realistic intervention boundary:
    use structured binding for acquisition, then hand manipulation back to the VLA.
    """
    target_key = context.get("target_key") or ""
    target_name = target_key[:-4] if target_key.endswith("_pos") else target_key
    for step in context.get("trace_so_far") or []:
        c = step.get("contact_object")
        name = c.get("object") if c else None
        if name and target_name and (name == target_name or name.startswith(target_name + "_")):
            return openvla_policy(obs, context)
    return openvla_target_gate_policy(obs, context)


def openvla_directional_bts_gate_policy(obs: Dict[str, Any], context: Dict[str, Any]) -> List[float]:
    """Selective BTS gate: intervene only when OpenVLA translation points to a wrong object.

    Diagnostic heuristic for the next BTS step. Before target contact, compute the endpoint of
    OpenVLA's proposed translation (`eef + action[:3]`). If the nearest object to that endpoint
    is a non-target object (and closer than the target by `BTS_DIRECTIONAL_MARGIN`), replace the
    translation with the oracle target-gate translation. Otherwise leave OpenVLA untouched.

    This tests whether a *selective* structured-binding gate can fix mis-binding without the
    broad success regressions of an always-on gate.
    """
    import os
    import re
    import numpy as np

    base = np.asarray(openvla_policy(obs, context), dtype=float).reshape(-1)[:7]
    target_key = context.get("target_key") or ""
    target_name = target_key[:-4] if target_key.endswith("_pos") else target_key

    # Release once the target has been contacted.
    for step in context.get("trace_so_far") or []:
        c = step.get("contact_object")
        name = c.get("object") if c else None
        if name and target_name and (name == target_name or name.startswith(target_name + "_")):
            return base.tolist()

    if not target_key or target_key not in obs or "robot0_eef_pos" not in obs:
        return base.tolist()

    def stem(name: str | None) -> str | None:
        return re.sub(r"_\d+$", "", name) if name else None

    target_stem = stem(target_name)
    eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
    endpoint = eef + np.asarray(base[:3], dtype=float)
    target_pos = np.asarray(obs[target_key], dtype=float)
    target_dist = float(np.linalg.norm(endpoint - target_pos))

    best = None
    for obj in context.get("object_names") or []:
        pos_key = f"{obj}_pos"
        if pos_key not in obs:
            continue
        dist = float(np.linalg.norm(endpoint - np.asarray(obs[pos_key], dtype=float)))
        if best is None or dist < best["dist"]:
            best = {"object": obj, "dist": dist}

    if best is None:
        return base.tolist()

    margin = float(os.environ.get("BTS_DIRECTIONAL_MARGIN", "0.00"))
    points_to_wrong = stem(best["object"]) != target_stem and best["dist"] + margin < target_dist
    if not points_to_wrong:
        return base.tolist()

    gated = _apply_target_translation_gate(base, obs, context)
    return gated.tolist()


def zero_policy(obs: Dict[str, Any], context: Dict[str, Any]) -> List[float]:
    """Safe no-op adapter for testing import path in environments without OpenVLA."""
    return [0.0] * 7
