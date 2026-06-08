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
    """Load OpenVLA dependencies lazily.

    This is a skeleton. Fill this in inside an OpenVLA-capable environment.
    Typical OpenVLA code path uses transformers AutoProcessor/AutoModelForVision2Seq
    or repo-specific helpers, depending on the OpenVLA checkout.
    """
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR
    raise RuntimeError(
        "OpenVLA adapter skeleton called without implementation. "
        "Install/load OpenVLA in a GPU-capable environment and implement _lazy_load()."
    )


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

    Returns a 7D action. In skeleton mode this raises until _lazy_load is implemented.
    """
    model, processor = _lazy_load()
    image = _to_pil(obs[_CONFIG.image_key])
    language = context.get("language") or "complete the task"
    prompt = make_prompt(language)

    # Pseudocode; adapt to the exact OpenVLA repo/API in the runtime environment:
    # action = model.predict_action(image, prompt, unnorm_key=_CONFIG.unnorm_key, center_crop=_CONFIG.center_crop)
    # return np.asarray(action, dtype=float).reshape(-1)[:7].tolist()
    raise NotImplementedError(
        "Implement OpenVLA predict_action call here in the OpenVLA environment. "
        f"Prepared prompt={prompt!r}, checkpoint={_CONFIG.checkpoint!r}."
    )


def zero_policy(obs: Dict[str, Any], context: Dict[str, Any]) -> List[float]:
    """Safe no-op adapter for testing import path in environments without OpenVLA."""
    return [0.0] * 7
