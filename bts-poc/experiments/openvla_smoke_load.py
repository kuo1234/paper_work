"""Decisive OpenVLA load+inference smoke test for the spark-only (GB10) route.

Run on spark with the dedicated openvla venv:

    ssh p76141495@192.168.65.11 'cd ~/bts && export HF_HOME=~/bts/hf_cache; \
      ~/openvla-spark/.venv/bin/python openvla_smoke_load.py'

Confirms the spark GB10 stack (torch 2.12+cu130) loads OpenVLA-7B via pure HF
trust_remote_code and runs predict_action without silent numeric failure.

Verified result (2026-06-09):
    load ~104s, predict_action ~1.4s, 7D finite nonzero action, SMOKE_OK
"""

import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

CKPT = "openvla/openvla-7b-finetuned-libero-spatial"


def main():
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to("cuda").eval()
    print("loaded in %.1fs" % (time.time() - t0))
    print("norm_stats keys:", list(getattr(model, "norm_stats", {}).keys()))

    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
    prompt = "In: What action should the robot take to pick up the black bowl?\nOut:"
    inputs = proc(prompt, img).to("cuda", dtype=torch.bfloat16)

    t1 = time.time()
    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)
    print("predict_action in %.2fs" % (time.time() - t1))

    a = np.asarray(action, dtype=float).reshape(-1)
    print("action shape:", a.shape)
    print("action:", np.round(a, 4).tolist())
    print("finite:", bool(np.all(np.isfinite(a))), "nonzero:", bool(np.any(a != 0)))
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
