#!/usr/bin/env python3
"""GroundingDINO smoke test on GB10: load weights, run one gRefCOCO image,
print candidate boxes+scores. Confirms no silent failure / no compile hell."""
import json, torch
from PIL import Image
from transformers import GroundingDinoForObjectDetection, AutoProcessor

MODEL = "IDEA-Research/grounding-dino-base"
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "torch", torch.__version__)

proc = AutoProcessor.from_pretrained(MODEL)
model = GroundingDinoForObjectDetection.from_pretrained(MODEL).to(dev).eval()
print("model loaded OK")

# one gRefCOCO val sample
base = "/home/p76141495/selective-grounding"
with open(f"{base}/dump/owlvit_gref_val.jsonl") as f:
    r = json.loads(f.readline())
img_path = f"{base}/data/coco/images/train2014/{r['image_file']}"
expr = r['expression']
print("image:", r['image_file'], "| expr:", expr, "| n_gt:", r['n_gt'])

image = Image.open(img_path).convert("RGB")
# GroundingDINO wants lowercase text ending with period
text = expr.lower().strip()
if not text.endswith('.'): text += '.'
inputs = proc(images=image, text=text, return_tensors="pt").to(dev)
with torch.no_grad():
    out = model(**inputs)

# post-process to boxes+scores (box_threshold low to get full candidate pool)
results = proc.post_process_grounded_object_detection(
    out, threshold=0.0, target_sizes=[image.size[::-1]])[0]
scores = results['scores']
boxes = results['boxes']
print(f"candidates: {len(scores)}  top5 scores:",
      [round(float(s),3) for s in scores[:5].tolist()] if len(scores) else [])
print("top1 box:", [round(float(x),1) for x in boxes[0].tolist()] if len(boxes) else None)
print("SMOKE OK")
