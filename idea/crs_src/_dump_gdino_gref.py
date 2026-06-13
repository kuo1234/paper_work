#!/usr/bin/env python
"""gRefCOCO dump for GroundingDINO (HF transformers, no compile) — matches the
owlvit gref schema exactly so crs_gate / ltt / p1 reuse it unchanged.

GroundingDINO: open-set detector, ~900 candidate queries, score = phrase match.
We feed the referring expression as the text prompt (lowercased, period-terminated),
keep top-PRED_KEEP candidates with score>=PRED_MIN_SCORE.
"""
import os, json, argparse, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/selective-grounding/data/hf_cache"))
import numpy as np
import torch
from PIL import Image
from transformers import GroundingDinoForObjectDetection, AutoProcessor

TOPK = 20
PRED_KEEP = 50
PRED_MIN_SCORE = 0.01

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grefs", required=True)
    p.add_argument("--instances", default="")
    p.add_argument("--split", default="val")
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompt", default="{q}")
    p.add_argument("--prompt_variant", default="canonical")
    p.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--log_every", type=int, default=500)
    return p.parse_args()

def xywh_to_xyxy(b):
    x, y, w, h = b
    return [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)]

@torch.no_grad()
def main():
    args = get_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(args.model)
    model = GroundingDinoForObjectDetection.from_pretrained(args.model).to(dev).eval()

    grefs = json.load(open(args.grefs))
    refs = [r for r in grefs if r["split"] == args.split]
    if args.limit:
        refs = refs[:args.limit]

    ann_by_id = {}
    if args.instances:
        inst = json.load(open(args.instances))
        ann_by_id = {a["id"]: a for a in inst["annotations"]}
        print(f"[gref] loaded {len(ann_by_id)} annotations for GT join", flush=True)

    n_nt = sum(1 for r in refs if r["no_target"])
    print(f"[gref] split={args.split} refs={len(refs)} no_target={n_nt} target={len(refs)-n_nt}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    idx = 0; t0 = time.time(); img_file_cur = None; img = None
    with open(args.out, "w") as f:
        for r in refs:
            coco_file = f"COCO_train2014_{r['image_id']:012d}.jpg"
            if coco_file != img_file_cur:
                img = Image.open(os.path.join(args.images, coco_file)).convert("RGB")
                img_file_cur = coco_file
            W, H = img.size

            gt_boxes = []
            if not r["no_target"] and ann_by_id:
                for aid in r["ann_id"]:
                    a = ann_by_id.get(aid)
                    if a is not None:
                        gt_boxes.append(xywh_to_xyxy(a["bbox"]))

            for sent in r["sentences"]:
                expr = sent["sent"]
                q = args.prompt.format(q=expr)
                text = q.lower().strip()
                if not text.endswith("."):
                    text += "."
                inp = proc(images=img, text=text, return_tensors="pt").to(dev)
                out = model(**inp)
                ts = [(H, W)]
                res = proc.post_process_grounded_object_detection(
                    out, threshold=0.0, target_sizes=ts)[0]
                scores = res["scores"]
                boxes = res["boxes"]
                order = torch.argsort(scores, descending=True)
                sc_sorted = scores[order]
                bx_sorted = boxes[order]

                top1 = float(sc_sorted[0]) if len(sc_sorted) else 0.0
                top2 = float(sc_sorted[1]) if len(sc_sorted) > 1 else 0.0
                topk = sc_sorted[:TOPK].float().cpu().numpy()
                if topk.size == 0:
                    topk = np.array([0.0])
                pdist = topk / max(topk.sum(), 1e-8)
                entropy = float(-(pdist * np.log(pdist + 1e-12)).sum())

                keep = min(PRED_KEEP, len(sc_sorted))
                pred_boxes = []; pred_scores = []
                for i in range(keep):
                    s = float(sc_sorted[i])
                    if s < PRED_MIN_SCORE:
                        break
                    b = bx_sorted[i].tolist()
                    pred_boxes.append([round(float(x), 2) for x in b])
                    pred_scores.append(round(s, 6))

                rec = {
                    "sample_uid": f"gref:{args.split}:{r['ref_id']}:{sent['sent_id']}:{idx}",
                    "dataset": "grefcoco", "split": args.split,
                    "ref_id": r["ref_id"], "sent_id": sent["sent_id"],
                    "image_file": coco_file, "expression": expr,
                    "prompt_variant": args.prompt_variant, "base": "gdino",
                    "no_target": bool(r["no_target"]),
                    "n_gt": len(gt_boxes),
                    "top1_score": round(top1, 6), "top2_score": round(top2, 6),
                    "margin12": round(top1 - top2, 6), "score_entropy": round(entropy, 6),
                    "score_mean_topk": round(float(topk.mean()), 6),
                    "score_std_topk": round(float(topk.std()), 6),
                    "n_cands": int(len(scores)),
                    "pred_boxes_xyxy": pred_boxes, "pred_scores": pred_scores,
                    "gt_boxes_xyxy": gt_boxes,
                }
                f.write(json.dumps(rec) + "\n")
                idx += 1
            if idx and idx % args.log_every < len(r["sentences"]):
                rate = idx / (time.time() - t0)
                print(f"  {idx} rows {rate:.1f} it/s", flush=True)
    print(f"[gref] wrote {idx} rows -> {args.out} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
