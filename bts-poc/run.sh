#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# BTS PoC end-to-end runner
# ------------------------------------------------------------
# 用法：
#   bash run.sh               # belief 版本
#   bash run.sh --single      # 單點 baseline
# ------------------------------------------------------------

MODE="belief"
if [[ "${1:-}" == "--single" ]]; then
  MODE="single"
fi

echo "[1/3] Generate dataset..."
python -m data.generate \
  --out-dir data/gridworld_mixed \
  --n-episodes 20000 \
  --size 7 \
  --horizon 20 \
  --n-objects 4 \
  --spec-mode mixed \
  --test-holdout-tasks green_triangle

if [[ "$MODE" == "belief" ]]; then
  OUTDIR="runs/bts_poc_belief"
  EXTRA=""
else
  OUTDIR="runs/bts_poc_single"
  EXTRA="--single-point-baseline"
fi

echo "[2/3] Train model ($MODE)..."
python train.py \
  --train-jsonl data/gridworld_mixed/train.jsonl \
  --test-jsonl data/gridworld_mixed/test.jsonl \
  --out-dir "$OUTDIR" \
  --epochs 15 \
  --batch-size 256 \
  --lr 3e-4 \
  --lambda-belief 0.5 \
  --d-model 128 \
  --layers 4 \
  --heads 4 \
  $EXTRA

echo "[3/3] Evaluate phenomena..."
python eval/phenomena.py \
  --ckpt "$OUTDIR/best.pt" \
  --test-jsonl data/gridworld_mixed/test.jsonl \
  --out-dir "$OUTDIR/figs"

echo "Done. Outputs in $OUTDIR"
