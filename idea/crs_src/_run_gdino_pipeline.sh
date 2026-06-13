#!/usr/bin/env bash
# Auto pipeline: wait for running val dump -> testA -> testB -> full analysis.
# Launched detached; survives session. Results land in dump/RESULT_*.txt
set -e
cd ~/selective-grounding
G=data/grefcoco/grefs_unc.json
I=data/grefcoco/instances.json
IMG=data/coco/images/train2014
P=src/dump_gdino_gref.py
PY=.venv/bin/python

echo "[pipe] $(date) waiting for val dump..."
while pgrep -f "dump_gdino_gref.py.*--split val" >/dev/null; do sleep 60; done
echo "[pipe] val done: $(wc -l < dump/gdino_gref_val.jsonl) rows"

echo "[pipe] testA dump..."
$PY $P --grefs $G --instances $I --split testA --images $IMG \
  --out dump/gdino_gref_testA.jsonl --log_every 2000 > dump/gdino_testA.log 2>&1
echo "[pipe] testA done: $(wc -l < dump/gdino_gref_testA.jsonl) rows"

echo "[pipe] testB dump..."
$PY $P --grefs $G --instances $I --split testB --images $IMG \
  --out dump/gdino_gref_testB.jsonl --log_every 2000 > dump/gdino_testB.log 2>&1
echo "[pipe] testB done: $(wc -l < dump/gdino_gref_testB.jsonl) rows"

echo "[pipe] running analyses..."
$PY src/tp_separability.py > dump/RESULT_tp_separability.txt 2>&1 || true
$PY src/money_figure.py    > dump/RESULT_money_figure.txt 2>&1 || true
$PY src/full_results.py    > dump/RESULT_full_table.txt 2>&1 || true
echo "[pipe] $(date) ALL DONE"
