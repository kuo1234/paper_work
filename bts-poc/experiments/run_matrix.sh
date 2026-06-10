#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# v7 實驗 matrix：{single,dual} hint × {full,partial} obs × N seeds
# 每組每 seed 訓 belief + single 兩個模型，再跑 phenomena eval。
# headline 對照 = dual_partial vs dual_full（隔離 partial-obs 引入的位置不確定）。
# 用法：
#   bash experiments/run_matrix.sh            # 完整規模
#   SEEDS="0" N_EPISODES=300 EPOCHS=1 OBS_MODES="full partial" bash experiments/run_matrix.sh  # smoke
# ------------------------------------------------------------

# 可由環境變數覆蓋的參數（頂部集中）
SEEDS="${SEEDS:-0 1 2 3 4}"
OBS_MODES="${OBS_MODES:-full partial}"
N_EPISODES="${N_EPISODES:-8000}"
EPOCHS="${EPOCHS:-15}"
HORIZON="${HORIZON:-20}"
SIZE="${SIZE:-7}"
N_OBJECTS="${N_OBJECTS:-4}"
HOLDOUT="${HOLDOUT:-green_triangle}"
BATCH="${BATCH:-256}"
DMODEL="${DMODEL:-128}"
LAYERS="${LAYERS:-4}"
HEADS="${HEADS:-4}"
LAMBDA="${LAMBDA:-0.5}"

DATA_ROOT="data/exp"
RUN_ROOT="runs/exp"
mkdir -p "$DATA_ROOT" "$RUN_ROOT"

# variant 名稱 → n_hints
declare -A NHINTS=( ["single"]=1 ["dual"]=2 )

for variant in single dual; do
  nh="${NHINTS[$variant]}"
  for obs in $OBS_MODES; do
    if [ "$obs" = "partial" ]; then
      obs_flag="--no-observe-object-identity"
    else
      obs_flag="--observe-object-identity"
    fi
    for seed in $SEEDS; do
      tag="${variant}_${obs}_s${seed}"
      ddir="${DATA_ROOT}/${tag}"
      echo "==== [${tag}] generate (n_hints=${nh}, obs=${obs}, seed=${seed}) ===="
      python -m data.generate \
        --out-dir "$ddir" \
        --n-episodes "$N_EPISODES" \
        --size "$SIZE" --horizon "$HORIZON" --n-objects "$N_OBJECTS" \
        --spec-mode mixed --test-holdout-tasks "$HOLDOUT" \
        --n-hints "$nh" --seed "$seed" $obs_flag >/dev/null

      echo "==== [${tag}] train belief ===="
      python train.py \
        --train-jsonl "${ddir}/train.jsonl" --test-jsonl "${ddir}/test.jsonl" \
        --out-dir "${RUN_ROOT}/${tag}_belief" \
        --epochs "$EPOCHS" --batch-size "$BATCH" --lr 3e-4 --lambda-belief "$LAMBDA" \
        --d-model "$DMODEL" --layers "$LAYERS" --heads "$HEADS" --seed "$seed" >/dev/null

      echo "==== [${tag}] train single ===="
      python train.py \
        --train-jsonl "${ddir}/train.jsonl" --test-jsonl "${ddir}/test.jsonl" \
        --out-dir "${RUN_ROOT}/${tag}_single" \
        --epochs "$EPOCHS" --batch-size "$BATCH" --lr 3e-4 --lambda-belief "$LAMBDA" \
        --d-model "$DMODEL" --layers "$LAYERS" --heads "$HEADS" --seed "$seed" \
        --single-point-baseline >/dev/null

      echo "==== [${tag}] eval ===="
      python eval/phenomena.py \
        --ckpt "${RUN_ROOT}/${tag}_belief/best.pt" \
        --baseline-ckpt "${RUN_ROOT}/${tag}_single/best.pt" \
        --test-jsonl "${ddir}/test.jsonl" \
        --out-dir "${RUN_ROOT}/${tag}_compare/figs" >/dev/null
      echo "==== [${tag}] done ===="
    done
  done
done

echo "ALL DONE. Summarize with: python experiments/summarize.py"
