#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

MAX_HOURS="${MAX_HOURS:-8.75}"
STEPS="${STEPS:-12000}"
BATCH_SIZE="${BATCH_SIZE:-256}"

# The outer timeout is a second independent guard. The Python trainer uses a
# 9.5-hour maximum and checkpoints the best validation state before export.
timeout --signal=INT --kill-after=120s 33600s \
  python training/train.py \
    --output artifacts \
    --max-hours "$MAX_HOURS" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE"
