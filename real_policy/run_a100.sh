#!/usr/bin/env bash
set -euo pipefail

POLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REAL_DATA_ROOT:-}" ]]; then
  echo "REAL_DATA_ROOT must point to the audited processed-data directory" >&2
  exit 2
fi
DATA_ROOT="${REAL_DATA_ROOT}"
OUTPUT_ROOT="${REAL_OUTPUT_ROOT:-${POLICY_ROOT}/artifacts}"
MAX_HOURS="${MAX_HOURS:-9.25}"
STEPS="${STEPS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-192}"

if python - "$MAX_HOURS" <<'PY'
import sys
raise SystemExit(0 if 0 < float(sys.argv[1]) <= 9.5 else 1)
PY
then
  :
else
  echo "MAX_HOURS must be in (0, 9.5]" >&2
  exit 2
fi

cd "$POLICY_ROOT"
export PYTHONPATH="$POLICY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
timeout --signal=TERM 34200s python -u training/train_real.py \
  --data "$DATA_ROOT" \
  --output "$OUTPUT_ROOT" \
  --max-hours "$MAX_HOURS" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE"

test -s "$OUTPUT_ROOT/training_complete.json"
