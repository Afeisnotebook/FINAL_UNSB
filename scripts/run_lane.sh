#!/usr/bin/env bash
set -euo pipefail

source "${1:-server.env}"
cd "$FINAL_UNSB_REPO"
PY="${FINAL_UNSB_PYTHON:-$FINAL_UNSB_REPO/.venv/bin/python}"
ARGS=(
  --lane "$FINAL_UNSB_LANE"
  --data-view "$FINAL_UNSB_VIEW"
  --manifest manifests/FULL_DATA_MANIFEST.csv
  --run-root "$FINAL_UNSB_RUNS"
  --gpu-id "$FINAL_UNSB_GPU"
)
if [[ "${FINAL_UNSB_RESUME:-0}" == "1" ]]; then
  ARGS+=(--resume)
fi
"$PY" -m production.train_lane "${ARGS[@]}"
