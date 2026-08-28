#!/usr/bin/env bash
set -euo pipefail

source "${1:-server.env}"
cd "$FINAL_UNSB_REPO"
PY="${FINAL_UNSB_PYTHON:-$FINAL_UNSB_REPO/.venv/bin/python}"
for EPOCH in 1 10 25 50 100 150 200; do
  CHECKPOINT="$FINAL_UNSB_RUNS/$FINAL_UNSB_LANE/milestones/full_state_e${EPOCH}.pt"
  test -f "$CHECKPOINT"
  "$PY" -m production.evaluate_lane \
    --lane "$FINAL_UNSB_LANE" \
    --checkpoint "$CHECKPOINT" \
    --manifest manifests/FULL_DATA_MANIFEST.csv \
    --data-root "$FINAL_UNSB_DATA" \
    --split discovery \
    --output "reports/inbox/${FINAL_UNSB_LANE}_DISCOVERY_E${EPOCH}.json" \
    --gpu-id "$FINAL_UNSB_GPU" \
    --replicates 4
done
