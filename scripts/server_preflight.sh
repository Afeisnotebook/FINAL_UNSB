#!/usr/bin/env bash
set -euo pipefail

source "${1:-server.env}"
cd "$FINAL_UNSB_REPO"
PY="${FINAL_UNSB_PYTHON:-$FINAL_UNSB_REPO/.venv/bin/python}"

test -n "$FINAL_UNSB_DATA"
test -n "$FINAL_UNSB_VIEW"
test -n "$FINAL_UNSB_RUNS"
test -n "$FINAL_UNSB_LANE"
git diff --exit-code
git diff --cached --exit-code
"$PY" - <<'PY'
import lpips

model = lpips.LPIPS(net="alex")
assert model is not None
PY
"$PY" tools/validate_contracts.py
"$PY" tools/build_data_manifest.py \
  --data-root "$FINAL_UNSB_DATA" \
  --output manifests/FULL_DATA_MANIFEST.local.csv \
  --hash-content
cmp manifests/FULL_DATA_MANIFEST.local.csv manifests/FULL_DATA_MANIFEST.csv
"$PY" tools/materialize_views.py \
  --manifest manifests/FULL_DATA_MANIFEST.csv \
  --data-root "$FINAL_UNSB_DATA" \
  --view-root "$FINAL_UNSB_VIEW" \
  --mode auto
"$PY" tools/capture_environment.py \
  --output "reports/inbox/${FINAL_UNSB_LANE}_ENVIRONMENT.json"
"$PY" -m production.train_lane \
  --lane "$FINAL_UNSB_LANE" \
  --data-view "$FINAL_UNSB_VIEW" \
  --manifest manifests/FULL_DATA_MANIFEST.csv \
  --run-root "$FINAL_UNSB_RUNS" \
  --gpu-id "$FINAL_UNSB_GPU" \
  --e0-only
