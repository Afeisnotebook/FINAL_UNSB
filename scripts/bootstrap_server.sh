#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_PATH="${FINAL_UNSB_VENV:-$REPO/.venv}"

"$PYTHON_BIN" -m venv "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install --upgrade pip
"$VENV_PATH/bin/python" -m pip install \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$VENV_PATH/bin/python" -m pip install -r "$REPO/environment/requirements.txt"
# LPIPS is a required e100+ scientific guard, not an optional reporting extra.
# Instantiate it during bootstrap so a clean server downloads and validates the
# frozen AlexNet trunk before any multi-hour run reaches a terminal milestone.
"$VENV_PATH/bin/python" - <<'PY'
import lpips

model = lpips.LPIPS(net="alex")
assert model is not None
PY
"$VENV_PATH/bin/python" "$REPO/tools/capture_environment.py" \
  --output "$REPO/reports/inbox/ENVIRONMENT.local.json"
