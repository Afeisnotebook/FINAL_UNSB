#!/usr/bin/env python3
"""Capture the portable environment identity used by local or server preflight."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--porcelain"),
        "nvidia_smi": command(
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader"
        ),
    }
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["torch_cuda"] = torch.version.cuda
        payload["cudnn"] = torch.backends.cudnn.version()
        payload["cuda_available"] = torch.cuda.is_available()
        payload["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as error:
        payload["torch_error"] = repr(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
