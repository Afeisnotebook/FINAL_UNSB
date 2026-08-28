#!/usr/bin/env python3
"""Semantically compare two full-state checkpoints while ignoring metadata."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def equal(left, right) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict) and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right) and len(left) == len(right)
            and all(equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()
    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)
    components = ["networks", "optimizers", "schedulers", "rng", "method_state"]
    result = {component: equal(left[component], right[component]) for component in components}
    result["exact_training_state"] = all(result.values())
    print(json.dumps(result, indent=2))
    return 0 if result["exact_training_state"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
