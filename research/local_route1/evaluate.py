"""Lane-blind CRN evaluation on discovery70; confirmation is unaddressable."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from production.metrics import (
    build_rollout_bundle,
    bundle_hash,
    psnr_unit,
    rollout_endpoint,
    ssim_unit,
    to_unit,
)

from .protocol import EVALUATION_SCHEMA
from .runtime import capture_rng, restore_rng


def select_discovery70(rows: list[dict], count_per_domain: int = 70) -> list[dict]:
    if int(count_per_domain) != 70:
        raise RuntimeError("route-1 evaluation is frozen to discovery70")
    selected = []
    for domain in sorted({row["domain"] for row in rows}):
        candidates = sorted(
            (row for row in rows if row["domain"] == domain and row["split"] == "discovery"),
            key=lambda row: int(row["order"]),
        )
        take = candidates[:70]
        if len(take) != 70:
            raise RuntimeError(f"{domain}: discovery70 is incomplete")
        selected.extend(take)
    if any(row.get("split") != "discovery" for row in selected):
        raise RuntimeError("confirmation20 access blocked")
    return selected


def read_image(path: Path, size: int = 128) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        value = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(value).permute(2, 0, 1).contiguous().unsqueeze(0)


def _lpips(device):
    try:
        import lpips

        return lpips.LPIPS(net="alex").to(device).eval()
    except Exception:
        return None


def evaluation_input_hash(selected: list[dict], protocol_hash: str) -> str:
    digest = hashlib.sha256()
    digest.update(EVALUATION_SCHEMA.encode())
    digest.update(str(protocol_hash).encode())
    for row in selected:
        digest.update(f'{row["domain"]}|{row["stem"]}|{row["order"]}'.encode())
    return digest.hexdigest()


def evaluate_model(
    model, *, rows: list[dict], data_root: Path, protocol_hash: str,
    include_lpips: bool,
) -> dict:
    selected = select_discovery70(rows)
    saved_rng = capture_rng()
    modes = {name: getattr(model, "net" + name).training for name in model.model_names}
    perceptual = None
    image_rows = []
    try:
        model.eval()
        perceptual = _lpips(model.device) if include_lpips else None
        for row in selected:
            source = read_image(Path(data_root) / row["input_relpath"]).to(model.device)
            target = read_image(Path(data_root) / row["target_relpath"]).to(model.device)
            bundle = build_rollout_bundle(
                protocol_hash=protocol_hash,
                domain=row["domain"], stem=row["stem"], replicate=0,
                latent_dim=4 * int(model.opt.ngf), height=128, width=128,
                num_timesteps=int(model.opt.num_timesteps),
            )
            with torch.no_grad():
                endpoint = rollout_endpoint(
                    model.netG, source, bundle, tau=float(model.opt.tau)
                ).clamp(-1.0, 1.0)
                prediction_unit = to_unit(endpoint)
                target_unit = to_unit(target)
                lpips_value = None
                if perceptual is not None:
                    lpips_value = float(perceptual(endpoint, target).item())
            image_rows.append({
                "domain": row["domain"],
                "stem": row["stem"],
                "order": int(row["order"]),
                "psnr": psnr_unit(prediction_unit, target_unit),
                "ssim": ssim_unit(prediction_unit, target_unit),
                "lpips": lpips_value,
                "crn_bundle_sha256": bundle_hash(bundle),
            })
    finally:
        for name, was_training in modes.items():
            getattr(model, "net" + name).train(was_training)
        restore_rng(saved_rng)

    grouped = defaultdict(list)
    for row in image_rows:
        grouped[row["domain"]].append(row)
    domains = {}
    for domain, values in sorted(grouped.items()):
        domains[domain] = {
            "n": len(values),
            "psnr": float(np.mean([row["psnr"] for row in values])),
            "ssim": float(np.mean([row["ssim"] for row in values])),
            "lpips": (
                None if any(row["lpips"] is None for row in values)
                else float(np.mean([row["lpips"] for row in values]))
            ),
        }
    return {
        "schema": EVALUATION_SCHEMA,
        "split": "discovery",
        "count_per_domain": 70,
        "replicates": 1,
        "protocol_fingerprint": protocol_hash,
        "evaluation_input_sha256": evaluation_input_hash(selected, protocol_hash),
        "macro_psnr": float(np.mean([row["psnr"] for row in domains.values()])),
        "macro_ssim": float(np.mean([row["ssim"] for row in domains.values()])),
        "macro_lpips": (
            None if any(row["lpips"] is None for row in domains.values())
            else float(np.mean([row["lpips"] for row in domains.values()]))
        ),
        "lpips_requested": bool(include_lpips),
        "lpips_available": perceptual is not None if include_lpips else None,
        "domains": domains,
        "images": image_rows,
        "confirmation20_opened": False,
    }


def compare_to_plain(method: dict, plain: dict, *, epoch: int) -> dict:
    identity_keys = ("schema", "count_per_domain", "protocol_fingerprint", "evaluation_input_sha256")
    for key in identity_keys:
        if method.get(key) != plain.get(key):
            raise RuntimeError(f"evaluation identity mismatch for {key}")
    domain_delta = {}
    for domain in sorted(plain["domains"]):
        m, p = method["domains"][domain], plain["domains"][domain]
        domain_delta[domain] = {
            "psnr": m["psnr"] - p["psnr"],
            "ssim": m["ssim"] - p["ssim"],
            "lpips": (
                None if m["lpips"] is None or p["lpips"] is None
                else m["lpips"] - p["lpips"]
            ),
        }
    psnr_deltas = [row["psnr"] for row in domain_delta.values()]
    lpips_delta = (
        None if method["macro_lpips"] is None or plain["macro_lpips"] is None
        else method["macro_lpips"] - plain["macro_lpips"]
    )
    return {
        "epoch": int(epoch),
        "updates": int(epoch) * 150,
        "macro_psnr": method["macro_psnr"],
        "plain_macro_psnr": plain["macro_psnr"],
        "macro_psnr_delta": method["macro_psnr"] - plain["macro_psnr"],
        "macro_ssim_delta": method["macro_ssim"] - plain["macro_ssim"],
        "macro_lpips_delta": lpips_delta,
        "positive_domains": sum(value > 0.0 for value in psnr_deltas),
        "worst_domain_delta": min(psnr_deltas),
        "domain_delta": domain_delta,
        "guardrails_pass": (
            min(psnr_deltas) > -1.0
            and method["macro_ssim"] >= plain["macro_ssim"]
            and (lpips_delta is None or lpips_delta <= 0.0)
        ),
        "confirmation20_opened": False,
    }
