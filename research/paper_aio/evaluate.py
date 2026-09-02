"""Lane-blind full-data evaluation; confirmation samples are unaddressable."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from production.metrics import (
    bridge_times,
    build_rollout_bundle,
    bundle_hash,
    psnr_unit,
    ssim_unit,
    to_unit,
)
from research.local_route1.runtime import capture_rng, restore_rng, write_json

from .protocol import EVALUATION_SCHEMA, LaneSpec, load_protocol


def select_discovery(rows: list[dict], count_per_domain: int) -> list[dict]:
    protocol = load_protocol()
    allowed = {
        int(protocol["evaluation"]["trajectory_discovery_per_domain"]),
        int(protocol["evaluation"]["terminal_discovery_per_domain"]),
    }
    if int(count_per_domain) not in allowed:
        raise RuntimeError(f"paper evaluation count is not frozen: {count_per_domain}")
    selected = []
    for domain in sorted({row["domain"] for row in rows}):
        candidates = [row for row in rows if row["domain"] == domain and row["split"] == "discovery"]
        candidates.sort(key=lambda row: int(row["order"]))
        take = candidates[: int(count_per_domain)]
        if len(take) != int(count_per_domain):
            raise RuntimeError(f"{domain}: discovery split is incomplete")
        selected.extend(take)
    if any(row["split"] != "discovery" for row in selected):
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


@torch.no_grad()
def rollout_prefix(net_g, source: torch.Tensor, bundle: dict, *, nfe: int, tau: float):
    """Run the first ``nfe`` operators of the frozen five-step bridge schedule."""
    full_steps = len(bundle["z"])
    if nfe < 1 or nfe > full_steps:
        raise ValueError(f"nfe must be in [1,{full_steps}]")
    times = bridge_times(full_steps)
    state = source
    endpoint = None
    for step in range(nfe):
        if step > 0:
            delta = float(times[step] - times[step - 1])
            denominator = float(times[-1] - times[step - 1])
            alpha = delta / denominator
            variance = delta * (1.0 - alpha)
            state = (
                (1.0 - alpha) * state + alpha * endpoint.detach()
                + math.sqrt(variance * tau) * bundle["noise"][step].to(source.device)
            )
        time_index = torch.full(
            (source.shape[0],), step, dtype=torch.long, device=source.device,
        )
        endpoint = net_g(state, time_index, bundle["z"][step].to(source.device))
    return endpoint


def _prediction(model, spec: LaneSpec, source: torch.Tensor, bundle: dict, *, nfe: int):
    if spec.family == "unsb":
        return rollout_prefix(
            model.netG, source, bundle, nfe=nfe, tau=float(model.opt.tau),
        )
    if spec.id == "cyclegan":
        return model.netG_A(source)
    if spec.id == "cut":
        return model.netG(source)
    raise RuntimeError(f"no evaluation adapter for {spec.id}")


def evaluation_input_hash(selected: list[dict], protocol_hash: str) -> str:
    digest = hashlib.sha256()
    digest.update(EVALUATION_SCHEMA.encode())
    digest.update(protocol_hash.encode())
    for row in selected:
        digest.update(f'{row["domain"]}|{row["stem"]}|{row["order"]}'.encode())
    return digest.hexdigest()


def evaluate_model(
    model, *, spec: LaneSpec, rows: list[dict], data_root: Path,
    protocol_hash: str, count_per_domain: int, replicates: int,
    nfe_values: list[int], include_lpips: bool,
) -> dict:
    protocol = load_protocol()
    selected = select_discovery(rows, count_per_domain)
    saved_rng = capture_rng()
    modes = {
        name: getattr(model, "net" + name).training for name in model.model_names
    }
    perceptual = None
    image_rows = []
    try:
        model.eval()
        perceptual = _lpips(model.device) if include_lpips else None
        for row in selected:
            source = read_image(Path(data_root) / row["input_relpath"]).to(model.device)
            target = read_image(Path(data_root) / row["target_relpath"]).to(model.device)
            for replicate in range(int(replicates)):
                bundle = build_rollout_bundle(
                    protocol_hash=protocol_hash, domain=row["domain"], stem=row["stem"],
                    replicate=replicate, latent_dim=4 * int(getattr(model.opt, "ngf", 64)),
                    height=128, width=128, num_timesteps=5,
                )
                for nfe in nfe_values:
                    endpoint = _prediction(model, spec, source, bundle, nfe=nfe).clamp(-1.0, 1.0)
                    unit = to_unit(endpoint)
                    target_unit = to_unit(target)
                    lpips_value = None
                    if perceptual is not None:
                        lpips_value = float(perceptual(endpoint, target).item())
                    image_rows.append({
                        "domain": row["domain"], "stem": row["stem"],
                        "order": int(row["order"]), "replicate": replicate,
                        "nfe": int(nfe), "psnr": psnr_unit(unit, target_unit),
                        "ssim": ssim_unit(unit, target_unit), "lpips": lpips_value,
                        "crn_bundle_sha256": bundle_hash(bundle),
                    })
    finally:
        for name, was_training in modes.items():
            getattr(model, "net" + name).train(was_training)
        restore_rng(saved_rng)

    cells = {}
    for nfe in nfe_values:
        values = [row for row in image_rows if row["nfe"] == nfe]
        grouped = defaultdict(list)
        for row in values:
            grouped[row["domain"]].append(row)
        domains = {}
        for domain, domain_rows in sorted(grouped.items()):
            domains[domain] = {
                "n": len(domain_rows),
                "psnr": float(np.mean([row["psnr"] for row in domain_rows])),
                "ssim": float(np.mean([row["ssim"] for row in domain_rows])),
                "lpips": (
                    None if any(row["lpips"] is None for row in domain_rows)
                    else float(np.mean([row["lpips"] for row in domain_rows]))
                ),
            }
        cells[str(nfe)] = {
            "macro_psnr": float(np.mean([row["psnr"] for row in domains.values()])),
            "macro_ssim": float(np.mean([row["ssim"] for row in domains.values()])),
            "macro_lpips": (
                None if any(row["lpips"] is None for row in domains.values())
                else float(np.mean([row["lpips"] for row in domains.values()]))
            ),
            "domains": domains,
        }
    primary_nfe = 5 if spec.family == "unsb" else 1
    if str(primary_nfe) not in cells:
        primary_nfe = int(nfe_values[-1])
    primary = cells[str(primary_nfe)]
    return {
        "schema": EVALUATION_SCHEMA,
        "lane_id": spec.id,
        "split": "discovery",
        "count_per_domain": int(count_per_domain),
        "replicates": int(replicates),
        "nfe_values": list(nfe_values),
        "primary_nfe": primary_nfe,
        "protocol_fingerprint": protocol_hash,
        "evaluation_input_sha256": evaluation_input_hash(selected, protocol_hash),
        **{key: primary[key] for key in ("macro_psnr", "macro_ssim", "macro_lpips")},
        "domains": primary["domains"],
        "nfe_cells": cells,
        "images": image_rows,
        "lpips_requested": bool(include_lpips),
        "lpips_available": perceptual is not None if include_lpips else None,
        "confirmation20_opened": False,
    }


def evaluate_live_model(
    *, model, spec: LaneSpec, rows: list[dict], data_root: Path,
    protocol_hash: str, epoch: int, lane_root: Path,
) -> dict:
    protocol = load_protocol()
    path = Path(lane_root) / "metrics" / f"e{epoch:03d}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    nfe_values = (
        list(protocol["evaluation"]["reported_unsb_nfes"])
        if spec.family == "unsb" and epoch in protocol["training"]["nfe_epochs"]
        else [5 if spec.family == "unsb" else 1]
    )
    terminal = int(epoch) == int(protocol["training"]["target_data_epochs"])
    result = evaluate_model(
        model, spec=spec, rows=rows, data_root=Path(data_root),
        protocol_hash=protocol_hash,
        count_per_domain=(
            int(protocol["evaluation"]["terminal_discovery_per_domain"])
            if terminal else int(protocol["evaluation"]["trajectory_discovery_per_domain"])
        ),
        replicates=(
            int(protocol["evaluation"]["terminal_replicates"])
            if terminal else 1
        ),
        nfe_values=nfe_values,
        include_lpips=epoch in protocol["training"]["lpips_epochs"],
    )
    result.update({"epoch": int(epoch), "updates": int(epoch) * 8553})
    write_json(path, result)
    return result
