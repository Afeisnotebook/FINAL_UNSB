"""Lane-blind full-data evaluation; confirmation samples are unaddressable."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from production.metrics import (
    METRIC_SEMANTICS,
    bridge_times,
    build_rollout_bundle,
    bundle_hash,
    psnr_unit,
    ssim_unit,
    to_unit,
)
from research.local_route1.runtime import capture_rng, restore_rng, write_json

from .protocol import EVALUATION_SCHEMA, LaneSpec, load_protocol, object_sha256


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

        package_version = importlib.metadata.version("lpips")
        if package_version != "0.1.4":
            raise RuntimeError(
                "paper LPIPS is frozen to lpips==0.1.4; "
                f"found {package_version}"
            )
        model = lpips.LPIPS(
            net="alex", version="0.1", lpips=True, spatial=False,
            eval_mode=True, verbose=False,
        ).to(device).eval()
        model.requires_grad_(False)
        return model
    except Exception as error:
        raise RuntimeError(
            "LPIPS was requested by the frozen paper protocol but the "
            "lpips==0.1.4 AlexNet evaluator could not be loaded"
        ) from error


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


def aggregate_metric_rows(rows: list[dict]) -> dict:
    """Macro-average one fixed NFE/replicate slice through six domains."""
    grouped = defaultdict(list)
    for row in rows:
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
    if not domains:
        raise RuntimeError("cannot aggregate an empty paper evaluation slice")
    return {
        "macro_psnr": float(np.mean([row["psnr"] for row in domains.values()])),
        "macro_ssim": float(np.mean([row["ssim"] for row in domains.values()])),
        "macro_lpips": (
            None if any(row["lpips"] is None for row in domains.values())
            else float(np.mean([row["lpips"] for row in domains.values()]))
        ),
        "domains": domains,
    }


def replicate_stochasticity(cells: list[dict]) -> dict:
    if not cells:
        raise RuntimeError("paper stochasticity needs at least one rollout bundle")
    return {
        "macro_psnr_std": float(np.std([row["macro_psnr"] for row in cells])),
        "macro_ssim_std": float(np.std([row["macro_ssim"] for row in cells])),
        "macro_lpips_std": (
            None if any(row["macro_lpips"] is None for row in cells)
            else float(np.std([row["macro_lpips"] for row in cells]))
        ),
        "ddof": 0,
        "replicate_count": len(cells),
    }


def _close(left: Any, right: Any, *, label: str) -> None:
    if left is None or right is None:
        if left is not right:
            raise RuntimeError(f"paper metric {label} differs: {left!r} != {right!r}")
        return
    if not math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f"paper metric {label} differs: {left!r} != {right!r}")


def _validate_aggregate(actual: dict, expected: dict, *, label: str) -> None:
    for key in ("macro_psnr", "macro_ssim", "macro_lpips"):
        _close(actual.get(key), expected.get(key), label=f"{label}.{key}")
    actual_domains = actual.get("domains")
    expected_domains = expected.get("domains")
    if not isinstance(actual_domains, dict) or set(actual_domains) != set(expected_domains):
        raise RuntimeError(f"paper metric {label} has the wrong domain aggregate set")
    for domain, expected_row in expected_domains.items():
        actual_row = actual_domains[domain]
        if int(actual_row.get("n", -1)) != int(expected_row["n"]):
            raise RuntimeError(f"paper metric {label}.{domain}.n differs")
        for key in ("psnr", "ssim", "lpips"):
            _close(
                actual_row.get(key), expected_row.get(key),
                label=f"{label}.{domain}.{key}",
            )


def evaluation_sample_identity(result: dict) -> list[tuple[str, str, int]]:
    """Return the lane-independent discovery identities in a metric result."""
    return sorted({
        (str(row["domain"]), str(row["stem"]), int(row["order"]))
        for row in result.get("images", [])
    })


def evaluation_crn_identity(result: dict) -> list[tuple[str, str, int, int, str]]:
    """Return one CRN identity per image and replicate, independent of NFE."""
    values: dict[tuple[str, str, int, int], str] = {}
    for row in result.get("images", []):
        digest = row.get("crn_bundle_sha256")
        if digest is None:
            continue
        key = (
            str(row["domain"]), str(row["stem"]), int(row["order"]),
            int(row["replicate"]),
        )
        if key in values and values[key] != digest:
            raise RuntimeError("paper metric changes CRN bundle across NFE")
        values[key] = str(digest)
    return sorted((*key, digest) for key, digest in values.items())


def validate_evaluation_result(
    result: dict, *, lane_id: str, family: str, count_per_domain: int,
    replicates: int, nfe_values: list[int], include_lpips: bool,
    input_reference: bool = False,
) -> dict[str, Any]:
    """Recompute a complete paper metric result from its per-image evidence.

    This is intentionally independent of receipts and source-host metadata.  It
    prevents a correctly hashed but incomplete or internally inconsistent JSON
    payload from entering the unified paper cohort.
    """
    protocol = load_protocol()
    count_per_domain = int(count_per_domain)
    replicates = int(replicates)
    nfe_values = [int(value) for value in nfe_values]
    if count_per_domain < 1 or replicates < 1 or not nfe_values:
        raise RuntimeError("paper evaluation dimensions must be positive")
    if len(set(nfe_values)) != len(nfe_values):
        raise RuntimeError("paper evaluation NFE values must be unique")
    primary_nfe = (
        0 if input_reference else
        int(protocol["evaluation"]["primary_unsb_nfe"])
        if family == "unsb" else 1
    )
    primary_replicate = int(protocol["evaluation"]["primary_replicate"])
    if primary_nfe not in nfe_values or not 0 <= primary_replicate < replicates:
        raise RuntimeError("paper primary NFE or replicate is absent")
    expected_header = {
        "schema": EVALUATION_SCHEMA,
        "lane_id": lane_id,
        "split": "discovery",
        "count_per_domain": count_per_domain,
        "replicates": replicates,
        "nfe_values": nfe_values,
        "primary_nfe": primary_nfe,
        "primary_replicate": primary_replicate,
        "top_level_replicate_aggregation": "mean_over_fixed_replicates",
        "metric_semantics": METRIC_SEMANTICS,
        "metric_semantics_sha256": object_sha256(METRIC_SEMANTICS),
        "lpips_requested": bool(include_lpips),
        "lpips_available": True if include_lpips else None,
        "confirmation20_opened": False,
    }
    for key, expected in expected_header.items():
        if result.get(key) != expected:
            raise RuntimeError(
                f"paper evaluation header mismatch for {key}: "
                f"{result.get(key)!r} != {expected!r}"
            )

    images = result.get("images")
    if not isinstance(images, list):
        raise RuntimeError("paper evaluation lacks per-image evidence")
    domains = sorted({str(row.get("domain")) for row in images if isinstance(row, dict)})
    if len(domains) != int(protocol["manifest"]["domains"]):
        raise RuntimeError(f"paper evaluation must contain six domains, got {domains}")
    expected_total = len(domains) * count_per_domain * replicates * len(nfe_values)
    if len(images) != expected_total:
        raise RuntimeError(
            f"paper evaluation image-cell count differs: {len(images)} != {expected_total}"
        )

    seen = set()
    crn_by_replicate: dict[tuple[str, str, int, int], str] = {}
    for row in images:
        if not isinstance(row, dict):
            raise RuntimeError("paper evaluation image row is not an object")
        try:
            domain = str(row["domain"])
            stem = str(row["stem"])
            order = int(row["order"])
            replicate = int(row["replicate"])
            nfe = int(row["nfe"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("paper evaluation image identity is malformed") from error
        key = (domain, stem, order, replicate, nfe)
        if key in seen:
            raise RuntimeError(f"duplicate paper evaluation image cell: {key}")
        seen.add(key)
        if domain not in domains or not 0 <= order < count_per_domain:
            raise RuntimeError(f"paper evaluation image order is out of range: {key}")
        if not 0 <= replicate < replicates or nfe not in nfe_values:
            raise RuntimeError(f"paper evaluation replicate/NFE is out of range: {key}")
        for metric in ("psnr", "ssim"):
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(f"paper image {metric} is not finite: {key}")
        if float(row["psnr"]) > 120.0 + 1e-9:
            raise RuntimeError(f"paper image PSNR exceeds the frozen numerical ceiling: {key}")
        if not -1.01 <= float(row["ssim"]) <= 1.01:
            raise RuntimeError(f"paper image SSIM is outside its numerical range: {key}")
        lpips_value = row.get("lpips")
        if include_lpips:
            if not isinstance(lpips_value, (int, float)) or not math.isfinite(
                float(lpips_value)
            ):
                raise RuntimeError(f"paper image LPIPS is unavailable or non-finite: {key}")
        elif lpips_value is not None:
            raise RuntimeError(f"paper image unexpectedly contains LPIPS: {key}")
        digest = row.get("crn_bundle_sha256")
        if input_reference:
            if digest is not None:
                raise RuntimeError("deterministic Input reference must not claim a CRN bundle")
        else:
            if not isinstance(digest, str) or len(digest) != 64:
                raise RuntimeError(f"paper image CRN digest is malformed: {key}")
            try:
                int(digest, 16)
            except ValueError as error:
                raise RuntimeError(f"paper image CRN digest is not hexadecimal: {key}") from error
            crn_key = (domain, stem, order, replicate)
            prior = crn_by_replicate.setdefault(crn_key, digest)
            if prior != digest:
                raise RuntimeError("paper metric changes CRN bundle across NFE")

    for domain in domains:
        domain_rows = [row for row in images if str(row["domain"]) == domain]
        identities = {(str(row["stem"]), int(row["order"])) for row in domain_rows}
        if len(identities) != count_per_domain:
            raise RuntimeError(f"paper evaluation {domain} lacks unique discovery identities")
        if {order for _, order in identities} != set(range(count_per_domain)):
            raise RuntimeError(f"paper evaluation {domain} order is incomplete")
        for replicate in range(replicates):
            for nfe in nfe_values:
                cell = [
                    row for row in domain_rows
                    if int(row["replicate"]) == replicate and int(row["nfe"]) == nfe
                ]
                if len(cell) != count_per_domain:
                    raise RuntimeError(
                        f"paper evaluation cell is incomplete: {domain} r{replicate} nfe{nfe}"
                    )

    expected_cells = {}
    for nfe in nfe_values:
        nfe_rows = [row for row in images if int(row["nfe"]) == nfe]
        aggregate = aggregate_metric_rows(nfe_rows)
        replicate_cells = []
        for replicate in range(replicates):
            replicate_rows = [
                row for row in nfe_rows if int(row["replicate"]) == replicate
            ]
            replicate_cells.append({
                "replicate": replicate,
                **aggregate_metric_rows(replicate_rows),
            })
        expected_cells[str(nfe)] = {
            **aggregate,
            "replicate_cells": replicate_cells,
            "stochasticity": replicate_stochasticity(replicate_cells),
        }
    actual_cells = result.get("nfe_cells")
    if not isinstance(actual_cells, dict) or set(actual_cells) != set(expected_cells):
        raise RuntimeError("paper evaluation has the wrong NFE-cell set")
    for nfe, expected in expected_cells.items():
        actual = actual_cells[nfe]
        _validate_aggregate(actual, expected, label=f"nfe_cells.{nfe}")
        actual_replicates = actual.get("replicate_cells")
        if not isinstance(actual_replicates, list) or len(actual_replicates) != replicates:
            raise RuntimeError(f"paper evaluation NFE {nfe} replicate cells differ")
        for index, expected_replicate in enumerate(expected["replicate_cells"]):
            actual_replicate = actual_replicates[index]
            if int(actual_replicate.get("replicate", -1)) != index:
                raise RuntimeError(f"paper evaluation NFE {nfe} replicate order differs")
            _validate_aggregate(
                actual_replicate, expected_replicate,
                label=f"nfe_cells.{nfe}.replicate.{index}",
            )
        for key, value in expected["stochasticity"].items():
            if key in ("ddof", "replicate_count"):
                if actual.get("stochasticity", {}).get(key) != value:
                    raise RuntimeError(f"paper evaluation NFE {nfe} stochasticity {key} differs")
            else:
                _close(
                    actual.get("stochasticity", {}).get(key), value,
                    label=f"nfe_cells.{nfe}.stochasticity.{key}",
                )

    primary = expected_cells[str(primary_nfe)]
    _validate_aggregate(result, primary, label="primary")
    actual_primary_replicates = result.get("replicate_cells")
    if actual_primary_replicates != actual_cells[str(primary_nfe)]["replicate_cells"]:
        raise RuntimeError("paper primary replicate cells differ from primary NFE")
    actual_stochasticity = result.get("stochasticity")
    if actual_stochasticity != actual_cells[str(primary_nfe)]["stochasticity"]:
        raise RuntimeError("paper primary stochasticity differs from primary NFE")
    return {
        "domains": domains,
        "sample_identity": evaluation_sample_identity(result),
        "crn_identity": evaluation_crn_identity(result),
        "image_cells": len(images),
    }


@torch.no_grad()
def evaluate_input_baseline(
    *, rows: list[dict], data_root: Path, protocol_hash: str,
    count_per_domain: int, device: torch.device, include_lpips: bool,
) -> dict:
    """Evaluate the degraded input itself under the frozen discovery split.

    Input is an evaluation-only deterministic reference, not a trainable lane.
    It therefore has NFE zero and one deterministic replicate, but it shares
    the exact image ordering, resize, target metrics and evaluator environment
    used by every model in the final one-container table.
    """
    protocol = load_protocol()
    selected = select_discovery(rows, count_per_domain)
    saved_rng = capture_rng()
    perceptual = None
    image_rows = []
    try:
        perceptual = _lpips(device) if include_lpips else None
        for row in selected:
            image_size = int(protocol["evaluation"]["image_size"])
            source = read_image(
                Path(data_root) / row["input_relpath"], size=image_size,
            ).to(device)
            target = read_image(
                Path(data_root) / row["target_relpath"], size=image_size,
            ).to(device)
            source_unit = to_unit(source)
            target_unit = to_unit(target)
            lpips_value = None
            if perceptual is not None:
                lpips_value = float(perceptual(source, target).item())
            image_rows.append({
                "domain": row["domain"], "stem": row["stem"],
                "order": int(row["order"]), "replicate": 0, "nfe": 0,
                "psnr": psnr_unit(source_unit, target_unit),
                "ssim": ssim_unit(source_unit, target_unit),
                "lpips": lpips_value,
                "crn_bundle_sha256": None,
            })
    finally:
        restore_rng(saved_rng)
    aggregate = aggregate_metric_rows(image_rows)
    replicate_cell = {"replicate": 0, **aggregate}
    result = {
        "schema": EVALUATION_SCHEMA,
        "lane_id": "input",
        "split": "discovery",
        "count_per_domain": int(count_per_domain),
        "replicates": 1,
        "nfe_values": [0],
        "primary_nfe": 0,
        "primary_replicate": int(protocol["evaluation"]["primary_replicate"]),
        "top_level_replicate_aggregation": "mean_over_fixed_replicates",
        "metric_semantics": METRIC_SEMANTICS,
        "metric_semantics_sha256": object_sha256(METRIC_SEMANTICS),
        "protocol_fingerprint": protocol_hash,
        "evaluation_input_sha256": evaluation_input_hash(selected, protocol_hash),
        **{key: aggregate[key] for key in ("macro_psnr", "macro_ssim", "macro_lpips")},
        "domains": aggregate["domains"],
        "replicate_cells": [replicate_cell],
        "stochasticity": replicate_stochasticity([replicate_cell]),
        "nfe_cells": {"0": {**aggregate, "replicate_cells": [replicate_cell],
                              "stochasticity": replicate_stochasticity([replicate_cell])}},
        "images": image_rows,
        "lpips_requested": bool(include_lpips),
        "lpips_available": perceptual is not None if include_lpips else None,
        "evaluation_only_reference": True,
        "confirmation20_opened": False,
    }
    validate_evaluation_result(
        result, lane_id="input", family="input",
        count_per_domain=count_per_domain, replicates=1, nfe_values=[0],
        include_lpips=include_lpips, input_reference=True,
    )
    return result


def evaluate_model(
    model, *, spec: LaneSpec, rows: list[dict], data_root: Path,
    protocol_hash: str, count_per_domain: int, replicates: int,
    nfe_values: list[int], include_lpips: bool,
) -> dict:
    protocol = load_protocol()
    selected = select_discovery(rows, count_per_domain)
    replicates = int(replicates)
    nfe_values = [int(value) for value in nfe_values]
    if replicates < 1:
        raise ValueError("paper evaluation requires at least one replicate")
    if not nfe_values or len(set(nfe_values)) != len(nfe_values):
        raise ValueError("paper evaluation requires unique NFE values")
    primary_nfe = (
        int(protocol["evaluation"]["primary_unsb_nfe"])
        if spec.family == "unsb" else 1
    )
    if primary_nfe not in nfe_values:
        raise ValueError(f"paper primary NFE {primary_nfe} is absent")
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
            image_size = int(protocol["evaluation"]["image_size"])
            source = read_image(
                Path(data_root) / row["input_relpath"], size=image_size,
            ).to(model.device)
            target = read_image(
                Path(data_root) / row["target_relpath"], size=image_size,
            ).to(model.device)
            for replicate in range(replicates):
                bundle = build_rollout_bundle(
                    protocol_hash=protocol_hash, domain=row["domain"], stem=row["stem"],
                    replicate=replicate, latent_dim=4 * int(getattr(model.opt, "ngf", 64)),
                    height=image_size, width=image_size,
                    num_timesteps=int(protocol["unsb"]["num_timesteps"]),
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
        aggregate = aggregate_metric_rows(values)
        replicate_cells = [
            {
                "replicate": replicate,
                **aggregate_metric_rows([
                    row for row in values if int(row["replicate"]) == replicate
                ]),
            }
            for replicate in range(replicates)
        ]
        cells[str(nfe)] = {
            **aggregate,
            "replicate_cells": replicate_cells,
            "stochasticity": replicate_stochasticity(replicate_cells),
        }
    primary = cells[str(primary_nfe)]
    result = {
        "schema": EVALUATION_SCHEMA,
        "lane_id": spec.id,
        "split": "discovery",
        "count_per_domain": int(count_per_domain),
        "replicates": replicates,
        "nfe_values": list(nfe_values),
        "primary_nfe": primary_nfe,
        "primary_replicate": int(protocol["evaluation"]["primary_replicate"]),
        "top_level_replicate_aggregation": "mean_over_fixed_replicates",
        "metric_semantics": METRIC_SEMANTICS,
        "metric_semantics_sha256": object_sha256(METRIC_SEMANTICS),
        "protocol_fingerprint": protocol_hash,
        "evaluation_input_sha256": evaluation_input_hash(selected, protocol_hash),
        **{key: primary[key] for key in ("macro_psnr", "macro_ssim", "macro_lpips")},
        "domains": primary["domains"],
        "replicate_cells": primary["replicate_cells"],
        "stochasticity": primary["stochasticity"],
        "nfe_cells": cells,
        "images": image_rows,
        "lpips_requested": bool(include_lpips),
        "lpips_available": perceptual is not None if include_lpips else None,
        "confirmation20_opened": False,
    }
    validate_evaluation_result(
        result, lane_id=spec.id, family=spec.family,
        count_per_domain=count_per_domain, replicates=replicates,
        nfe_values=nfe_values, include_lpips=include_lpips,
    )
    return result


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
