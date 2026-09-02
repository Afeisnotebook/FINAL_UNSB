"""Post-freeze KID/FID evaluation for fixed e200 paper checkpoints.

This module is deliberately unusable before an explicit algorithm, baseline
and claim freeze.  It renders the already-frozen discovery80 outputs, extracts
Clean-FID Inception features in one evaluator runtime, and reports KID as the
primary distribution metric with pooled FID as a small-sample supplement.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from production.metrics import build_rollout_bundle, bundle_hash, to_unit
from research.local_route1.runtime import capture_rng, restore_rng, write_json

from .evaluate import _prediction, evaluation_input_hash, read_image, select_discovery
from .gates import environment_record
from .protocol import (
    LaneSpec,
    EXPECTED_MANIFEST_SHA256,
    FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
    ROOT,
    evaluation_bundle_fingerprint,
    file_sha256,
    load_protocol,
    object_sha256,
    protocol_fingerprint,
)


SCHEMA = "final-unsb-paper-post-freeze-distribution-metrics-v1"
FREEZE_SCHEMA = "final-unsb-paper-algorithm-and-baseline-freeze-v1"
FREEZE_STATUS = "FROZEN_FULL_DATA_ALGORITHM_BASELINE_AND_CLAIM_SET"
MODE = "clean"
FEATURE_MODEL = "inception_v3"
KID_NUM_SUBSETS = 100
KID_MAX_SUBSET_SIZE = 1000
FEATURE_BATCH_SIZE = 32


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_freeze_receipt(path: Path, *, lane_id: str) -> dict[str, Any]:
    """Require an explicit committed freeze before distribution evaluation."""
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeError("post-freeze distribution evaluation needs a freeze receipt")
    value = _read_json(path)
    lanes = value.get("distribution_lanes")
    if (
        value.get("schema") != FREEZE_SCHEMA
        or value.get("status") != FREEZE_STATUS
        or value.get("algorithm_configuration_frozen") is not True
        or value.get("baseline_configuration_frozen") is not True
        or value.get("paper_claims_frozen") is not True
        or value.get("e200_results_frozen") is not True
        or int(value.get("primary_epoch", -1)) != 200
        or value.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or value.get("evaluation_bundle_fingerprint")
        != FROZEN_EVALUATION_BUNDLE_FINGERPRINT
        or not isinstance(value.get("source_portfolio_sha256"), str)
        or len(value["source_portfolio_sha256"]) != 64
        or not isinstance(lanes, list)
        or len(lanes) != len(set(lanes))
        or lane_id not in lanes
        or value.get("best_checkpoint_selection") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation_authorized") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("paper algorithm/baseline/claim freeze receipt is invalid")
    return value


def committed_freeze_identity(
    path: Path, *, lane_id: str,
) -> tuple[dict[str, Any], str]:
    """Prove that the exact freeze receipt exists in Git, not just on disk."""
    path = Path(path).resolve()
    value = validate_freeze_receipt(path, lane_id=lane_id)
    try:
        relative = path.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError("freeze receipt must be stored inside the repository") from error
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", relative], cwd=ROOT, text=True,
    ).strip()
    if status:
        raise RuntimeError("freeze receipt has uncommitted changes")
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=ROOT, text=True,
    ).strip()
    if len(commit) != 40:
        raise RuntimeError("freeze receipt has no committed Git identity")
    committed_text = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT, text=True,
    )
    try:
        committed = json.loads(committed_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("committed freeze receipt is not JSON") from error
    if object_sha256(committed) != object_sha256(value):
        raise RuntimeError("working freeze receipt differs from its committed Git blob")
    return value, commit


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _kid(fid_module, left: np.ndarray, right: np.ndarray, *, seed: int) -> float:
    if len(left) < 2 or len(right) < 2:
        raise RuntimeError("KID requires at least two features in each set")
    state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        return float(fid_module.kernel_distance(
            left, right, num_subsets=KID_NUM_SUBSETS,
            max_subset_size=KID_MAX_SUBSET_SIZE,
        ))
    finally:
        np.random.set_state(state)


def summarize_feature_sets(
    *, lane_id: str, target: dict[str, np.ndarray],
    predictions: list[dict[str, np.ndarray]], fid_module,
) -> dict[str, Any]:
    """Compute deterministic domain-macro KID and pooled KID/FID."""
    domains = sorted(target)
    if not domains or any(set(row) != set(domains) for row in predictions):
        raise RuntimeError("distribution feature domains are incomplete or inconsistent")
    target_counts = {domain: int(len(target[domain])) for domain in domains}
    if len(set(target_counts.values())) != 1:
        raise RuntimeError("distribution target domains are not balanced")
    target_pooled = np.concatenate([target[domain] for domain in domains], axis=0)
    replicate_rows = []
    bundle_identity = evaluation_bundle_fingerprint()
    for replicate, predicted in enumerate(predictions):
        domain_scores = {}
        for domain in domains:
            if len(predicted[domain]) != len(target[domain]):
                raise RuntimeError(f"{domain}: prediction/target feature count differs")
            domain_scores[domain] = {
                "n": int(len(target[domain])),
                "kid": _kid(
                    fid_module, predicted[domain], target[domain],
                    seed=_stable_seed(bundle_identity, lane_id, replicate, domain),
                ),
            }
        predicted_pooled = np.concatenate(
            [predicted[domain] for domain in domains], axis=0,
        )
        replicate_rows.append({
            "replicate": replicate,
            "macro_domain_kid": float(np.mean([
                row["kid"] for row in domain_scores.values()
            ])),
            "pooled_kid": _kid(
                fid_module, predicted_pooled, target_pooled,
                seed=_stable_seed(bundle_identity, lane_id, replicate, "pooled"),
            ),
            "pooled_fid": float(fid_module.fid_from_feats(
                predicted_pooled, target_pooled,
            )),
            "domains": domain_scores,
        })
    if not replicate_rows:
        raise RuntimeError("distribution evaluation requires at least one replicate")
    return {
        "replicates": replicate_rows,
        "summary": {
            "replicate_count": len(replicate_rows),
            "macro_domain_kid_mean": float(np.mean([
                row["macro_domain_kid"] for row in replicate_rows
            ])),
            "macro_domain_kid_std": float(np.std([
                row["macro_domain_kid"] for row in replicate_rows
            ])),
            "pooled_kid_mean": float(np.mean([
                row["pooled_kid"] for row in replicate_rows
            ])),
            "pooled_kid_std": float(np.std([
                row["pooled_kid"] for row in replicate_rows
            ])),
            "pooled_fid_mean": float(np.mean([
                row["pooled_fid"] for row in replicate_rows
            ])),
            "pooled_fid_std": float(np.std([
                row["pooled_fid"] for row in replicate_rows
            ])),
            "ddof": 0,
        },
        "target_counts": target_counts,
        "pooled_target_count": int(len(target_pooled)),
    }


def _save_png(path: Path, value: torch.Tensor) -> None:
    unit = to_unit(value.detach()).clamp(0.0, 1.0)
    array = unit.squeeze(0).permute(1, 2, 0).cpu().numpy()
    array = np.rint(array * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path, format="PNG")


def _filename(row: dict[str, Any]) -> str:
    stem_hash = hashlib.sha256(str(row["stem"]).encode("utf-8")).hexdigest()[:16]
    return f'{int(row["order"]):04d}_{stem_hash}.png'


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _cleanfid_runtime():
    try:
        import cleanfid
        from cleanfid import fid
    except Exception as error:
        raise RuntimeError("Clean-FID is required after the paper freeze") from error
    return cleanfid, fid


def _extract_features(
    *, roots: dict[str, Path], fid_module, feature_model, device: torch.device,
) -> dict[str, np.ndarray]:
    return {
        domain: fid_module.get_folder_features(
            str(root), model=feature_model, num_workers=0,
            batch_size=FEATURE_BATCH_SIZE, device=device, mode=MODE,
            description=f"{domain}: ", verbose=False,
        )
        for domain, root in sorted(roots.items())
    }


@torch.no_grad()
def profile_distribution(
    *, model, spec: LaneSpec | None, rows: list[dict[str, Any]], data_root: Path,
    destination: Path, freeze_receipt: Path, checkpoint: Path | None,
    checkpoint_step: int | None, checkpoint_metadata: dict[str, Any] | None,
    gpu: int,
) -> dict[str, Any]:
    """Profile one frozen lane without retaining generated images in Git."""
    if model is not None and spec is None:
        raise RuntimeError("model distribution evaluation requires a lane spec")
    lane_id = "input" if model is None else spec.id
    freeze, freeze_git_commit = committed_freeze_identity(
        freeze_receipt, lane_id=lane_id,
    )
    destination = Path(destination).resolve()
    if destination.exists():
        raise RuntimeError(f"distribution receipt already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    count_per_domain = int(protocol["evaluation"]["terminal_discovery_per_domain"])
    if count_per_domain != 80:
        raise RuntimeError("post-freeze distribution evaluation requires discovery80")
    selected = select_discovery(rows, count_per_domain)
    domains = sorted({str(row["domain"]) for row in selected})
    if len(domains) != 6:
        raise RuntimeError("post-freeze distribution evaluation requires six domains")
    replicates = 1 if model is None or spec.family != "unsb" else int(
        protocol["evaluation"]["terminal_replicates"]
    )
    if model is not None and (
        int(checkpoint_step or -1) != int(protocol["training"]["target_updates"])
        or checkpoint_metadata is None
        or checkpoint_metadata.get("paired_controller_access") is not False
        or checkpoint_metadata.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("distribution evaluation requires a safe fixed e200 checkpoint")
    checkpoint_path = None if checkpoint is None else Path(checkpoint).resolve()
    checkpoint_hash_before = (
        None if checkpoint_path is None else file_sha256(checkpoint_path)
    )
    device = (
        torch.device(f"cuda:{int(gpu)}" if torch.cuda.is_available() else "cpu")
        if model is None else torch.device(model.device)
    )
    saved_rng = capture_rng()
    modes = {} if model is None else {
        name: getattr(model, "net" + name).training for name in model.model_names
    }
    crn_rows = []
    try:
        cleanfid_package, fid_module = _cleanfid_runtime()
        feature_model = fid_module.build_feature_extractor(
            MODE, device=device, use_dataparallel=False,
        )
        feature_model_hash = _model_state_sha256(feature_model)
        if model is not None:
            model.eval()
        with tempfile.TemporaryDirectory(
            prefix=f"distribution_{lane_id}_", dir=destination.parent,
        ) as temporary:
            root = Path(temporary)
            target_roots = {domain: root / "target" / domain for domain in domains}
            prediction_roots = [
                {domain: root / f"replicate_{replicate:02d}" / domain for domain in domains}
                for replicate in range(replicates)
            ]
            for row in selected:
                domain = str(row["domain"])
                filename = _filename(row)
                source = read_image(Path(data_root) / row["input_relpath"]).to(device)
                target = read_image(Path(data_root) / row["target_relpath"]).to(device)
                _save_png(target_roots[domain] / filename, target)
                for replicate in range(replicates):
                    if model is None:
                        prediction = source
                        crn_sha256 = None
                    else:
                        bundle = build_rollout_bundle(
                            protocol_hash=evaluation_bundle_fingerprint(protocol),
                            domain=domain, stem=str(row["stem"]), replicate=replicate,
                            latent_dim=4 * int(getattr(model.opt, "ngf", 64)),
                            height=128, width=128, num_timesteps=5,
                        )
                        prediction = _prediction(
                            model, spec, source, bundle,
                            nfe=5 if spec.family == "unsb" else 1,
                        ).clamp(-1.0, 1.0)
                        crn_sha256 = bundle_hash(bundle)
                    _save_png(prediction_roots[replicate][domain] / filename, prediction)
                    crn_rows.append({
                        "domain": domain, "stem": str(row["stem"]),
                        "order": int(row["order"]), "replicate": replicate,
                        "crn_bundle_sha256": crn_sha256,
                    })
            target_features = _extract_features(
                roots=target_roots, fid_module=fid_module,
                feature_model=feature_model, device=device,
            )
            prediction_features = [
                _extract_features(
                    roots=roots, fid_module=fid_module,
                    feature_model=feature_model, device=device,
                )
                for roots in prediction_roots
            ]
            scores = summarize_feature_sets(
                lane_id=lane_id, target=target_features,
                predictions=prediction_features, fid_module=fid_module,
            )
    finally:
        for name, was_training in modes.items():
            getattr(model, "net" + name).train(was_training)
        restore_rng(saved_rng)

    checkpoint_hash_after = (
        None if checkpoint_path is None else file_sha256(checkpoint_path)
    )
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("distribution evaluation changed the source checkpoint")
    result = {
        "schema": SCHEMA,
        "status": "PASS_POST_FREEZE_DISCOVERY80_DISTRIBUTION_EVALUATION",
        "lane_id": lane_id,
        "lane": None if model is None else spec.to_dict(),
        "primary_epoch": 200,
        "primary_nfe": 0 if model is None else (5 if spec.family == "unsb" else 1),
        "count_per_domain": count_per_domain,
        "domain_count": len(domains),
        "replicate_count": replicates,
        "protocol_fingerprint": protocol_fingerprint(),
        "evaluation_bundle_fingerprint": evaluation_bundle_fingerprint(protocol),
        "evaluation_input_sha256": evaluation_input_hash(
            selected, evaluation_bundle_fingerprint(protocol),
        ),
        "freeze_receipt": str(Path(freeze_receipt).resolve()),
        "freeze_receipt_sha256": file_sha256(Path(freeze_receipt).resolve()),
        "freeze_receipt_object_sha256": object_sha256(freeze),
        "freeze_git_commit": freeze_git_commit,
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash_before,
        "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after,
        "clean_fid": {
            "package": "clean-fid",
            "version": importlib.metadata.version("clean-fid"),
            "module_file": str(Path(cleanfid_package.__file__).resolve()),
            "module_sha256": file_sha256(Path(fid_module.__file__).resolve()),
            "mode": MODE,
            "feature_model": FEATURE_MODEL,
            "feature_model_state_sha256": feature_model_hash,
            "feature_batch_size": FEATURE_BATCH_SIZE,
            "kid_num_subsets": KID_NUM_SUBSETS,
            "kid_max_subset_size": KID_MAX_SUBSET_SIZE,
        },
        "metrics": scores,
        "crn_rows_sha256": object_sha256(crn_rows),
        "image_encoding": "RGB uint8 PNG after clamp and nearest-integer quantization",
        "fid_interpretation": "supplementary pooled score; only 480 discovery images",
        "kid_interpretation": "primary macro-domain distribution score",
        "environment": environment_record(),
        "generated_images_retained": False,
        "target_path_read_for_post_freeze_evaluation": True,
        "metric_values_used_for_training_or_scheduling": False,
        "best_checkpoint_selection": False,
        "confirmation_authorized": False,
        "confirmation20_opened": False,
    }
    write_json(destination, result)
    return result
