"""Source lineage and historical data-epoch evidence reconstruction."""

from __future__ import annotations

from pathlib import Path

from .protocol import ROOT, file_sha256, git_commit, load_protocol, protocol_fingerprint
from .runtime import write_json


HISTORICAL_DT_HASHES = {
    "__init__.py": "7185c8210638009e1f98e38f41a44fc26ea5ba21a2c4d5462640c82f0a35fbcc",
    "dtcovmatch.py": "beb09f334982d681c340ae6f09104b65e7b33cf2dafde6c8b4a1d8cec5af15bf",
    "model.py": "6060cef7e9fe6e5772494cac7bd8066688513996a482a36b4f135141dbdeb0de",
}
HISTORICAL_DT_SEMANTIC_HASHES = {
    "__init__.py": "32638f8652f1a3a9337d67d9e79c35e87ee236c95dfb6a3f181cded10f58b02a",
    "dtcovmatch.py": "08e268a4f73e8dc2c42ccf9d6106892f70a7ab6ec96e0324677c754089f855f5",
    "model.py": "8a44fa0b0b1d70291b78455e424c5863821d4d5572f516c8a8596ebdbc25cc0d",
}


LINEAGE_FILENAMES = {
    "dt": "DT_LINEAGE.json",
    "hj": "HJ_LINEAGE.json",
    "hnek": "HNEK_LINEAGE.json",
}


LATER_MECHANISM_OBJECTS = {
    "PCOA_NPOOA": {
        "unsb_object": "joint native/correction generator update geometry",
        "tested_operator": "Adam-metric projection and norm-preserving revision",
        "evidence_boundary": "short_horizon_negative_current_implementation",
    },
    "LBST": {
        "unsb_object": "endogenous bridge rollout distribution and its moving endpoint teacher",
        "tested_operator": "fixed one-data-epoch-half-life lagged EMA rollout teacher",
        "evidence_boundary": "short_horizon_negative_current_implementation",
    },
    "PTQ": {
        "unsb_object": "bridge-time sampling measure over the physical horizon",
        "tested_operator": "exact fixed physical-interval mass schedule",
        "evidence_boundary": "reversal_observed_short_protocol",
    },
    "DCUM_MACRO_MARGINAL": {
        "unsb_object": "empirical unpaired A/B domain marginals",
        "tested_operator": "domain-conditional or macro-balanced endpoint sampling",
        "evidence_boundary": "reversal_observed_short_protocol",
    },
    "AEB_BCAVP": {
        "unsb_object": "latent endpoint law and latent/time gradient estimator variance",
        "tested_operator": "antithetic endpoint averaging or paired latent control variate",
        "evidence_boundary": "short_horizon_negative_current_implementation",
    },
    "TA_MINIMAL": {
        "unsb_object": "explicit generator time coordinate",
        "tested_operator": "direct restored time-conditioned forward",
        "evidence_boundary": "long_horizon_negative_current_implementation",
    },
    "KCK_PATH_CONSISTENCY": {
        "unsb_object": "direct-versus-composed bridge transition consistency",
        "tested_operator": "path-consistency penalty on the restored-time model",
        "evidence_boundary": "short_horizon_negative_current_implementation",
    },
}


def _hashes(paths: list[Path]) -> dict:
    return {
        path.relative_to(ROOT).as_posix(): file_sha256(path)
        for path in paths if path.is_file()
    }


def build_lineage(manifest_path: Path) -> dict:
    protocol = load_protocol()
    dt_paths = sorted((ROOT / "src/models/dtcov").glob("*.py")) + [ROOT / "src/models/dtcov_model.py"]
    hj_paths = sorted((ROOT / "src/models/hj").glob("*.py")) + [ROOT / "src/models/hj_model.py"]
    hnek_paths = sorted((ROOT / "src/models/hnek").glob("*.py")) + [ROOT / "src/models/hnek_search_model.py"]
    return {
        "schema": "final-unsb-local-route1-lineage-v1",
        "git_commit": git_commit(),
        "protocol_fingerprint": protocol_fingerprint(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "baseline": {
            "source_hashes": _hashes([ROOT / "src/models/sb_model.py"]),
            "clean_profile": protocol["common"],
            "scientific_clock": "data_epoch = 150 optimizer updates on small25",
        },
        "probes": {
            "dt": {
                "historical_authoritative_source_hashes": HISTORICAL_DT_HASHES,
                "historical_semantic_source_hashes": HISTORICAL_DT_SEMANTIC_HASHES,
                "local_port_hashes": _hashes(dt_paths),
                "intentional_port_delta": [
                    "repository-relative imports replace the old foundation sys.path injection",
                    "shared EpochDiagnostics import replaces an unused external diagnostics dependency",
                    "DT-CovMatch equations, schedule, teacher, RNG preservation and state are unchanged",
                ],
                "physical_protocol": "plain e1-e20; active-age schedule e21-e45; lambda naturally zero thereafter; continue reached state to e200",
                "historical_clean_fact": "SEARCH-001 full100: +0.566439 dB at 2000 updates, then -0.806110/-1.519180 dB at 3000/4000",
            },
            "hj": {
                "historical_core_hash": "503a4e092470cd7355230495f10f094402fb7efb570ca9a632bdf75f4ab64e0a",
                "local_source_hashes": _hashes(hj_paths),
                "intentional_port_delta": [
                    "inactive HJ now preserves canonical CPU-then-device RNG draw exactly",
                    "step-relative SEARCH-001 activation is disabled; physical epoch 5 is authoritative",
                ],
                "physical_protocol": "plain-identical e1-e4; Layer-0 true/joint/central-consensus from the first batch of e5 through e200",
                "historical_trajectory_delta_db": {
                    "e100": -0.3524, "e125": 0.5057, "e150": 2.4497,
                    "e175": 3.6307, "e200": 1.3324
                },
            },
            "hnek": {
                "local_source_hashes": _hashes(hnek_paths),
                "configuration": protocol["anchor_probes"][2]["method"],
                "physical_protocol": "gamma=.25/residual/physical/all is active for e1-e200",
                "historical_evidence": {"e50_delta_db": 2.6173, "e200_delta_db": 0.7883720592327812, "positive_domains_e200": 4},
            },
        },
        "historical_to_clean_semantics": {
            "shared": ["UNSB G/F/D/E object family", "T=5 bridge", "tau=.01", "GAN/SB/NCE weights all 1", "constant lr=1e-4 for 200 epochs"],
            "must_not_be_conflated": [
                "historical batch16/train160 trajectories versus local batch1/small25 proxy",
                "data epochs versus raw optimizer updates",
                "old nondeterministic implementation versus deterministic port",
                "paired evaluation labels versus target-blind training observables",
            ],
        },
        "unsb_object_graph": {
            "native_objects": ["endpoint generator G(x_t,t,z)", "bridge rollout distribution", "GAN discriminator D", "SB critic E", "PatchNCE feature sampler F"],
            "dt": ["latent endpoint direction dispersion", "domain x physical-time normalization", "frozen first-use teacher"],
            "hj": ["Layer-0 PatchNCE feature finite-difference risk", "one-sided structure projection"],
            "hnek": ["remaining physical horizon", "residual bridge coordinate", "entropy/endpoint conditioning"],
            "later_pool": ["rollout teacher speed", "physical-time sampling measure", "domain sampling measure", "antithetic latent variance", "native/correction gradient game geometry"],
            "later_mechanisms": LATER_MECHANISM_OBJECTS,
        },
        "claim_boundary": "lineage is provenance, not evidence that any probe is a sustained algorithm",
        "confirmation20_opened": False,
    }


def split_lineage_documents(payload: dict) -> dict[str, dict]:
    """Create explicit per-probe and object-map deliverables from one lineage."""
    common = {
        "lineage_git_commit": payload["git_commit"],
        "protocol_fingerprint": payload["protocol_fingerprint"],
        "manifest_sha256": payload["manifest_sha256"],
        "baseline": payload["baseline"],
        "historical_to_clean_semantics": payload["historical_to_clean_semantics"],
        "confirmation20_opened": False,
    }
    documents: dict[str, dict] = {}
    for probe, filename in LINEAGE_FILENAMES.items():
        documents[filename] = {
            "schema": "final-unsb-route1-probe-lineage-v1",
            "probe": probe,
            **common,
            "lineage": payload["probes"][probe],
            "unsb_objects": payload["unsb_object_graph"][probe],
            "claim_boundary": (
                "This proves source and protocol provenance only; it does not prove "
                "that the probe is a sustained algorithm or a route-1 candidate."
            ),
        }
    documents["MECHANISM_OBJECT_MAP.json"] = {
        "schema": "final-unsb-route1-mechanism-object-map-v1",
        **common,
        "native_objects": payload["unsb_object_graph"]["native_objects"],
        "anchor_probes": {
            probe: payload["unsb_object_graph"][probe]
            for probe in ("dt", "hj", "hnek")
        },
        "later_mechanisms": payload["unsb_object_graph"]["later_mechanisms"],
        "status_vocabulary": (
            "Every evidence_boundary closes only the named implementation/protocol "
            "unless a separate mathematical invariant is explicitly falsified."
        ),
        "claim_boundary": "Object mapping is a derivation aid, not candidate preselection.",
    }
    return documents


def write_lineage(output_root: Path, manifest_path: Path) -> Path:
    path = output_root / "lineage" / "LINEAGE.json"
    payload = build_lineage(manifest_path)
    write_json(path, payload)
    for filename, document in split_lineage_documents(payload).items():
        write_json(path.parent / filename, document)
    return path
