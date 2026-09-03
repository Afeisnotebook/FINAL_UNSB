import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "configs" / "PAPER_DCLGAN_NEGCUT_SOURCE_GATE.json"


def _gate() -> dict:
    return json.loads(GATE.read_text(encoding="utf-8"))


def test_dclgan_is_source_locked_but_not_training_authorized() -> None:
    gate = _gate()
    assert gate["selected_next_external_engineering_target"] == "dclgan"
    assert gate["training_authorized"] is False
    assert gate["host_assigned"] is None
    assert gate["dclgan"]["source"]["authority"] == "author_official"
    assert len(gate["dclgan"]["source"]["commit"]) == 40
    assert gate["dclgan"]["cpu_smoke"]["status"] == "PASS"


def test_controlled_exposure_is_explicit_and_frozen() -> None:
    protocol = _gate()["dclgan"]["controlled_paper_adaptation"]
    assert protocol["manifest_sha256"] == (
        "02c01df580b882763fb0ff28dbdeac4b3729deb8bb772005f26f3e7bc2e36744"
    )
    assert protocol["training_identities_per_side"] == 8553
    assert protocol["data_epochs"] == 200
    assert protocol["updates"] == 1710600
    assert protocol["seed"] == 2026
    assert protocol["batch_size"] == 1
    assert protocol["resolution"] == 128
    assert protocol["schedule"] == "100_constant_plus_100_linear_decay"
    assert len(protocol["required_disclosure"]) == 3
    defaults = _gate()["dclgan"]["official_source_defaults"]
    assert defaults["generator_discriminator_learning_rate"] == 2e-4
    assert defaults["generator_discriminator_adam_betas"] == [0.5, 0.999]
    assert defaults["feature_optimizer_learning_rate"] == 1e-3
    assert defaults["feature_optimizer_adam_betas"] == [0.9, 0.999]


def test_resume_and_confirmation_gates_cannot_be_omitted() -> None:
    remaining = set(_gate()["dclgan"]["remaining_gate"])
    assert "full_state_GA_GB_F1_F2_DA_DB_optimizers_schedulers_sampler_and_all_rng" in remaining
    assert "continuous_1000_vs_500_plus_resume_exact_test" in remaining
    assert "confirmation20_access_rejection" in remaining


def test_negcut_defer_is_not_a_scientific_failure() -> None:
    negcut = _gate()["negcut"]
    assert negcut["source"]["authority"] == "author_official"
    assert negcut["status"] == "deferred_engineering_not_mechanism_falsified"
    assert any("device" in value for value in negcut["defer_reasons"])
    assert _gate()["scientific_boundaries"]["deferred_means_falsified"] is False


def test_gate_contains_no_secret_or_metric_control() -> None:
    raw = GATE.read_text(encoding="utf-8").lower()
    assert "password" not in raw
    boundaries = _gate()["scientific_boundaries"]
    assert boundaries["performance_values_read"] is False
    assert boundaries["paired_metric_control"] is False
    assert boundaries["confirmation20_opened"] is False
    assert boundaries["new_gpu_training_started"] is False
