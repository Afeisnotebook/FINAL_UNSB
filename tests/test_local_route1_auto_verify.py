import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = ROOT / "operations"
if str(OPERATIONS) not in sys.path:
    sys.path.insert(0, str(OPERATIONS))

import local_route1_auto_verify as auto  # noqa: E402


def accepted_payload(lane: str, epoch: int) -> dict:
    return {
        "schema": "final-unsb-route1-milestone-verification-v1",
        "status": "ACCEPTED_MILESTONE",
        "identity": {"probe_id": lane, "data_epoch": epoch, "updates": epoch * 150},
        "integrity": {
            "checkpoint_file_hash_matches_sidecar": True,
            "scientific_state_hash_matches_sidecar": True,
            "metric_protocol_matches": True,
            "evaluation_bundle_matches_frozen_crn": True,
            "paired_metric_used_for_training_control": False,
            "confirmation20_opened": False,
        },
    }


def materialize_artifacts(root: Path, lane: str, epoch: int) -> None:
    for path in auto.artifact_paths(root, lane, epoch):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")


def test_existing_evidence_is_idempotently_accepted(tmp_path: Path):
    evidence = tmp_path / "verification.json"
    evidence.write_text(json.dumps(accepted_payload("hj", 200)), encoding="utf-8")
    assert auto.accepted_evidence(evidence, lane="hj", epoch=200)
    payload = accepted_payload("hj", 200)
    payload["integrity"]["confirmation20_opened"] = True
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    assert not auto.accepted_evidence(evidence, lane="hj", epoch=200)


def test_watcher_verifies_multiple_ready_lanes_without_control(tmp_path: Path):
    run_root = tmp_path / "run"
    training_repo = tmp_path / "repo"
    training_repo.mkdir()
    output_dir = run_root / "operations" / "proofs"
    state = run_root / "operations" / "state.json"
    for lane in ("plain", "hnek"):
        materialize_artifacts(run_root, lane, 200)

    calls = []

    def fake_verify(**kwargs):
        calls.append(kwargs)
        return accepted_payload(kwargs["lane"], kwargs["epoch"])

    code = auto.watch(
        run_root=run_root, training_repo=training_repo,
        lanes=("plain", "hnek"), epoch=200, output_dir=output_dir,
        state_path=state, poll_seconds=0, timeout_seconds=1,
        maximum_failures=3, verifier=fake_verify,
    )
    assert code == 0
    assert [call["lane"] for call in calls] == ["plain", "hnek"]
    assert all(call["require_lpips"] for call in calls)
    final = json.loads(state.read_text(encoding="utf-8"))
    assert final["status"] == "COMPLETE"
    assert final["paired_metric_used_for_training_control"] is False
    assert final["confirmation20_opened"] is False


def test_watcher_stops_after_bounded_verification_failures(tmp_path: Path):
    run_root = tmp_path / "run"
    training_repo = tmp_path / "repo"
    training_repo.mkdir()
    materialize_artifacts(run_root, "hj", 200)
    state = run_root / "operations" / "state.json"

    def broken_verify(**_kwargs):
        raise RuntimeError("scientific hash mismatch")

    code = auto.watch(
        run_root=run_root, training_repo=training_repo, lanes=("hj",),
        epoch=200, output_dir=run_root / "proofs", state_path=state,
        poll_seconds=0, timeout_seconds=1, maximum_failures=2,
        verifier=broken_verify,
    )
    assert code == 1
    final = json.loads(state.read_text(encoding="utf-8"))
    assert final["lanes"]["hj"]["status"] == "FAILED"
    assert final["lanes"]["hj"]["consecutive_failures"] == 2
    assert "scientific hash mismatch" in final["lanes"]["hj"]["last_error"]
