from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_clean_server_installs_and_preflights_required_lpips_guard():
    requirements = (ROOT / "environment" / "requirements.txt").read_text(
        encoding="utf-8"
    )
    assert "lpips==0.1.4" in requirements.splitlines()
    for relative in ("scripts/bootstrap_server.sh", "scripts/server_preflight.sh"):
        script = (ROOT / relative).read_text(encoding="utf-8")
        assert "import lpips" in script
        assert 'lpips.LPIPS(net="alex")' in script
