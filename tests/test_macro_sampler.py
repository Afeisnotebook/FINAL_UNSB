from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from data.unaligned_dataset import UnalignedDataset


def _make_view(root: Path):
    for side in ("A", "B"):
        folder = root / f"train{side}"
        folder.mkdir(parents=True)
        for domain in ("d0", "d1", "d2"):
            for stem in ("s0", "s1", "s2"):
                Image.new("RGB", (8, 8), (10, 20, 30)).save(
                    folder / f"{domain}__{stem}.png"
                )


def test_macro_domains_are_independent_and_same_stem_is_forbidden(tmp_path: Path):
    _make_view(tmp_path)
    opt = SimpleNamespace(
        dataroot=str(tmp_path), phase="train", max_dataset_size=float("inf"),
        macro_marginal=True, isTrain=True,
    )
    dataset = UnalignedDataset(opt)
    cells = {(a, b): 0 for a in dataset._macro_domains for b in dataset._macro_domains}
    for _ in range(9000):
        a_path, b_path = dataset._sample_macro_pair()
        a_domain, a_stem = dataset._domain_and_stem(a_path)
        b_domain, b_stem = dataset._domain_and_stem(b_path)
        cells[(a_domain, b_domain)] += 1
        assert a_domain != b_domain or a_stem != b_stem
    expected = 9000 / 9
    assert all(abs(count - expected) < 0.15 * expected for count in cells.values())
