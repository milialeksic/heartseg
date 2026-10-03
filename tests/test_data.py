import json
from pathlib import Path
from medseg.data import list_cases, load_or_create_splits


def _fake_msd(tmp_path, n=10):
    meta = {
        "training": [
            {"image": f"./imagesTr/la_{i:03d}.nii.gz", "label": f"./labelsTr/la_{i:03d}.nii.gz"}
            for i in range(n)
        ]
    }
    (tmp_path / "dataset.json").write_text(json.dumps(meta))
    return tmp_path


def test_list_cases_parses_ids(tmp_path):
    root = _fake_msd(tmp_path)
    cases = list_cases(root)
    assert len(cases) == 10
    assert "la_000" in cases
    
    assert Path(cases["la_000"]["image"]).parts[-2:] == ("imagesTr", "la_000.nii.gz")


def test_splits_are_created_once_and_reused(tmp_path):
    root = _fake_msd(tmp_path)
    sf = tmp_path / "splits.json"
    a = load_or_create_splits(root, sf, n_test=2, n_folds=4, seed=42)
    assert sf.exists()
    b = load_or_create_splits(root, sf, n_test=2, n_folds=4, seed=999)  # seed ignored
    assert a == b