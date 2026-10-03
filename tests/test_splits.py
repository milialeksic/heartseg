import pytest

from medseg.splits import make_splits

IDS = [f"heart_{i:03d}" for i in range(20)]


def test_no_overlap_within_each_fold():
    s = make_splits(IDS)
    for fold in s["folds"]:
        assert not set(fold["train"]) & set(fold["val"])
        assert not set(fold["train"]) & set(s["test"])
        assert not set(fold["val"]) & set(s["test"])


def test_every_case_used_exactly_once_as_val_or_test():
    s = make_splits(IDS)
    val_all = [c for f in s["folds"] for c in f["val"]]
    assert len(val_all) == len(set(val_all))
    assert sorted(val_all + s["test"]) == sorted(IDS)


def test_deterministic_given_seed():
    assert make_splits(IDS, seed=1) == make_splits(IDS, seed=1)
    assert make_splits(IDS, seed=1) != make_splits(IDS, seed=2)


def test_rejects_duplicates():
    with pytest.raises(ValueError):
        make_splits(IDS + [IDS[0]])