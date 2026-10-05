import pandas as pd
import pytest
import torch

from medseg.robustness import apply_perturbation, summarize_robustness


def _img() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, 16, 16, 16, generator=g) * 100 + 10
    x[:, :2] = 0  # some background
    return x


def test_none_is_identity():
    x = _img()
    assert torch.equal(apply_perturbation(x, "none", 0.0), x)


@pytest.mark.parametrize(
    "kind,severity",
    [("noise", 0.1), ("bias", 0.5), ("gamma", 0.3), ("motion", 5.0), ("lowres", 2.0)],
)
def test_shape_preserved_finite_and_changed(kind, severity):
    x = _img()
    y = apply_perturbation(x, kind, severity, seed=1)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert not torch.allclose(y, x)


@pytest.mark.parametrize(
    "kind,severity",
    [("noise", 0.2), ("bias", 0.5), ("gamma", 0.3), ("motion", 10.0), ("lowres", 4.0)],
)
def test_background_stays_zero_and_scanned_area_nonzero(kind, severity):
    # Same nonzero region before and after, so NormalizeIntensityd(nonzero=True) uses the
    # same voxels as for the clean image.
    x = _img()
    y = apply_perturbation(x, kind, severity, seed=1)
    assert torch.equal(y == 0, x == 0)


@pytest.mark.parametrize("kind,severity", [("noise", 0.1), ("bias", 0.5), ("motion", 5.0)])
def test_deterministic_given_seed(kind, severity):
    x = _img()
    a = apply_perturbation(x, kind, severity, seed=3)
    b = apply_perturbation(x, kind, severity, seed=3)
    assert torch.allclose(a, b)


def test_noise_scales_with_severity():
    x = _img()
    small = (apply_perturbation(x, "noise", 0.05, seed=1) - x).std()
    large = (apply_perturbation(x, "noise", 0.2, seed=1) - x).std()
    assert large > 3 * small


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        apply_perturbation(_img(), "blur", 1.0)


def test_summary_paired_delta():
    df = pd.DataFrame(
        {
            "case_id": ["a", "b", "a", "b"],
            "kind": ["none", "none", "noise", "noise"],
            "severity": [0.0, 0.0, 0.1, 0.1],
            "dice": [0.90, 0.80, 0.85, 0.70],
            "hd95_mm": [5.0, 6.0, 7.0, 10.0],
        }
    )
    s = summarize_robustness(df).set_index("kind")
    assert s.loc["none", "delta_dice"] == pytest.approx(0.0)
    assert s.loc["noise", "delta_dice"] == pytest.approx(-0.075)
    assert s.loc["noise", "worst_dice"] == pytest.approx(0.70)
    assert s.loc["noise", "delta_hd95_mm"] == pytest.approx(3.0)
