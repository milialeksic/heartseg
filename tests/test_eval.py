import numpy as np
import pandas as pd
import pytest
import torch

from medseg.eval import (
    bootstrap_ci,
    case_metrics,
    ensemble_probs,
    keep_largest_component,
    summarize,
)

SPACING = (1.0, 1.0, 1.0)


def _onehot(fg: torch.Tensor) -> torch.Tensor:
    return torch.stack([1 - fg, fg]).unsqueeze(0)


def _cube_mask(start: int, size: int = 8, n: int = 24) -> torch.Tensor:
    fg = torch.zeros(n, n, n)
    fg[start : start + size, 4 : 4 + size, 4 : 4 + size] = 1
    return fg


def _onehot_cube(start: int, size: int = 8, n: int = 24) -> torch.Tensor:
    """(1, 2, n, n, n) one-hot with a foreground cube starting at x=start."""
    return _onehot(_cube_mask(start, size, n))


def test_perfect_prediction():
    gt = _onehot_cube(4)
    m = case_metrics(gt.clone(), gt, SPACING, nsd_tol_mm=1.0)
    assert m["dice"] == pytest.approx(1.0)
    assert m["iou"] == pytest.approx(1.0)
    assert m["hd95_mm"] == pytest.approx(0.0)
    assert m["assd_mm"] == pytest.approx(0.0)
    assert m["nsd"] == pytest.approx(1.0)
    assert m["abs_vol_err_ml"] == pytest.approx(0.0)
    assert m["n_components"] == 1


def test_shifted_prediction():
    gt = _onehot_cube(4)
    pred = _onehot_cube(6)  # shifted by 2 voxels along x -> overlap 6/8
    m = case_metrics(pred, gt, SPACING, nsd_tol_mm=3.0)
    assert m["dice"] == pytest.approx(0.75)
    assert m["iou"] == pytest.approx(0.6)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["rel_vol_err"] == pytest.approx(0.0)
    assert 0 < m["hd95_mm"] <= 3.0
    assert 0 < m["assd_mm"] <= 2.0
    assert m["nsd"] >= 0.99  # every surface point is within 3 mm


def test_spacing_scales_distances_and_volume():
    gt = _onehot_cube(4)
    pred = _onehot_cube(6)
    a = case_metrics(pred, gt, (1.0, 1.0, 1.0), nsd_tol_mm=3.0)
    b = case_metrics(pred, gt, (2.0, 2.0, 2.0), nsd_tol_mm=3.0)
    assert b["hd95_mm"] == pytest.approx(2 * a["hd95_mm"])
    assert b["gt_volume_ml"] == pytest.approx(8 * a["gt_volume_ml"])


def test_stray_blob_hurts_hd95_and_lcc_removes_it():
    gt = _onehot_cube(4)
    fg = _cube_mask(4)
    fg[20:22, 20:22, 20:22] = 1  # small false positive far away
    pred = _onehot(fg)

    before = case_metrics(pred, gt, SPACING, nsd_tol_mm=2.0)
    assert before["n_components"] == 2

    cleaned = keep_largest_component(pred)
    after = case_metrics(cleaned, gt, SPACING, nsd_tol_mm=2.0)
    assert after["n_components"] == 1
    assert after["dice"] == pytest.approx(1.0)
    assert after["hd95_mm"] < before["hd95_mm"] or before["hd95_mm"] == pytest.approx(0.0)
    assert torch.equal(cleaned, gt)


def test_lcc_is_noop_for_single_component():
    gt = _onehot_cube(4)
    assert torch.equal(keep_largest_component(gt), gt)


def test_ensemble_single_model_equals_softmax():
    logits = torch.randn(1, 2, 4, 4, 4)
    assert torch.allclose(ensemble_probs([logits]), torch.softmax(logits, dim=1))


def test_ensemble_identical_models_unchanged():
    logits = torch.randn(1, 2, 4, 4, 4)
    p = ensemble_probs([logits, logits.clone(), logits.clone()])
    assert torch.allclose(p, torch.softmax(logits, dim=1), atol=1e-6)


def test_ensemble_averages_probabilities():
    # voxel 0: confident foreground (p_fg ~0.98) vs. weak background (p_fg ~0.38) -> mean fg
    fg_conf = torch.tensor([0.0, 4.0]).view(1, 2, 1, 1, 1)
    bg_weak = torch.tensor([0.5, 0.0]).view(1, 2, 1, 1, 1)
    p = ensemble_probs([fg_conf, bg_weak])
    assert p.shape == (1, 2, 1, 1, 1)
    assert torch.allclose(p.sum(dim=1), torch.ones(1, 1, 1, 1))
    assert p[0, 1].item() > 0.5


def test_ensemble_requires_models():
    with pytest.raises(ValueError):
        ensemble_probs([])


def test_bootstrap_ci_contains_mean_and_ignores_nonfinite():
    vals = [0.8, 0.85, 0.9, 0.95, np.inf, np.nan]
    lo, hi = bootstrap_ci(vals, n_boot=500)
    assert lo <= np.mean([0.8, 0.85, 0.9, 0.95]) <= hi


def test_summarize_counts_nonfinite():
    df = pd.DataFrame(
        {"dice": [0.9, 0.8, 0.7], "hd95_mm": [2.0, np.inf, 4.0], "nsd": [0.9, 0.8, 0.95]}
    )
    s = summarize(df).set_index("metric")
    assert s.loc["hd95_mm", "n_finite"] == 2
    assert s.loc["hd95_mm", "n_total"] == 3
    assert s.loc["dice", "mean"] == pytest.approx(0.8)