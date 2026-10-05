import pytest
import torch
from omegaconf import OmegaConf

from medseg.models import build_model
from medseg.utils import set_seed


def _cfg(name):
    return OmegaConf.create({"model": {"name": name, "out_channels": 2}})


@pytest.mark.parametrize("name", ["unet", "segresnet"])
def test_output_shape(name):
    model = build_model(_cfg(name)).eval()
    x = torch.zeros(1, 1, 32, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 2, 32, 32, 32)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model(_cfg("nope"))


def test_set_seed_is_reproducible():
    set_seed(7)
    a = torch.rand(5)
    set_seed(7)
    b = torch.rand(5)
    assert torch.equal(a, b)
