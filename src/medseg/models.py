"""Model factory. Add new architectures here and select them via cfg.model.name."""

from __future__ import annotations

from monai.networks.nets import UNet, SegResNet


def build_model(cfg):
    name = str(cfg.model.name).lower()
    out_ch = int(cfg.model.out_channels)

    if name == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=out_ch,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="instance",
        )
    if name == "segresnet":
        return SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=out_ch,
            init_filters=16,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.1,
        )
    raise ValueError(f"Unknown model name: {cfg.model.name!r} (use 'unet' or 'segresnet')")