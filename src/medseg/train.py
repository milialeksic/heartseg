"""Training loop for one CV fold, with MLflow tracking.

Usage (from repo root):
    python -m medseg.train fold=0
    python -m medseg.train fold=0 train.max_epochs=2 model.name=segresnet
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from omegaconf import OmegaConf

from medseg.data import get_loaders
from medseg.models import build_model
from medseg.utils import git_commit, git_is_dirty, set_seed


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


@torch.no_grad()
def validate(model, loader, cfg, device, use_amp) -> float:
    """Mean foreground Dice over the validation loader (sliding-window inference)."""
    n_cls = int(cfg.model.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=n_cls)
    post_label = AsDiscrete(to_onehot=n_cls)
    metric = DiceMetric(include_background=False, reduction="mean")

    model.eval()
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = sliding_window_inference(x, tuple(cfg.data.patch_size), 4, model)
        preds = [post_pred(o) for o in decollate_batch(out)]
        labels = [post_label(t) for t in decollate_batch(y)]
        metric(y_pred=preds, y=labels)
    score = metric.aggregate().item()
    metric.reset()
    return score


def train_fold(cfg, fold: int) -> float:
    set_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.train.amp) and device.type == "cuda"

    train_loader, val_loader = get_loaders(cfg, fold)
    model = build_model(cfg).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg.train.max_epochs)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_name = f"{cfg.model.name}-fold{fold}"
    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(cfg.tracking.experiment)
    best = -1.0
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: str(v) for k, v in _flatten(OmegaConf.to_container(cfg)).items()})
        mlflow.log_param("fold", fold)
        mlflow.set_tag("git_commit", git_commit())
        mlflow.set_tag("git_dirty", str(git_is_dirty()))
        mlflow.set_tag("device", device.type)

        for epoch in range(int(cfg.train.max_epochs)):
            model.train()
            running = 0.0
            for batch in train_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += loss.item()
            scheduler.step()
            train_loss = running / max(len(train_loader), 1)
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            if (epoch + 1) % int(cfg.train.val_interval) == 0:
                dice = validate(model, val_loader, cfg, device, use_amp)
                mlflow.log_metric("val_dice", dice, step=epoch)
                print(f"epoch {epoch + 1}: loss {train_loss:.4f}  val_dice {dice:.4f}")
                if dice > best:
                    best = dice
                    torch.save(model.state_dict(), ckpt_dir / "best.pt")
                    mlflow.log_metric("best_val_dice", best, step=epoch)
            else:
                print(f"epoch {epoch + 1}: loss {train_loss:.4f}")

        mlflow.log_artifact(str(ckpt_dir / "best.pt")) if (ckpt_dir / "best.pt").exists() else None
    return best


def main() -> None:
    base = OmegaConf.load("configs/default.yaml")
    cfg = OmegaConf.merge(base, OmegaConf.from_cli())
    fold = int(cfg.get("fold", 0))
    best = train_fold(cfg, fold)
    print(f"Best val Dice (fold {fold}): {best:.4f}")


if __name__ == "__main__":
    main()