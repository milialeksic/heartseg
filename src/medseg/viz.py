"""Static figures for the README: case overlays and per-case metric charts.

Colors: categorical slots 1 and 2 of the reference palette (validated adjacent pair);
text in neutral ink, never in series colors; recessive grid.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SERIES = ["#2a78d6", "#eb6834"]  # slot 1 blue, slot 2 orange
GT_COLOR, PRED_COLOR = SERIES
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3de"
CONNECTOR = "#b9b8b2"


def _centroid(mask: np.ndarray) -> tuple[int, ...]:
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return tuple(s // 2 for s in mask.shape)
    return tuple(int(v) for v in idx.mean(axis=0).round())


def _window(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def save_overlay(image, gt, pred, spacing, path, title: str) -> None:
    """Axial/coronal/sagittal slices through the GT centroid, plus an axial projection
    of both masks over all slices (reveals stray components far from the atrium)."""
    img = _window(image)
    c = _centroid(gt if gt.any() else pred)
    sx, sy, sz = spacing
    # Neutral plane names: the Task02 Heart headers are nominally RAS but do not match the
    # true anatomy (the axis-0/1 plane shows sagittal anatomy), so anatomical labels would mislead.
    views = [
        ("Plane axis 0-1", img[:, :, c[2]], gt[:, :, c[2]], pred[:, :, c[2]], sy / sx),
        ("Plane axis 0-2", img[:, c[1], :], gt[:, c[1], :], pred[:, c[1], :], sz / sx),
        ("Plane axis 1-2", img[c[0], :, :], gt[c[0], :, :], pred[c[0], :, :], sz / sy),
        ("Projection along axis 2", img[:, :, c[2]], gt.any(2), pred.any(2), sy / sx),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), facecolor=SURFACE)
    for ax, (name, im, g, p, aspect) in zip(axes, views, strict=True):
        ax.imshow(im.T, cmap="gray", origin="lower", aspect=aspect, interpolation="nearest")
        if g.any():
            ax.contour(g.T, levels=[0.5], colors=GT_COLOR, linewidths=1.6)
        if p.any():
            ax.contour(p.T, levels=[0.5], colors=PRED_COLOR, linewidths=1.6)
        ax.set_title(name, color=TEXT_2, fontsize=10)
        ax.axis("off")
    handles = [
        Line2D([0], [0], color=GT_COLOR, lw=2, label="Ground truth"),
        Line2D([0], [0], color=PRED_COLOR, lw=2, label="Prediction"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, labelcolor=TEXT_2)
    fig.suptitle(title, color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def per_case_chart(
    frames: list, labels: list[str], metric: str, xlabel: str, path: Path, log: bool = False
) -> None:
    """Horizontal dot chart, one row per case, sorted by the first variant.
    With two variants the dots are joined (paired comparison)."""
    base = frames[0].copy()
    base["row"] = base["case_id"] + "  (fold " + base["fold"].astype(str) + ")"
    base = base[np.isfinite(base[metric].astype(float))].sort_values(metric)
    order = list(base["case_id"])
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8, 0.34 * len(order) + 1.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    vals = []
    for df in frames:
        s = df.set_index("case_id")[metric].astype(float)
        vals.append(np.array([s.get(cid, np.nan) for cid in order]))
    if len(vals) == 2:
        for yi, a, b in zip(y, vals[0], vals[1], strict=True):
            if np.isfinite(a) and np.isfinite(b):
                ax.plot([a, b], [yi, yi], color=CONNECTOR, lw=2, zorder=1)
    for i, (v, label) in enumerate(zip(vals, labels, strict=True)):
        ax.scatter(
            v, y, s=46, color=SERIES[i], edgecolor=SURFACE, linewidth=2, zorder=2 + i, label=label
        )

    ax.set_yticks(y, base["row"], color=TEXT_2, fontsize=9)
    ax.set_xlabel(xlabel, color=TEXT_2)
    if log:
        ax.set_xscale("log")
    ax.tick_params(colors=TEXT_2, length=0)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if len(frames) > 1:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(frames),
            frameon=False,
            labelcolor=TEXT_2,
        )
    dropped = len(frames[0]) - len(base)
    if dropped:
        ax.text(
            1,
            -0.12,
            f"{dropped} case(s) with non-finite value omitted",
            transform=ax.transAxes,
            ha="right",
            color=TEXT_2,
            fontsize=8,
        )
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# --- Previews for the web UI (prediction only, no ground truth) ---------------------------
# Viewer style: images on a dark background whatever the page theme, as in clinical viewers.
VIEW_BG = "#000000"
VIEW_TEXT = "#c3c2b7"


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=VIEW_BG, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return buf.getvalue()


def _draw_prediction(ax, im, m, aspect: float, title: str | None = None, overlay=True) -> None:
    ax.set_facecolor(VIEW_BG)
    ax.imshow(im.T, cmap="gray", origin="lower", aspect=aspect, interpolation="nearest")
    if overlay and m.any():
        fill = np.ma.masked_where(~m.T, m.T)
        ax.imshow(
            fill,
            cmap=ListedColormap([PRED_COLOR]),
            alpha=0.28,
            origin="lower",
            aspect=aspect,
            interpolation="nearest",
        )
        ax.contour(m.T, levels=[0.5], colors=PRED_COLOR, linewidths=1.6)
    if title:
        ax.set_title(title, color=VIEW_TEXT, fontsize=10)
    ax.axis("off")


def render_overview(image: np.ndarray, mask: np.ndarray, spacing) -> bytes:
    """PNG: three planes through the predicted atrium plus a projection along axis 2."""
    img = _window(image)
    m = mask.astype(bool)
    c = _centroid(m)
    sx, sy, sz = spacing
    views = [
        ("Plane axis 0-1", img[:, :, c[2]], m[:, :, c[2]], sy / sx),
        ("Plane axis 0-2", img[:, c[1], :], m[:, c[1], :], sz / sx),
        ("Plane axis 1-2", img[c[0], :, :], m[c[0], :, :], sz / sy),
        ("Projection along axis 2", img[:, :, c[2]], m.any(2), sy / sx),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), facecolor=VIEW_BG)
    for ax, (name, im, mm, aspect) in zip(axes, views, strict=True):
        _draw_prediction(ax, im, mm, aspect, name)
    return _png(fig)


def render_slices(
    image: np.ndarray, mask: np.ndarray, spacing, n: int = 21, margin: int = 3
) -> list[tuple[int, bytes, bytes]]:
    """Up to n slices along axis 2 spread over the predicted atrium.

    Returns (slice index, PNG with segmentation, PNG without segmentation)."""
    img = _window(image)
    m = mask.astype(bool)
    last = image.shape[2] - 1
    zs = np.where(m.any(axis=(0, 1)))[0]
    if len(zs) == 0:
        lo, hi = 0, last
    else:
        lo, hi = max(int(zs.min()) - margin, 0), min(int(zs.max()) + margin, last)
    sx, sy, _ = spacing
    out = []
    for k in np.unique(np.linspace(lo, hi, n).round().astype(int)):
        pngs = []
        for overlay in (True, False):
            fig = plt.figure(figsize=(5, 5), facecolor=VIEW_BG)
            ax = fig.add_axes([0, 0, 1, 1])
            _draw_prediction(ax, img[:, :, k], m[:, :, k], sy / sx, overlay=overlay)
            pngs.append(_png(fig))
        out.append((int(k), pngs[0], pngs[1]))
    return out
