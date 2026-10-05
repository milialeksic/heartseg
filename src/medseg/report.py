"""Build README-ready results from per-case CSVs written by medseg.eval.

Usage (from repo root):
    python -m medseg.report outputs/eval/unet_foldall_val.csv
    python -m medseg.report outputs/eval/unet_foldall_val.csv \
        outputs/eval/unet_foldall_val_lcc.csv --labels "UNet" "UNet + largest component"

Writes to results/ (commit this folder):
    summary.md                      tables to paste into the README
    <csv files>                     copies of the per-case CSVs
    figures/per_case_dice.png       per-case Dice (paired if two CSVs)
    figures/per_case_hd95.png       per-case HD95, log scale
    figures/overlay_<case>.png      overlays of the worst and best cases (if they were saved)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from medseg.eval import summarize
from medseg.viz import per_case_chart

DISTANCE = {"hd95_mm", "assd_mm", "abs_vol_err_ml"}


def _fmt(metric: str, v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    if metric in DISTANCE:
        return f"{v:.1f}"
    if metric == "n_components":
        return f"{v:.2f}"
    return f"{v:.3f}"


def summary_table(df: pd.DataFrame) -> str:
    lines = ["| Metric | Mean ± SD | 95% CI (bootstrap) | Finite cases |", "|---|---|---|---|"]
    for _, r in summarize(df).iterrows():
        m = r["metric"]
        lines.append(
            f"| {m} | {_fmt(m, r['mean'])} ± {_fmt(m, r['std'])} | "
            f"[{_fmt(m, r['ci95_low'])}, {_fmt(m, r['ci95_high'])}] | "
            f"{r['n_finite']}/{r['n_total']} |"
        )
    return "\n".join(lines)


def per_fold_table(df: pd.DataFrame) -> str:
    g = df.groupby("fold")[["dice", "hd95_mm", "nsd", "n_components"]].mean()
    lines = [
        "| Fold | Dice | HD95 (mm) | NSD | Mean components |",
        "|---|---|---|---|---|",
    ]
    for fold, r in g.iterrows():
        lines.append(
            f"| {fold} | {r['dice']:.3f} | {r['hd95_mm']:.1f} | {r['nsd']:.3f} | "
            f"{r['n_components']:.2f} |"
        )
    return "\n".join(lines)


def worst_cases(df: pd.DataFrame, n: int = 5) -> str:
    cols = ["case_id", "fold", "dice", "hd95_mm", "nsd", "n_components", "rel_vol_err"]
    w = df.sort_values("dice").head(n)[[c for c in cols if c in df.columns]]
    lines = [
        "| Case | Fold | Dice | HD95 (mm) | NSD | Components | Rel. volume error |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in w.iterrows():
        lines.append(
            f"| {r['case_id']} | {r['fold']} | {r['dice']:.3f} | {r['hd95_mm']:.1f} | "
            f"{r['nsd']:.3f} | {int(r['n_components'])} | {r['rel_vol_err']:+.1%} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("csvs", nargs="+", help="one or two per-case CSVs from medseg.eval")
    p.add_argument("--labels", nargs="+", help="display names, one per CSV")
    p.add_argument("--out", default="results")
    p.add_argument("--n-overlays", type=int, default=3, help="worst cases to copy overlays for")
    args = p.parse_args()

    if len(args.csvs) > 2:
        raise SystemExit("Pass one or two CSVs (two = paired comparison).")
    paths = [Path(c) for c in args.csvs]
    labels = args.labels or [c.stem for c in paths]
    frames = [pd.read_csv(c) for c in paths]

    out = Path(args.out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for c in paths:
        shutil.copy(c, out / c.name)

    per_case_chart(frames, labels, "dice", "Dice (higher is better)", figs / "per_case_dice.png")
    per_case_chart(
        frames, labels, "hd95_mm", "HD95 in mm, log scale (lower is better)",
        figs / "per_case_hd95.png", log=True,
    )

    # Overlays: worst N by Dice plus the best case, from each variant that saved them.
    base = frames[0].sort_values("dice")
    picks = list(base["case_id"].head(args.n_overlays)) + [base["case_id"].iloc[-1]]
    copied = []
    for c, label in zip(paths, labels, strict=True):
        src_dir = c.with_suffix("") / "overlays"
        for cid in picks:
            src = src_dir / f"{cid}.png"
            if src.exists():
                dst = figs / f"overlay_{cid}_{c.stem}.png"
                shutil.copy(src, dst)
                copied.append((label, cid, dst.name))

    md = ["# Results", ""]
    for df, label in zip(frames, labels, strict=True):
        md += [f"## {label}", "", f"n = {len(df)} cases", "", summary_table(df), ""]
        md += ["### Per fold", "", per_fold_table(df), ""]
        md += ["### Worst cases by Dice", "", worst_cases(df), ""]
    md += [
        "## Figures",
        "",
        "![Per-case Dice](figures/per_case_dice.png)",
        "",
        "![Per-case HD95](figures/per_case_hd95.png)",
        "",
    ]
    for label, cid, name in copied:
        md += [f"**{cid}** ({label})", "", f"![{cid}](figures/{name})", ""]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out / 'summary.md'} and figures in {figs}")


if __name__ == "__main__":
    main()