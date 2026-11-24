"""Static figures for docs/ (matplotlib, light surface, colour-blind-safe slots)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"        # v2 (categorical slot 1)
ORANGE = "#eb6834"      # v1 (categorical slot 2)
HOLDOUT_SHADE = "#f0efec"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def market_figure(market_csv: Path, vxn: pd.Series, holdout_start: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mk = pd.read_csv(market_csv, index_col=0, parse_dates=True)
    mk = mk[np.isfinite(mk["M"])]
    v = vxn.reindex(mk.index)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, dpi=200,
                                 gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
    fig.patch.set_facecolor(SURFACE)
    for ax in (a1, a2):
        _style(ax)
        ax.axvspan(pd.Timestamp(holdout_start), mk.index[-1], color=HOLDOUT_SHADE, zorder=0)
    a1.fill_between(mk.index, mk["M_lo"], mk["M_hi"], color=BLUE, alpha=0.18, linewidth=0)
    a1.plot(mk.index, mk["M_legacy"], color=ORANGE, linewidth=1.0, alpha=0.9)
    a1.plot(mk.index, mk["M"], color=BLUE, linewidth=1.4)
    a1.axhline(2.0, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    a1.text(mk.index[0], 2.08, "alert line M = 2", color=INK2, fontsize=8, va="bottom")
    last = mk.index[-1]
    a1.annotate("v2 market fear M (90% parameter band)", (mk.index[int(len(mk) * 0.02)], -2.7),
                color=BLUE, fontsize=9)
    a1.annotate("v1 expanding z-score", (mk.index[int(len(mk) * 0.02)], -3.3), color=ORANGE, fontsize=9)
    a1.set_ylabel("standard-normal score", color=INK2, fontsize=9)
    a1.set_ylim(-3.6, max(5.0, float(np.nanmax(mk[["M", "M_legacy", "M_hi"]].to_numpy())) + 0.3))
    a1.set_title("AI-stock market fear, causal daily readings", loc="left", color=INK, fontsize=12)
    a1.text(pd.Timestamp(holdout_start), a1.get_ylim()[1], "  2026 holdout (frozen spec, run once)",
            color=INK2, fontsize=8, va="top")
    a2.plot(v.index, v.to_numpy(), color=INK2, linewidth=1.1)
    a2.set_ylabel("VXN (validation only)", color=INK2, fontsize=9)
    fig.text(0.01, 0.005, "Sources: Yahoo Finance bars (adj close, close, low) for 134 AI stocks; "
             "CBOE VXN via FRED (VXNCLS). VXN never enters the signal.", color=INK2, fontsize=7)
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def reliability_figure(results_json: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads(Path(results_json).read_text())
    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for key, color, label in (("tail_forecast", BLUE, "v2"), ("tail_forecast_v1_published", ORANGE, "v1 published")):
        rows = res[key]["reliability"]
        f = np.array([r["mean_forecast"] for r in rows])
        o = np.array([r["observed"] for r in rows])
        n = np.array([r["n"] for r in rows])
        ax.plot(f, o, color=color, linewidth=1.4, marker="o", markersize=5, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        for fi, oi, ni in zip(f, o, n):
            if color == BLUE:
                ax.annotate(f"{ni:,}", (fi, oi), textcoords="offset points", xytext=(6, -10),
                            fontsize=7, color=INK2)
    lim = 0.3
    ax.plot([1e-3, lim], [1e-3, lim], color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, lim)
    ax.set_ylim(1e-3, lim)
    ax.set_xlabel("forecast P(next low >= 3 sigma below close)", color=INK2, fontsize=9)
    ax.set_ylabel("observed frequency, 2026", color=INK2, fontsize=9)
    ax.set_title("Tail forecast calibration, 2026 holdout", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.0, "Points: forecast bins (stock-days per bin for v2). Diagonal = perfect calibration.",
             color=INK2, fontsize=7)
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
