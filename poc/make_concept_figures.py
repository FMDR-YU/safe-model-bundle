#!/usr/bin/env python3
"""
Black-and-white publication figures for the Safe Model Bundle manuscript.
Titles are NOT baked into the images (PLOS: titles belong in the caption);
sans-serif; grayscale only (distinguished by line style / shade / hatch).

The manuscript carries three figures (renumbered after the rework):
  Fig 1  risk decomposition, ingress vs egress (drawn)          -> fig2()
  Fig 2  MIA ROC, leaky vs safe (recomputed from the PoC)       -> fig3_roc_bw()
  Fig 3  three-gate export pipeline (drawn)                     -> fig4()

Outputs to ../results/ : {Fig1,Fig2,Fig3}.png (200 dpi) and matching
grayscale LZW .tif (600 dpi). Deterministic: thread-pinned + fixed seed
-> identical to the canonical run. The retired concept/data figures
(spectrum, DCR, cost) remain defined below for reference but are no
longer emitted by default.
"""
import os
# pin numeric backends before numpy/xgboost import (bit-reproducible recompute)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "savefig.dpi": 200,
})

# grayscale palette
PROC_F = "#e9e9e9"
PLAT_F = "#f3f3f3"
NEU_F  = "#ffffff"
EMPH_F = "#ffffff"
MIT_F  = "#dddddd"
BLACK  = "#000000"
GRAY   = "#666666"

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")


def _add_poc_to_path():
    if HERE not in sys.path:
        sys.path.insert(0, HERE)


def box(ax, cx, cy, w, h, text, fc, fs=9.5, weight="normal", lw=1.4,
        pad=0.018, hatch=None, ec=BLACK):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle=f"round,pad={pad},rounding_size=0.06",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3, hatch=hatch)
    ax.add_patch(p)
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                color=BLACK, weight=weight, zorder=4, linespacing=1.3)


def arrow(ax, p0, p1, lw=1.8, style="-|>", connstyle="arc3,rad=0", ms=12, ls="solid"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        lw=lw, color=BLACK, connectionstyle=connstyle,
                        zorder=2, linestyle=ls)
    ax.add_patch(a)


# ---------------------------------------------------------------- Fig 1
def fig1():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.4); ax.axis("off")
    stages = [
        ("Open\ndownload", "Full records\nleave", 5),
        ("Authenticated\ndownload", "Full records leave\n(credentialed)", 4),
        ("Closed cloud\nTRE", "Only results /\noutputs leave", 2),
        ("Federated\nanalysis", "Only model\nparameters leave", 1),
    ]
    cxs = [1.55, 4.05, 6.55, 9.05]
    bw = 2.15
    for cx, (name, _, _) in zip(cxs, stages):
        box(ax, cx, 4.75, bw, 0.66, name, PROC_F, fs=10, weight="bold")
    for cx, (_, leaves, _) in zip(cxs, stages):
        ax.text(cx, 4.27, leaves, ha="center", va="top",
                fontsize=8.3, color=BLACK, linespacing=1.15)
    arrow(ax, (0.7, 3.55), (10.3, 3.55), lw=1.4, ms=12)
    ax.text(5.5, 3.63, "movement of the data itself decreases",
            ha="center", va="bottom", fontsize=8.6, style="italic", color=GRAY)
    ax.text(0.35, 2.98, "relative\ndata-movement\nscore",
            ha="right", va="center", fontsize=8.2, color=BLACK, linespacing=1.1)
    for cx, (_, _, score) in zip(cxs, stages):
        for i in range(5):
            dx = cx + (i - 2) * 0.235
            filled = i < score
            ax.plot(dx, 3.10, marker="o", markersize=10,
                    markerfacecolor=(BLACK if filled else "white"),
                    markeredgecolor=BLACK, markeredgewidth=1.3, zorder=3)
        ax.text(cx, 2.66, f"{score} / 5", ha="center", va="center",
                fontsize=8.6, color=BLACK, weight="bold")
        arrow(ax, (cx, 2.34), (cx, 1.46), lw=1.3, ms=10)
    box(ax, 5.5, 0.98, 10.3, 0.92,
        "Export / egress interface  —  invariant control point across the spectrum",
        PLAT_F, fs=10, weight="bold")
    ax.text(5.5, 0.18,
            "Every stage funnels outputs through this boundary — where the 2026 UK Biobank breach occurred, "
            "and where the Safe Model Bundle applies.",
            ha="center", va="center", fontsize=8.6, color=BLACK, weight="bold",
            linespacing=1.25)
    return fig


# ---------------------------------------------------------------- Fig 2
def fig2():
    # Left/right symmetric: Risk A (ingress) | Risk B (egress), mirrored about x=5.5.
    # No summary sentence in the image (it belongs in the caption).
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")
    LX, RX = 2.75, 8.25                      # symmetric column centres

    # shared platform box spanning the top centre
    box(ax, 5.5, 5.55, 4.2, 0.8,
        "Closed platform / TRE\nde-identified data held in-environment",
        PLAT_F, fs=9.5, weight="bold")

    # central vertical divider between the two risks
    ax.plot([5.5, 5.5], [0.35, 4.9], ls=(0, (5, 4)), color=GRAY, lw=1.3, zorder=1)

    def column(cx, header, sub, node_label, threat, mit):
        ax.text(cx, 4.78, header, ha="center", va="center",
                fontsize=11, weight="bold", color=BLACK)
        ax.text(cx, 4.40, sub, ha="center", va="center",
                fontsize=8.4, color=BLACK, linespacing=1.2)
        box(ax, cx, 3.60, 2.35, 0.72, node_label, NEU_F, fs=9)
        arrow(ax, (cx, 3.18), (cx, 2.66), lw=2.1)
        ax.text(cx, 2.28, threat, ha="center", va="center",
                fontsize=8.3, color=BLACK, weight="bold", linespacing=1.2)
        box(ax, cx, 1.20, 2.9, 0.82, mit, MIT_F, fs=8.5, weight="bold")

    column(LX, "Risk A — INGRESS",
           "code execution on load\n· property of the FORMAT",
           "Model file (pickle)",
           "arbitrary code runs",
           "Mitigation: code-free format\n(skops / ONNX)")
    column(RX, "Risk B — EGRESS",
           "training-data leakage\n· property of the MODEL",
           "Released model",
           "membership inference\n→ training data leaked",
           "Mitigation: leakage-audit\ncertificate")
    return fig


# ---------------------------------------------------------------- Fig 3 (ROC, recomputed)
def fig3_roc_bw():
    import numpy as np  # noqa
    _add_poc_to_path()
    from config import MASTER_SEED
    from src.d2_risk_b_mia import run_one
    rl = run_one("leaky", MASTER_SEED)
    rs = run_one("safe", MASTER_SEED)
    fl, tl = rl["roc"]
    fs_, ts = rs["roc"]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(fl, tl, color=BLACK, ls="solid", lw=1.9,
            label=f"Leaky (AUC {rl['mia_shadow_auc']:.3f})")
    ax.plot(fs_, ts, color=BLACK, ls=(0, (6, 4)), lw=1.9,
            label=f"Safe (AUC {rs['mia_shadow_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=GRAY, ls=(0, (1, 3)), lw=1.4,
            label="Chance (AUC 0.500)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8.8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Fig 4
def fig4():
    fig, ax = plt.subplots(figsize=(12.4, 5.0))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 5.2); ax.axis("off")
    yg = 2.55
    box(ax, 1.0, yg, 1.5, 1.25, "Trained\nmodel in\nplatform\n(pickle)", NEU_F, fs=8.8)

    def gate(cx, title, body):
        box(ax, cx, yg, 2.3, 1.55, "", PROC_F)
        ax.text(cx, yg + 0.56, title, ha="center", va="center",
                fontsize=8.8, weight="bold", color=BLACK, zorder=5)
        ax.text(cx, yg - 0.16, body, ha="center", va="center",
                fontsize=8.3, color=BLACK, zorder=5, linespacing=1.32)

    gate(3.25, "GATE 1 — Safe format",
         "pickle → skops / ONNX\nno load-time code\ninference parity (Δ=0)")
    gate(5.85, "GATE 2 — Leakage audit",
         "leakage certificate\nshadow MIA + LiRA /\nattribute / reconstr.\n(AUC ≤ 0.55)")
    gate(8.6, "GATE 3 — Escrow",
         "model + environment\n(container, lock, seed,\ndata version) · signed hash\nre-run Δ ≤ 1e-4")
    # Gate 3 -> bundle arrow, and the export-boundary line at its MIDPOINT
    bx0, bx1 = 9.78, 10.13   # gate3 right edge (cx8.6 + 1.15) -> bundle left edge
    xb = (bx0 + bx1) / 2     # dashed line bisects the arrow
    ax.plot([xb, xb], [0.55, 4.55], ls=(0, (5, 4)), color=GRAY, lw=1.4, zorder=1)
    ax.text(xb, 4.62, "export boundary", ha="center", va="bottom",
            fontsize=8, style="italic", color=GRAY)
    box(ax, 11.15, yg, 2.0, 1.7,
        "SAFE MODEL\nBUNDLE\n\nsafe-format model\n+ leakage certificate\n+ signed escrow",
        EMPH_F, fs=8.2, weight="bold", lw=2.4)
    arrow(ax, (1.78, yg), (2.08, yg), lw=1.9)
    arrow(ax, (4.40, yg), (4.70, yg), lw=1.9)
    arrow(ax, (7.00, yg), (7.45, yg), lw=2.0, ls="solid")
    # "PASS" lifted just above the box tops (clear of both Gate 2 and Gate 3 edges)
    ax.text(7.225, yg + 0.86, "PASS", ha="center", va="bottom", fontsize=8.2,
            color=BLACK, weight="bold")
    arrow(ax, (bx0, yg), (bx1, yg), lw=2.0)
    box(ax, 5.85, 4.4, 2.0, 0.78, "Regularized\nre-training", MIT_F, fs=8.6, weight="bold")
    arrow(ax, (5.3, yg + 0.78), (5.3, 4.01), lw=2.0, ls=(0, (4, 3)))
    ax.text(4.62, yg + 1.32, "BLOCK", ha="center", va="center", fontsize=8.2,
            color=BLACK, weight="bold")
    arrow(ax, (6.4, 4.01), (6.4, yg + 0.78), lw=2.0, ls=(0, (4, 3)))
    ax.text(7.05, yg + 1.32, "re-audit", ha="center", va="center", fontsize=8.0,
            color=BLACK, style="italic")
    return fig


# ---------------------------------------------------------------- S1 Fig (DCR, recomputed)
def s1_dcr_bw():
    import numpy as np
    _add_poc_to_path()
    from config import MASTER_SEED, N_POPULATION
    from src.d1_synthesize import generate_population, cart_synthesize, dcr_report
    pop = generate_population(N_POPULATION, MASTER_SEED)
    syn = cart_synthesize(pop, n=len(pop), seed=MASTER_SEED + 1)
    _, summ, d_syn, d_real = dcr_report(pop, syn)
    hi = float(np.percentile(np.concatenate([d_syn, d_real]), 99))
    bins = np.linspace(0, hi, 40)
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.hist(d_syn, bins=bins, density=True, facecolor="#cccccc",
            edgecolor=BLACK, linewidth=0.7, zorder=2,
            label="Synthetic → source (DCR)")
    ax.hist(d_real, bins=bins, density=True, facecolor="none",
            edgecolor=BLACK, linewidth=0.7, hatch="////", zorder=3,
            label="Real → real (baseline)")
    ax.axvline(summ["dcr_synth_median"], color=BLACK, ls="solid", lw=1.5,
               label=f"Synthetic median ({summ['dcr_synth_median']:.2f})")
    ax.axvline(summ["dcr_real_median"], color=BLACK, ls=(0, (6, 4)), lw=1.5,
               label=f"Baseline median ({summ['dcr_real_median']:.2f})")
    ax.set_xlabel("Distance to closest record (SD-standardised Euclidean)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8.2, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- S2 Fig (cost)
def figS2_cost():
    fig, ax = plt.subplots(figsize=(10, 3.8))
    labels = ["Safe Model Bundle\nescrow",
              "Warm-keeping:\nbare VM (floor)",
              "Warm-keeping:\nmanaged seat (low)",
              "Warm-keeping:\nmanaged seat (high)"]
    vals = [500, 414000, 8775000, 13500000]
    notes = ["₩500  (≈US$0.37)", "₩414,000  (≈US$307)",
             "₩8,775,000  (≈US$6,500)", "₩13,500,000  (≈US$10,000)"]
    shades = ["#ffffff", "#c9c9c9", "#8f8f8f", "#595959"]
    hatches = ["....", None, None, None]
    y = list(range(len(labels)))[::-1]
    for yi, v, sh, ht in zip(y, vals, shades, hatches):
        ax.barh(yi, v, color=sh, edgecolor=BLACK, linewidth=1.1, hatch=ht, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(200, 1e10)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Cost over a 6-month revision window (KRW, log scale)", fontsize=9.5)
    for yi, v, n in zip(y, vals, notes):
        ax.text(v * 1.5, yi, n, va="center", ha="left", fontsize=8.6, color=BLACK)
    ax.tick_params(axis="x", labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def _save(fig, stem):
    os.makedirs(RESULTS, exist_ok=True)
    png = os.path.join(RESULTS, stem + ".png")
    tif = os.path.join(RESULTS, stem + ".tif")
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(tif, bbox_inches="tight", facecolor="white", dpi=600)
    plt.close(fig)
    # convert to grayscale + LZW for PLOS submission; cap physical width at
    # 7.4 in @ 600 dpi (downscale high-res render -> preserves layout, no overlap)
    from PIL import Image
    im = Image.open(tif).convert("L")
    max_px = int(7.4 * 600)  # 4440 px
    if im.size[0] > max_px:
        new_h = round(im.size[1] * max_px / im.size[0])
        im = im.resize((max_px, new_h), Image.LANCZOS)
    im.save(tif, format="TIFF", compression="tiff_lzw", dpi=(600, 600))
    return png, tif


if __name__ == "__main__":
    # manuscript figures, in submission order (see module docstring)
    for fn, stem in [(fig2, "Fig1"), (fig3_roc_bw, "Fig2"), (fig4, "Fig3")]:
        png, tif = _save(fn(), stem)
        print("wrote", os.path.basename(png), "+", os.path.basename(tif))
    print("done ->", RESULTS)
