"""Supplementary cost figure (Fig S2): warm-keeping vs Safe Model Bundle escrow.

Reads the same cost model used in D4, so the figure stays consistent with Table D3
and the Supplement. Deterministic (no randomness).
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import COST
from src.d4_gates import cost_model


def _usd(krw: float, fx: float) -> str:
    u = krw / fx
    return f"US${u:,.2f}" if u < 100 else f"US${u:,.0f}"


def run(out_dir, model_artifact_kb: float = 50.0):
    c = cost_model(model_artifact_kb)
    fx = c["krw_per_usd"]
    items = [
        ("Safe Model Bundle escrow", c["escrow_total_krw"]),
        ("Warm-keeping: bare VM (floor)", c["warm_floor_krw"]),
        ("Warm-keeping: managed seat (low)", c["warm_seat_low_krw"]),
        ("Warm-keeping: managed seat (high)", c["warm_seat_high_krw"]),
    ]
    labels = [a for a, _ in items]
    vals = [b for _, b in items]
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#8c1a1a"]

    plt.figure(figsize=(9.0, 3.6))
    ys = list(range(len(items)))
    plt.barh(ys, vals, color=colors)
    plt.xscale("log")
    plt.gca().invert_yaxis()
    plt.yticks(ys, labels, fontsize=9)
    plt.xlabel("Cost over a 6-month revision window (KRW, log scale)")
    for i, v in enumerate(vals):
        plt.text(v * 1.25, i, f"₩{v:,.0f}  (≈{_usd(v, fx)})",
                 va="center", fontsize=8)
    plt.title("Safe Model Bundle escrow vs keeping the environment warm "
              "(6-month revision; NCP list prices)", fontsize=9)
    plt.xlim(left=min(v for v in vals if v > 0) / 3, right=max(vals) * 1000)
    plt.tight_layout()
    path = os.path.join(out_dir, "figS2_cost.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import OUT
    print(run(OUT))
