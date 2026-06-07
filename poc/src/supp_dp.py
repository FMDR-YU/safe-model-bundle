"""
Supplementary - formal differential-privacy epsilon-sweep.

Formal DP (diffprivlib logistic regression) is reported here, not in the main
text, because at small epsilon it collapses utility to chance while buying little
extra protection over plain regularisation in this small-cohort tabular setting.
We sweep epsilon and plot utility (held-out AUC) against membership leakage
(shadow-MIA AUC), with the non-DP regularised ("safe") model as the reference.
Deterministic (seeded); no real data.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (OUTCOME, FEATURES, N_POPULATION, MASTER_SEED, DP_EPSILON_SWEEP)
from src.utils import make_model, model_quality, save_table
from src.d1_synthesize import generate_population, membership_split
from src.d2_risk_b_mia import shadow_attack


def run(out_dir, seed: int = MASTER_SEED):
    df = generate_population(N_POPULATION, seed)
    members, attacker, nonmembers = membership_split(df, seed)

    rows = []
    for eps in DP_EPSILON_SWEEP:
        model = make_model("dp", seed=seed + 7, epsilon=eps)
        model.fit(members[FEATURES], members[OUTCOME])
        util = model_quality(model, nonmembers[FEATURES], nonmembers[OUTCOME])["auc"]
        shadow_auc, _, tpr01, _ = shadow_attack(
            "dp", model, members, attacker, nonmembers, seed, epsilon=eps)
        rows.append({"epsilon": eps, "utility_auc": round(util, 4),
                     "mia_shadow_auc": round(shadow_auc, 4),
                     "mia_tpr_at_fpr_0.01": round(tpr01, 4)})

    # non-DP reference: the regularised "safe" model from the main text
    safe = make_model("safe", seed=seed + 7)
    safe.fit(members[FEATURES], members[OUTCOME])
    util_s = model_quality(safe, nonmembers[FEATURES], nonmembers[OUTCOME])["auc"]
    sa, _, t01, _ = shadow_attack("safe", safe, members, attacker, nonmembers, seed)
    rows.append({"epsilon": "inf (non-DP safe)", "utility_auc": round(util_s, 4),
                 "mia_shadow_auc": round(sa, 4), "mia_tpr_at_fpr_0.01": round(t01, 4)})

    table = pd.DataFrame(rows)
    save_table(table, "tableS1_dp_sweep", out_dir)

    xs = [r["epsilon"] for r in rows[:-1]]
    util = [r["utility_auc"] for r in rows[:-1]]
    atk = [r["mia_shadow_auc"] for r in rows[:-1]]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, util, "o-", label="utility AUC (held-out)")
    plt.plot(xs, atk, "s-", label="MIA shadow AUC")
    plt.axhline(0.5, ls="--", color="k", lw=1, label="random (0.50)")
    plt.xscale("log")
    plt.xlabel("DP epsilon (log scale)")
    plt.ylabel("AUC")
    plt.title("Supplementary: formal-DP epsilon sweep\n(utility vs membership leakage)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "figS1_dp_sweep.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return table, fig_path
