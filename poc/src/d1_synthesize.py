"""
D1 - Synthetic data generation (KoGES schema; NO real records).

Two components:
  1. generate_population(): a documented, fully-seeded generative model (DGP)
     that produces a KoGES-*schema* cohort with clinically plausible
     dependencies and an incident-T2DM outcome with real signal.
  2. cart_synthesize(): a synthpop-style sequential CART synthesiser that
     LEARNS from a source table and emits a synthetic release. This is the
     recommended (transparent, deterministic, seed-fixed) synthesis method;
     CTGAN can be swapped in as a sensitivity comparison.

The DGP output is what the "researcher" trains models on. Splitting it into
members / attacker-pool / non-members supports the Risk-B (MIA) demo.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors

from config import (CONTINUOUS_VARS, CATEGORICAL_VARS, OUTCOME, FEATURES,
                    N_POPULATION, N_TRAIN_MEMBERS, ATTACKER_FRAC)


# --------------------------------------------------------------------------- #
# 1. Generative model (ground-truth DGP)
# --------------------------------------------------------------------------- #
def generate_population(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sex = rng.integers(0, 2, n)                          # 0=F, 1=M
    age = np.clip(rng.normal(56, 8, n), 40, 79)

    bmi = np.clip(rng.normal(23.8 + 0.02 * (age - 56) + 0.6 * sex, 3.0, n), 15, 42)
    waist = np.clip(2.0 * bmi + 6 * sex + rng.normal(28, 5, n), 55, 130)

    sbp = np.clip(rng.normal(118 + 0.45 * (age - 56) + 0.7 * (bmi - 24), 14, n), 85, 200)
    dbp = np.clip(0.55 * sbp + rng.normal(8, 7, n), 55, 120)

    fhx = rng.binomial(1, 0.18, n)                       # family history of DM
    smoking = rng.choice([0, 1, 2], n, p=[0.55, 0.25, 0.20])  # never/former/current
    alcohol = rng.choice([0, 1, 2], n, p=[0.45, 0.40, 0.15])
    activity = rng.choice([0, 1, 2], n, p=[0.30, 0.45, 0.25])  # low/mid/high

    fasting_glucose = np.clip(
        90 + 0.9 * (bmi - 24) + 0.25 * (age - 56) + 6 * fhx
        - 2.0 * activity + rng.normal(0, 8, n), 70, 220)
    total_chol = np.clip(rng.normal(193 + 0.3 * (age - 56), 32, n), 110, 330)
    hdl = np.clip(rng.normal(54 - 6 * sex - 0.15 * (bmi - 24), 12, n), 20, 100)
    triglyceride = np.clip(rng.normal(120 + 2.2 * (bmi - 24) + 18 * (alcohol == 2), 55, n), 35, 500)

    # Incident T2DM: logistic on a transparent linear predictor.
    lp = (-2.3
          + 0.045 * (age - 56)
          + 0.12 * (bmi - 24)
          + 0.06 * (fasting_glucose - 100)
          + 0.9 * fhx
          + 0.015 * (waist - 85)
          - 0.25 * activity
          + 0.30 * (smoking == 2)
          + 0.004 * (triglyceride - 120))
    prob = 1.0 / (1.0 + np.exp(-lp))
    t2dm = rng.binomial(1, prob)

    df = pd.DataFrame({
        "age": age, "bmi": bmi, "waist": waist, "sbp": sbp, "dbp": dbp,
        "fasting_glucose": fasting_glucose, "total_chol": total_chol,
        "hdl": hdl, "triglyceride": triglyceride,
        "sex": sex, "smoking": smoking, "alcohol": alcohol,
        "phys_activity": activity, "family_history_dm": fhx,
        OUTCOME: t2dm,
    })
    return df[FEATURES + [OUTCOME]].round(3)


# --------------------------------------------------------------------------- #
# 2. synthpop-style sequential CART synthesiser
# --------------------------------------------------------------------------- #
def cart_synthesize(source: pd.DataFrame, n: int, seed: int,
                    order: list[str] | None = None) -> pd.DataFrame:
    """Generate `n` synthetic rows by modelling each column conditional on the
    already-synthesised predecessors with a CART, then sampling from the matched
    leaf's empirical values (the synthpop default mechanism)."""
    rng = np.random.default_rng(seed)
    cols = order or (FEATURES + [OUTCOME])
    cat = set(CATEGORICAL_VARS + [OUTCOME])
    syn = pd.DataFrame(index=range(n))

    # first column: bootstrap from empirical marginal
    syn[cols[0]] = rng.choice(source[cols[0]].values, size=n, replace=True)

    for j, col in enumerate(cols[1:], start=1):
        preds = cols[:j]
        Xs, ys = source[preds].values, source[col].values
        if col in cat:
            tree = DecisionTreeClassifier(min_samples_leaf=10, random_state=seed + j)
        else:
            tree = DecisionTreeRegressor(min_samples_leaf=10, random_state=seed + j)
        tree.fit(Xs, ys)
        leaf_src = tree.apply(Xs)
        leaf_syn = tree.apply(syn[preds].values)
        members = {}
        for lf in np.unique(leaf_src):
            members[lf] = ys[leaf_src == lf]
        out = np.empty(n, dtype=ys.dtype)
        for i, lf in enumerate(leaf_syn):
            pool = members.get(lf, ys)
            out[i] = pool[rng.integers(len(pool))]
        syn[col] = out
    return syn[cols]


def fidelity_report(source: pd.DataFrame, syn: pd.DataFrame) -> pd.DataFrame:
    """Marginal-fidelity check: |mean| and |std| deltas + outcome prevalence."""
    rows = []
    for c in FEATURES + [OUTCOME]:
        rows.append({
            "variable": c,
            "src_mean": round(float(source[c].mean()), 3),
            "syn_mean": round(float(syn[c].mean()), 3),
            "abs_mean_delta": round(abs(float(source[c].mean()) - float(syn[c].mean())), 3),
            "src_std": round(float(source[c].std()), 3),
            "syn_std": round(float(syn[c].std()), 3),
        })
    return pd.DataFrame(rows)


def disclosure_report(source: pd.DataFrame, syn: pd.DataFrame) -> dict:
    """Disclosure-risk check for the Table D0 caption (deterministic).

    Two standard synthetic-data privacy metrics:
      * exact-duplicate rate -- synthetic rows that reproduce a source row
        verbatim (a verbatim copy is the clearest disclosure failure);
      * nearest-neighbour distance -- for each synthetic row, the Euclidean
        distance to the closest source row, after standardising every column by
        its source SD so the distance is scale-free. A floor at (or near) zero
        would flag near-copies; a healthy synthesiser keeps the minimum well
        above zero.
    The synthpop sequential-CART mechanism samples leaf-level empirical values
    per column, recombining across rows, so verbatim full-row copies are
    structurally unlikely -- this quantifies that claim.
    """
    cols = FEATURES + [OUTCOME]
    S = source[cols].to_numpy(dtype=float)
    Y = syn[cols].to_numpy(dtype=float)

    # exact duplicates (compare on the 3-dp rounding the release is stored at)
    src_rows = set(map(tuple, np.round(S, 3)))
    n_dup = sum(1 for r in np.round(Y, 3) if tuple(r) in src_rows)

    # standardised nearest-source-row distance
    sd = S.std(axis=0)
    sd[sd == 0] = 1.0
    nn = NearestNeighbors(n_neighbors=1).fit(S / sd)
    dist, _ = nn.kneighbors(Y / sd)
    dist = dist.ravel()

    return {
        "n_synth": int(len(syn)),
        "exact_duplicates": int(n_dup),
        "exact_duplicate_rate": round(n_dup / len(syn), 5),
        "nn_dist_min": round(float(dist.min()), 4),
        "nn_dist_p01": round(float(np.percentile(dist, 1)), 4),
        "nn_dist_mean": round(float(dist.mean()), 4),
        "nn_dist_metric": "Euclidean on per-column SD-standardised features",
    }


def dcr_report(source: pd.DataFrame, syn: pd.DataFrame):
    """Distance-to-closest-record (DCR) analysis with a real-to-real baseline.

    The standard privacy test for a synthetic release is not just that DCR is
    non-zero, but that synthetic rows are *no closer* to the source than two real
    records are to each other. We therefore compare two distributions on the same
    SD-standardised Euclidean metric:
      * synthetic -> nearest source row (the DCR of the release);
      * source    -> nearest *other* source row (leave-one-out, the real-real
        baseline -- the irreducible closeness already present among real records).
    If the synthetic DCR distribution is not shifted below the baseline, the
    synthesiser is not memorising. Returns (percentile table, summary dict, raw
    synthetic DCRs, raw real-real distances) -- all deterministic.
    """
    cols = FEATURES + [OUTCOME]
    S = source[cols].to_numpy(dtype=float)
    Y = syn[cols].to_numpy(dtype=float)
    sd = S.std(axis=0)
    sd[sd == 0] = 1.0
    Sn, Yn = S / sd, Y / sd

    d_syn = NearestNeighbors(n_neighbors=1).fit(Sn).kneighbors(Yn)[0].ravel()
    # second neighbour of each source row to itself == nearest *other* source row
    d_real = NearestNeighbors(n_neighbors=2).fit(Sn).kneighbors(Sn)[0][:, 1]

    pcts = [0, 1, 5, 25, 50]
    labels = ["min", "p01", "p05", "p25", "median"]
    table = pd.DataFrame({
        "percentile": labels,
        "synth_to_source_DCR": [round(float(np.percentile(d_syn, p)), 4) for p in pcts],
        "real_to_real_baseline": [round(float(np.percentile(d_real, p)), 4) for p in pcts],
    })
    real_p01 = float(np.percentile(d_real, 1))
    summary = {
        "metric": "Euclidean on per-column SD-standardised features",
        "dcr_synth_min": round(float(d_syn.min()), 4),
        "dcr_real_min": round(float(d_real.min()), 4),
        "dcr_synth_median": round(float(np.median(d_syn)), 4),
        "dcr_real_median": round(float(np.median(d_real)), 4),
        # >=1 means the closest synthetic row is no nearer than the closest real pair
        "min_ratio_synth_over_real": round(float(d_syn.min() / max(d_real.min(), 1e-9)), 3),
        # share of synthetic rows nearer to source than the 1st percentile of the
        # real-real baseline (a small value => no anomalous near-copies)
        "pct_synth_below_real_p01": round(float((d_syn < real_p01).mean()) * 100, 2),
    }
    return table, summary, d_syn, d_real


def plot_dcr(d_syn, d_real, out_path: str):
    """Overlay DCR histograms (synthetic-to-source vs real-to-real baseline)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hi = float(np.percentile(np.concatenate([d_syn, d_real]), 99))
    bins = np.linspace(0, hi, 60)
    plt.figure(figsize=(6, 4))
    plt.hist(d_real, bins=bins, density=True, alpha=0.55, label="real-to-real (baseline)")
    plt.hist(d_syn, bins=bins, density=True, alpha=0.55, label="synthetic-to-source (DCR)")
    plt.axvline(float(d_real.min()), color="C0", ls="--", lw=1)
    plt.axvline(float(d_syn.min()), color="C1", ls="--", lw=1)
    plt.xlabel("Distance to closest record (SD-standardised Euclidean)")
    plt.ylabel("Density")
    plt.title("Distance-to-closest-record: synthetic vs real baseline\n"
              "(Fig D2 / Supplementary)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# --------------------------------------------------------------------------- #
# 3. Membership split
# --------------------------------------------------------------------------- #
def membership_split(df: pd.DataFrame, seed: int, n_members: int | None = None):
    """Return (target_train=members, attacker_pool, nonmembers).

    `n_members` fixes the target training-set size (default N_TRAIN_MEMBERS); the
    small-cohort regime that makes a high-capacity default model overfit and leak.
    """
    n_members = N_TRAIN_MEMBERS if n_members is None else n_members
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_tr = int(n_members)
    n_at = int(ATTACKER_FRAC * len(df))
    members = df.iloc[idx[:n_tr]].reset_index(drop=True)
    attacker = df.iloc[idx[n_tr:n_tr + n_at]].reset_index(drop=True)
    nonmembers = df.iloc[idx[n_tr + n_at:]].reset_index(drop=True)
    return members, attacker, nonmembers
