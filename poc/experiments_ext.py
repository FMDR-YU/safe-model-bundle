#!/usr/bin/env python3
"""
Reviewer-robustness experiments that extend the main PoC (run_all.py).
Addresses PLOS reviewer points that a single leaky-vs-safe pair cannot:

  R1#3  Continuous leakage as a FUNCTION of cohort size and model capacity
        (defeats the "two engineered points" framing).
  R2 M5 Model-family scope: the leakage gate is exercised on LINEAR (logistic)
        and NEURAL (MLP) targets, not only gradient-boosted trees.
  R1#4  A stronger attack BATTERY against the same targets: an offline
        likelihood-ratio attack (LiRA), a Fredrikson-style attribute-inference
        attack, and an attribute-reconstruction advantage (member vs non-member).

Everything is deterministic (single-thread numeric backends + fixed seeds) and
uses the same public substrate as run_all.py (CMS DE-SynPUF by default). No new
numbers are invented; this script computes them. Outputs go to ../results/.

Run:  python experiments_ext.py            (full)
      POC_EXT_SMOKE=1 python experiments_ext.py   (fast smoke: 1 seed, few shadows)
"""
from __future__ import annotations
import os
# pin numeric backends BEFORE importing numpy/sklearn/xgboost (bit-reproducible)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve

from config import FEATURES, OUTCOME, N_SHADOW_MODELS, N_TRAIN_MEMBERS
from src.utils import make_model, predict_proba_pos, model_quality, save_table, save_json
from src.d1_synthesize import get_population, membership_split
from src.d2_risk_b_mia import _attack_features, _threshold_auc

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESULTS, exist_ok=True)
SMOKE = os.environ.get("POC_EXT_SMOKE") == "1"

SEEDS_EXT = [20260601 + i for i in range(1 if SMOKE else 5)]     # families / battery
SEEDS_SWEEP = [20260601 + i for i in range(1 if SMOKE else 3)]   # trend only
K_SHADOW = 4 if SMOKE else N_SHADOW_MODELS
K_LIRA = 8 if SMOKE else 48
NONMEMBER_EVAL_CAP = 300          # balanced evaluation set for MIA/LiRA
SENSITIVE = "sp_depressn"         # a genuinely sensitive binary comorbidity flag
_T = {2: 12.706, 3: 4.303, 5: 2.776}                      # t_{.975} small-n fallback


# --------------------------------------------------------------------------- #
# Model factory (family x regime). Tree reuses the canonical make_model.
# --------------------------------------------------------------------------- #
def family_model(family: str, regime: str, seed: int):
    if family == "tree":
        return make_model(regime, seed=seed)             # canonical leaky/safe XGB
    if family == "linear":
        C = 10000.0 if regime == "leaky" else 0.05
        return Pipeline([("s", StandardScaler()),
                         ("clf", LogisticRegression(C=C, max_iter=3000,
                                                    random_state=seed))])
    if family == "neural":
        if regime == "leaky":
            mlp = MLPClassifier(hidden_layer_sizes=(128, 128), alpha=1e-6,
                                max_iter=1500, random_state=seed,
                                early_stopping=False)
        else:
            mlp = MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0,
                                max_iter=1500, random_state=seed,
                                early_stopping=True, n_iter_no_change=15,
                                validation_fraction=0.2)
        return Pipeline([("s", StandardScaler()), ("clf", mlp)])
    raise ValueError(family)


def xgb_rounds(n_rounds: int, seed: int):
    """Leaky-style GBT with complexity varied ONLY through the number of boosting
    rounds (n_estimators), depth fixed. Depth saturates leakage almost immediately
    at 400 rounds, so rounds is the axis that resolves the full underfit->overfit
    range and gives a smooth complexity->leakage curve."""
    if HAS_XGB:
        return XGBClassifier(n_estimators=n_rounds, max_depth=6,
                             learning_rate=0.3, subsample=1.0,
                             colsample_bytree=1.0, reg_lambda=0.0,
                             min_child_weight=1, eval_metric="logloss",
                             random_state=seed, n_jobs=1, tree_method="exact")
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=n_rounds, max_depth=6,
                                      learning_rate=0.3, random_state=seed)


def _eval_nonmembers(nonmembers, seed):
    """Balanced, capped non-member evaluation set (seeded subsample)."""
    if len(nonmembers) <= NONMEMBER_EVAL_CAP:
        return nonmembers.reset_index(drop=True)
    rng = np.random.default_rng(seed + 4242)
    idx = rng.choice(len(nonmembers), NONMEMBER_EVAL_CAP, replace=False)
    return nonmembers.iloc[idx].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Generic shadow-model MIA (family-aware), reusing d2's attack features.
# --------------------------------------------------------------------------- #
def shadow_mia(factory, target, members, pool, nonmembers, seed, n_members):
    rng = np.random.default_rng(seed + 991)
    pool = pool.reset_index(drop=True)
    m = min(n_members, len(pool) // 2)
    feats, labs = [], []
    for k in range(K_SHADOW):
        idx = rng.permutation(len(pool))
        s_in, s_out = pool.iloc[idx[:m]], pool.iloc[idx[m:2 * m]]
        sm = factory(seed + 100 + k)
        sm.fit(s_in[FEATURES], s_in[OUTCOME])
        feats.append(np.vstack([_attack_features(sm, s_in[FEATURES], s_in[OUTCOME]),
                                _attack_features(sm, s_out[FEATURES], s_out[OUTCOME])]))
        labs.append(np.concatenate([np.ones(m), np.zeros(m)]))
    atk = RandomForestClassifier(n_estimators=200, max_depth=6,
                                 random_state=seed, n_jobs=1)
    atk.fit(np.vstack(feats), np.concatenate(labs))
    fm = _attack_features(target, members[FEATURES], members[OUTCOME])
    fn = _attack_features(target, nonmembers[FEATURES], nonmembers[OUTCOME])
    s = np.concatenate([atk.predict_proba(fm)[:, 1], atk.predict_proba(fn)[:, 1]])
    yt = np.concatenate([np.ones(len(fm)), np.zeros(len(fn))])
    return float(roc_auc_score(yt, s))


# --------------------------------------------------------------------------- #
# Offline LiRA (Carlini et al.): per-example Gaussian null from shadow OUT
# models, one-sided likelihood-ratio score on the logit confidence.
# --------------------------------------------------------------------------- #
def _logit_conf_true(model, df):
    p = np.clip(predict_proba_pos(model, df[FEATURES]), 1e-6, 1 - 1e-6)
    y = df[OUTCOME].values
    conf = np.clip(np.where(y == 1, p, 1 - p), 1e-6, 1 - 1e-6)
    return np.log(conf / (1 - conf))


def lira_offline(target, members, nonmembers_eval, factory, seed, n_members):
    rng = np.random.default_rng(seed + 555)
    ev = pd.concat([members, nonmembers_eval], ignore_index=True)
    labels = np.concatenate([np.ones(len(members)), np.zeros(len(nonmembers_eval))])
    n = len(ev)
    train_size = min(n_members, n // 2)
    s_sum = np.zeros(n); s_sq = np.zeros(n); cnt = np.zeros(n)
    for k in range(K_LIRA):
        idx = rng.permutation(n)
        in_idx = idx[:train_size]
        mask_in = np.zeros(n, bool); mask_in[in_idx] = True
        sm = factory(seed + 700 + k)
        sm.fit(ev.iloc[in_idx][FEATURES], ev.iloc[in_idx][OUTCOME])
        ph = _logit_conf_true(sm, ev)
        out = ~mask_in
        s_sum[out] += ph[out]; s_sq[out] += ph[out] ** 2; cnt[out] += 1
    cnt = np.maximum(cnt, 1)
    mu = s_sum / cnt
    sig = np.sqrt(np.clip(s_sq / cnt - mu ** 2, 1e-4, None))
    score = (_logit_conf_true(target, ev) - mu) / sig     # higher => member
    fpr, tpr, _ = roc_curve(labels, score)
    return float(roc_auc_score(labels, score)), float(np.interp(0.01, fpr, tpr))


# --------------------------------------------------------------------------- #
# Attribute inference (Fredrikson) + attribute-reconstruction advantage.
# --------------------------------------------------------------------------- #
def attribute_infer_acc(target, records, sensitive):
    """Guess the hidden sensitive value that maximises the target's confidence on
    the record's true label; accuracy over the records."""
    vals = sorted(records[sensitive].unique())
    y = records[OUTCOME].values
    cols = np.zeros((len(records), len(vals)))
    for j, v in enumerate(vals):
        Xv = records[FEATURES].copy()
        Xv[sensitive] = v
        p = np.clip(predict_proba_pos(target, Xv), 1e-6, 1 - 1e-6)
        cols[:, j] = np.where(y == 1, p, 1 - p)
    guess = np.array(vals)[cols.argmax(axis=1)]
    return float((guess == records[sensitive].values).mean())


def attribute_baseline(records, sensitive):
    p = records[sensitive].mean()
    return float(max(p, 1 - p))


# --------------------------------------------------------------------------- #
# CI helper
# --------------------------------------------------------------------------- #
def _ci(vals):
    a = np.asarray(vals, float)
    n = len(a)
    mean = float(a.mean())
    if n < 2:
        return mean, mean, mean
    half = _T.get(n, 2.776) * float(a.std(ddof=1)) / np.sqrt(n)
    return mean, mean - half, mean + half


# =========================================================================== #
# Experiments
# =========================================================================== #
def run_sweeps():
    """R1#3: MIA AUC as a smooth function of (a) cohort size, (b) model depth."""
    sizes = [100, 300] if SMOKE else [100, 200, 300, 500, 1000, 2000]
    rounds = [3, 300] if SMOKE else [1, 3, 10, 30, 100, 300, 600]
    coh_rows, cap_rows = [], []

    for N in sizes:
        aucs = []
        for seed in SEEDS_SWEEP:
            df = get_population(seed)
            members, pool, nonmembers = membership_split(df, seed, n_members=N)
            nm = _eval_nonmembers(nonmembers, seed)
            tgt = make_model("leaky", seed=seed + 7)
            tgt.fit(members[FEATURES], members[OUTCOME])
            aucs.append(shadow_mia(lambda s: make_model("leaky", seed=s),
                                   tgt, members, pool, nm, seed, N))
        m, lo, hi = _ci(aucs)
        coh_rows.append({"cohort_size": N, "mia_auc": round(m, 4),
                         "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)})
        print(f"  [sweep] N={N:5d}  MIA AUC={m:.3f} [{lo:.3f},{hi:.3f}]")

    for r in rounds:
        aucs = []
        for seed in SEEDS_SWEEP:
            df = get_population(seed)
            members, pool, nonmembers = membership_split(df, seed, n_members=300)
            nm = _eval_nonmembers(nonmembers, seed)
            tgt = xgb_rounds(r, seed + 7)
            tgt.fit(members[FEATURES], members[OUTCOME])
            aucs.append(shadow_mia(lambda s, rr=r: xgb_rounds(rr, s),
                                   tgt, members, pool, nm, seed, 300))
        m, lo, hi = _ci(aucs)
        cap_rows.append({"n_rounds": r, "mia_auc": round(m, 4),
                         "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)})
        print(f"  [sweep] rounds={r:3d}  MIA AUC={m:.3f} [{lo:.3f},{hi:.3f}]")

    coh = pd.DataFrame(coh_rows); cap = pd.DataFrame(cap_rows)
    save_table(coh, "tableS_sweep_cohort", RESULTS)
    save_table(cap, "tableS_sweep_capacity", RESULTS)

    # figure (two panels, grayscale)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].errorbar(coh["cohort_size"], coh["mia_auc"],
                   yerr=[coh["mia_auc"] - coh["ci95_lo"], coh["ci95_hi"] - coh["mia_auc"]],
                   marker="o", color="black", lw=1.5, capsize=3)
    ax[0].axhline(0.55, ls="--", color="0.4", lw=1)
    ax[0].set_xscale("log"); ax[0].set_xlabel("training cohort size (log)")
    ax[0].set_ylabel("membership-inference AUC")
    ax[0].set_title("(a) leakage vs cohort size", fontsize=10)
    ax[1].errorbar(cap["n_rounds"], cap["mia_auc"],
                   yerr=[cap["mia_auc"] - cap["ci95_lo"], cap["ci95_hi"] - cap["mia_auc"]],
                   marker="s", color="black", lw=1.5, capsize=3)
    ax[1].axhline(0.55, ls="--", color="0.4", lw=1)
    ax[1].set_xscale("log"); ax[1].set_xlabel("model complexity (boosting rounds, log)")
    ax[1].set_ylabel("membership-inference AUC")
    ax[1].set_title("(b) leakage vs complexity (N=300)", fontsize=10)
    for a in ax:
        a.grid(True, ls=":", lw=0.5, color="0.8")
    fig.tight_layout()
    fpath = os.path.join(RESULTS, "figS_leakage_sweep.png")
    fig.savefig(fpath, dpi=200); plt.close(fig)
    print("  wrote", os.path.basename(fpath))
    return coh, cap


def run_families():
    """R2 M5: leaky/safe leakage + utility across tree / linear / neural."""
    rows = []
    for family in ["tree", "linear", "neural"]:
        for regime in ["leaky", "safe"]:
            mia, util, gap = [], [], []
            for seed in SEEDS_EXT:
                df = get_population(seed)
                members, pool, nonmembers = membership_split(df, seed)
                nm = _eval_nonmembers(nonmembers, seed)
                tgt = family_model(family, regime, seed + 7)
                tgt.fit(members[FEATURES], members[OUTCOME])
                mia.append(shadow_mia(lambda s, f=family, r=regime: family_model(f, r, s),
                                      tgt, members, pool, nm, seed, N_TRAIN_MEMBERS))
                util.append(model_quality(tgt, nm[FEATURES], nm[OUTCOME])["auc"])
                tr = model_quality(tgt, members[FEATURES], members[OUTCOME])["auc"]
                gap.append(tr - util[-1])
            mm, mlo, mhi = _ci(mia); um, ulo, uhi = _ci(util); gm, _, _ = _ci(gap)
            rows.append({"family": family, "regime": regime,
                         "mia_auc": round(mm, 4),
                         "mia_ci95": f"{mlo:.3f}-{mhi:.3f}",
                         "utility_auc": round(um, 4),
                         "train_test_gap": round(gm, 4),
                         "gate": "BLOCK" if mm > 0.55 else "PASS"})
            print(f"  [family] {family:7s} {regime:5s}  MIA={mm:.3f}  util={um:.3f}  "
                  f"{'BLOCK' if mm > 0.55 else 'PASS'}")
    fam = pd.DataFrame(rows)
    save_table(fam, "tableS_model_families", RESULTS)
    return fam


def run_battery():
    """R1#4: LiRA + attribute inference + reconstruction advantage, leaky vs safe
    (gradient-boosted family; the canonical main-text target)."""
    rows = []
    for regime in ["leaky", "safe"]:
        lira_auc, lira_tpr, shadow, ai_mem, ai_non, recon = [], [], [], [], [], []
        base = None
        for seed in SEEDS_EXT:
            df = get_population(seed)
            members, pool, nonmembers = membership_split(df, seed)
            nm = _eval_nonmembers(nonmembers, seed)
            fac = lambda s: make_model(regime, seed=s)
            tgt = make_model(regime, seed=seed + 7)
            tgt.fit(members[FEATURES], members[OUTCOME])

            shadow.append(shadow_mia(fac, tgt, members, pool, nm, seed, N_TRAIN_MEMBERS))
            la, lt = lira_offline(tgt, members, nm, fac, seed, N_TRAIN_MEMBERS)
            lira_auc.append(la); lira_tpr.append(lt)

            am = attribute_infer_acc(tgt, members, SENSITIVE)
            an = attribute_infer_acc(tgt, nm, SENSITIVE)
            ai_mem.append(am); ai_non.append(an); recon.append(am - an)
            base = attribute_baseline(members, SENSITIVE)
        sa, salo, sahi = _ci(shadow)
        la, lalo, lahi = _ci(lira_auc)
        lt, _, _ = _ci(lira_tpr)
        aim, _, _ = _ci(ai_mem); ain, _, _ = _ci(ai_non); rc, rclo, rchi = _ci(recon)
        rows.append({"regime": regime,
                     "shadow_mia_auc": round(sa, 4), "shadow_ci95": f"{salo:.3f}-{sahi:.3f}",
                     "lira_auc": round(la, 4), "lira_ci95": f"{lalo:.3f}-{lahi:.3f}",
                     "lira_tpr_at_1pct_fpr": round(lt, 4),
                     "attr_infer_member_acc": round(aim, 4),
                     "attr_infer_nonmember_acc": round(ain, 4),
                     "attr_baseline": round(base, 4),
                     "recon_member_advantage": round(rc, 4),
                     "recon_adv_ci95": f"{rclo:.3f}-{rchi:.3f}"})
        print(f"  [battery] {regime:5s}  shadow={sa:.3f}  LiRA={la:.3f} "
              f"(tpr@1%={lt:.3f})  attr_adv(mem-non)={rc:.3f}")
    bat = pd.DataFrame(rows)
    save_table(bat, "tableS_attack_battery", RESULTS)
    return bat


if __name__ == "__main__":
    print(f"experiments_ext  (SMOKE={SMOKE}; sensitive={SENSITIVE})")
    print("== sweeps (R1#3) ==");    coh, cap = run_sweeps()
    print("== families (R2 M5) ==");  fam = run_families()
    print("== battery (R1#4) ==");    bat = run_battery()
    save_json({"seeds_ext": SEEDS_EXT, "seeds_sweep": SEEDS_SWEEP,
               "k_shadow": K_SHADOW, "k_lira": K_LIRA,
               "sensitive_attribute": SENSITIVE,
               "cohort_sweep": coh.to_dict("records"),
               "capacity_sweep": cap.to_dict("records"),
               "model_families": fam.to_dict("records"),
               "attack_battery": bat.to_dict("records")},
              os.path.join(RESULTS, "summary_ext.json"))
    print("done ->", RESULTS)
