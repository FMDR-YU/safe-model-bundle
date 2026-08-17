"""
D2 / Risk B - Membership inference attack (an EGRESS / leakage threat).

We mount two standard MIAs against a target model trained on the "members":
  * threshold attack (Yeom et al.): per-example loss as the membership score;
  * shadow-model attack (Shokri et al.): K shadow models trained on the
    attacker pool produce (feature, in/out) pairs to train an attack classifier.

Main-text conditions are two REGIMES of the same modelling task on a small cohort:
  leaky -> a high-capacity default model fit to N_TRAIN_MEMBERS rows (current bad
           practice; overfits -> leaks);
  safe  -> the same task done disclosure-aware (regularised, shallow).
The formal-DP comparator is moved to the supplementary epsilon-sweep (supp_dp.py).

Statistical robustness: the whole pipeline (synthesise -> split -> train -> attack)
is repeated over SEED_LIST and reported as mean +/- 95% CI. Leakage is a property
of the *model*, not of the serialization format -- so blocking pickle does not
address Risk B. The ROC plot (seed 0) is Fig 3 / Fig D1.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from config import (OUTCOME, FEATURES, N_SHADOW_MODELS, N_TRAIN_MEMBERS,
                    N_POPULATION)
from src.utils import make_model, predict_proba_pos, per_example_loss, model_quality
from src.d1_synthesize import get_population, membership_split

# t_{0.975, df=9} for the 10-seed 95% CI (mean +/- t * sd / sqrt(n))
_T_975_DF9 = 2.262


def _attack_features(model, X, y):
    """Per-example features the adversary observes from the target's outputs."""
    p = np.clip(predict_proba_pos(model, X), 1e-7, 1 - 1e-7)
    y = np.asarray(y)
    conf_true = np.where(y == 1, p, 1 - p)                 # confidence on true label
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    return np.column_stack([conf_true, loss, entropy, correct])


def _threshold_auc(model, members, nonmembers):
    """Yeom loss-threshold attack AUC (lower loss => more likely a member)."""
    lm = per_example_loss(model, members[FEATURES], members[OUTCOME])
    ln = per_example_loss(model, nonmembers[FEATURES], nonmembers[OUTCOME])
    scores = -np.concatenate([lm, ln])                     # higher => member
    labels = np.concatenate([np.ones(len(lm)), np.zeros(len(ln))])
    return float(roc_auc_score(labels, scores))


def shadow_attack(kind, target_model, members, attacker_pool, nonmembers, seed,
                  epsilon: float = 1.0):
    """Shokri-style shadow-model attack. Trains N_SHADOW_MODELS shadows of `kind`
    (same training size as the target) on the attacker pool, fits an attack
    classifier on their in/out behaviour, and scores the target's members vs
    non-members. Returns (shadow_auc, (fpr, tpr), tpr@1%FPR, tpr@10%FPR)."""
    rng = np.random.default_rng(seed + 991)
    pool = attacker_pool.reset_index(drop=True)
    m = min(N_TRAIN_MEMBERS, len(pool) // 2)               # match shadow size to target
    feats, labs = [], []
    for k in range(N_SHADOW_MODELS):
        idx = rng.permutation(len(pool))
        s_in = pool.iloc[idx[:m]]
        s_out = pool.iloc[idx[m:2 * m]]
        sm = make_model(kind, seed=seed + 100 + k, epsilon=epsilon)
        sm.fit(s_in[FEATURES], s_in[OUTCOME])
        feats.append(np.vstack([
            _attack_features(sm, s_in[FEATURES], s_in[OUTCOME]),
            _attack_features(sm, s_out[FEATURES], s_out[OUTCOME])]))
        labs.append(np.concatenate([np.ones(m), np.zeros(m)]))
    attack = RandomForestClassifier(n_estimators=200, max_depth=6,
                                    random_state=seed, n_jobs=1)
    attack.fit(np.vstack(feats), np.concatenate(labs))

    f_mem = _attack_features(target_model, members[FEATURES], members[OUTCOME])
    f_non = _attack_features(target_model, nonmembers[FEATURES], nonmembers[OUTCOME])
    scores = np.concatenate([attack.predict_proba(f_mem)[:, 1],
                             attack.predict_proba(f_non)[:, 1]])
    ytrue = np.concatenate([np.ones(len(f_mem)), np.zeros(len(f_non))])
    shadow_auc = float(roc_auc_score(ytrue, scores))
    fpr, tpr, _ = roc_curve(ytrue, scores)
    return (shadow_auc, (fpr, tpr),
            float(np.interp(0.01, fpr, tpr)), float(np.interp(0.10, fpr, tpr)))


def run_one(kind, seed, epsilon: float = 1.0) -> dict:
    """One full draw for one condition: load/synthesise -> split -> train -> attack."""
    df = get_population(seed)
    members, attacker, nonmembers = membership_split(df, seed)
    model = make_model(kind, seed=seed + 7, epsilon=epsilon)
    model.fit(members[FEATURES], members[OUTCOME])

    test = model_quality(model, nonmembers[FEATURES], nonmembers[OUTCOME])   # generalisation
    train = model_quality(model, members[FEATURES], members[OUTCOME])        # fit on members
    thr_auc = _threshold_auc(model, members, nonmembers)
    shadow_auc, roc, tpr01, tpr10 = shadow_attack(
        kind, model, members, attacker, nonmembers, seed, epsilon)
    return {
        "utility_auc": test["auc"],
        "train_auc": train["auc"],
        "train_test_gap": train["auc"] - test["auc"],
        "mia_shadow_auc": shadow_auc,
        "mia_threshold_auc": thr_auc,
        "mia_tpr_at_fpr_0.01": tpr01,
        "mia_tpr_at_fpr_0.10": tpr10,
        "roc": roc,
    }


def _ci95(vals):
    a = np.asarray(vals, dtype=float)
    n = len(a)
    mean = float(a.mean())
    if n < 2:
        return mean, 0.0, mean, mean
    half = _T_975_DF9 * float(a.std(ddof=1)) / np.sqrt(n)
    return mean, half, mean - half, mean + half


_AGG_KEYS = ["utility_auc", "train_auc", "train_test_gap", "mia_shadow_auc",
             "mia_threshold_auc", "mia_tpr_at_fpr_0.01", "mia_tpr_at_fpr_0.10"]


def run(seed_list, out_dir, conditions=("leaky", "safe")):
    """Multi-seed MIA. Returns (Table D1 dataframe, full agg dict, Fig3 path,
    seed-0 point metrics for the gate decision)."""
    per = {c: [] for c in conditions}
    roc0 = {}
    for si, seed in enumerate(seed_list):
        for c in conditions:
            r = run_one(c, seed)
            if si == 0:
                roc0[c] = (r["roc"][0], r["roc"][1], r["mia_shadow_auc"])
            per[c].append({k: v for k, v in r.items() if k != "roc"})

    agg, seed0, table_rows = {}, {}, []
    for c in conditions:
        agg[c] = {"model_kind": c, "n_seeds": len(seed_list)}
        row = {"condition": c, "n_seeds": len(seed_list)}
        for key in _AGG_KEYS:
            mean, half, lo, hi = _ci95([d[key] for d in per[c]])
            agg[c][key] = round(mean, 4)
            agg[c][key + "_ci95_lo"] = round(lo, 4)
            agg[c][key + "_ci95_hi"] = round(hi, 4)
            agg[c][key + "_halfwidth"] = round(half, 4)
        # compact Table D1 row (mean with 95% CI for the two headline metrics)
        row["utility_auc"] = agg[c]["utility_auc"]
        row["utility_auc_95CI"] = f"{agg[c]['utility_auc_ci95_lo']}-{agg[c]['utility_auc_ci95_hi']}"
        row["train_test_gap"] = agg[c]["train_test_gap"]
        row["mia_shadow_auc"] = agg[c]["mia_shadow_auc"]
        row["mia_shadow_auc_95CI"] = f"{agg[c]['mia_shadow_auc_ci95_lo']}-{agg[c]['mia_shadow_auc_ci95_hi']}"
        row["mia_threshold_auc"] = agg[c]["mia_threshold_auc"]
        row["mia_tpr_at_fpr_0.01"] = agg[c]["mia_tpr_at_fpr_0.01"]
        row["mia_tpr_at_fpr_0.10"] = agg[c]["mia_tpr_at_fpr_0.10"]
        table_rows.append(row)
        seed0[c] = {"model_kind": c,
                    "mia_shadow_auc": round(per[c][0]["mia_shadow_auc"], 4),
                    "utility_auc": round(per[c][0]["utility_auc"], 4)}

    table = pd.DataFrame(table_rows)

    # Fig 3 / Fig D1 -- ROC curves from the representative draw (seed 0)
    plt.figure(figsize=(6, 6))
    for c in conditions:
        fpr, tpr, auc = roc0[c]
        plt.plot(fpr, tpr, lw=2, label=f"{c} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="random (AUC=0.500)")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Membership inference attack on the released model\n"
              "(Fig 3 / Fig D1; representative seed)")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fig3_mia_roc.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return table, agg, fig_path, seed0
