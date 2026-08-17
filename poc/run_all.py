"""
Orchestrate the full proof-of-concept: D1 -> D4, writing every exhibit to
outputs/. Run:  python run_all.py
All randomness derives from config.MASTER_SEED, so re-running reproduces the
numbers (the paper demonstrates its own reproducibility claim).
"""
from __future__ import annotations
import os, sys, json

# Pin every numeric backend to a single thread BEFORE numpy/xgboost import, so
# floating-point reduction order is fixed and shadow-MIA is bit-reproducible
# (the paper's own reproducibility claim, demonstrated on itself).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (MASTER_SEED, N_POPULATION, OUT, FEATURES, OUTCOME, SEED_LIST,
                    DATA_VERSION, SUBSTRATE)
from src.utils import make_model, model_quality, save_table, save_json, sha256_file
from src import d1_synthesize as d1
from src import d2_risk_a_pickle as d2a
from src import d2_risk_b_mia as d2b
from src import d3_formats as d3
from src import d4_gates as d4
from src import supp_dp
from src import cost_plot

import pickle


def main():
    print("=" * 70)
    print(f"Model-export PoC  |  substrate={SUBSTRATE}  seed={MASTER_SEED}")
    print("=" * 70)

    # ---------------- D1: cohort ----------------
    print(f"\n[D1] loading cohort (substrate={SUBSTRATE}) ...")
    pop = d1.get_population(MASTER_SEED)
    pop.to_csv(os.path.join(OUT, "cohort.csv"), index=False)
    prov = getattr(pop, "attrs", {}).get("provenance", {})
    print(f"     n={len(pop)}  {OUTCOME} prevalence={pop[OUTCOME].mean():.3f}")
    if prov:
        print(f"     provenance: {prov}")

    # Self-synthesis fidelity/DCR exhibits (tableD0/tableD0b/figD2) apply ONLY to
    # the legacy self-authored substrate; with a standard public benchmark the
    # model trains on the public data directly (no author-made synthetic release).
    disclosure = dcr_summary = dcr_fig = None
    if SUBSTRATE == "koges":
        syn = d1.cart_synthesize(pop, n=len(pop), seed=MASTER_SEED + 1)
        fidelity = d1.fidelity_report(pop, syn)
        save_table(fidelity, "tableD0_synth_fidelity", OUT)
        disclosure = d1.disclosure_report(pop, syn)
        dcr_table, dcr_summary, d_syn, d_real = d1.dcr_report(pop, syn)
        save_table(dcr_table, "tableD0b_dcr", OUT)
        dcr_fig = d1.plot_dcr(d_syn, d_real, os.path.join(OUT, "figD2_dcr.png"))
        print(f"     synth fidelity(max|Δmean|)={fidelity['abs_mean_delta'].max():.3f} "
              f"exact_dups={disclosure['exact_duplicates']} "
              f"DCR_ratio={dcr_summary['min_ratio_synth_over_real']}")

    members, attacker, nonmembers = d1.membership_split(pop, MASTER_SEED)
    print(f"     members={len(members)} attacker_pool={len(attacker)} "
          f"non-members={len(nonmembers)}")

    # target model = current practice (leaky: high-capacity default on small cohort)
    target = make_model("leaky", seed=MASTER_SEED + 7)
    target.fit(members[FEATURES], members[OUTCOME])
    pkl_path = os.path.join(OUT, "artifacts", "target_model.pkl")
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(target, f)

    # ---------------- D2-A: pickle ACE (Risk A) ----------------
    print("\n[D2-A] Risk A - pickle deserialization code execution ...")
    risk_a = d2a.run(OUT)
    print(f"     code executed on load: {risk_a['pickle_executes_code_on_load']}  "
          f"| picklescan detected: {risk_a['picklescan_detected']}")

    # ---------------- D2-B: MIA (Risk B), multi-seed leaky vs safe ----------------
    print(f"\n[D2-B] Risk B - membership inference, {len(SEED_LIST)} seeds "
          "(this can take a few min) ...")
    tableD1, mia_agg, fig3, mia0 = d2b.run(SEED_LIST, OUT)
    save_table(tableD1, "tableD1_risk_decomposition", OUT)
    for c in ("leaky", "safe"):
        a = mia_agg[c]
        print(f"     {c:5s} util={a['utility_auc']:.3f} gap={a['train_test_gap']:.3f} "
              f"MIA_shadow={a['mia_shadow_auc']:.3f} "
              f"[{a['mia_shadow_auc_ci95_lo']}-{a['mia_shadow_auc_ci95_hi']}] "
              f"TPR@1%={a['mia_tpr_at_fpr_0.01']:.3f}")
    print(f"     Fig 3 saved -> {os.path.basename(fig3)}")

    # ---------------- D3: format comparison (Gate 1) ----------------
    print("\n[D3] serialization format comparison (Gate 1) ...")
    # use a portable sklearn RF so ONNX/skops parity always runs live
    portable = make_model("portable_rf", seed=MASTER_SEED + 7)
    portable.fit(members[FEATURES].values, members[OUTCOME].values)
    fmt_df = d3.run(portable, nonmembers, OUT)
    save_table(fmt_df, "tableD2_format_comparison", OUT)
    print(fmt_df[["format", "code_exec_on_load", "parity_max_abs_delta"]].to_string(index=False))

    # ---------------- D4: 3-gate + escrow + cost (Table D3) ----------------
    print("\n[D4] 3-gate pipeline: audit + escrow + reproducibility + cost ...")
    # safe model artifact for escrow (safe-format stand-in: skops/onnx in D3)
    safe_model = make_model("safe", seed=MASTER_SEED + 7)
    safe_model.fit(members[FEATURES], members[OUTCOME])
    safe_pkl = os.path.join(OUT, "artifacts", "safe_model.pkl")
    with open(safe_pkl, "wb") as f:
        pickle.dump(safe_model, f)

    # Gate-2 decision uses the 10-seed MEAN leakage (robust to single-seed noise),
    # not a single representative draw. Fig 3 ROC still shows the seed-0 curve.
    d3_table, extras, sweep = d4.run(
        leaky_metrics=mia_agg["leaky"],
        safe_metrics=mia_agg["safe"],
        members=members, full_df=pop, seed=MASTER_SEED,
        model_artifact_path=safe_pkl, out_dir=OUT,
        escrow_metrics=mia0["safe"])   # seed-0 instance for Gate-3 reproducibility
    save_table(d3_table, "tableD3_three_gate_results", OUT)
    save_table(sweep, "tableD3b_threshold_sweep", OUT)
    save_json(extras, os.path.join(OUT, "tableD3_extras.json"))
    print(d3_table.to_string(index=False))
    print("     threshold sweep:", extras["threshold_sweep"])
    print("     reproducibility:", extras["gate3_reproducibility"])
    print("     cost model:", extras["cost_model"])

    # ---------------- Supplementary: DP epsilon-sweep ----------------
    print("\n[Supp] formal-DP epsilon sweep (supplementary) ...")
    dp_table, dp_fig = supp_dp.run(OUT, seed=MASTER_SEED)
    print(dp_table.to_string(index=False))
    print(f"     Fig S1 saved -> {os.path.basename(dp_fig)}")
    cost_fig = cost_plot.run(OUT)
    print(f"     Fig S2 (cost) saved -> {os.path.basename(cost_fig)}")

    # ---------------- summary ----------------
    tables = ["tableD1_risk_decomposition", "tableD2_format_comparison",
              "tableD3_three_gate_results", "tableD3b_threshold_sweep",
              "tableS1_dp_sweep"]
    artifacts = {
        "cohort": "cohort.csv",
        "fig3": os.path.basename(fig3),
        "figS1_dp_sweep": os.path.basename(dp_fig),
        "figS2_cost": os.path.basename(cost_fig),
    }
    summary = {
        "substrate": SUBSTRATE,
        "data_version": DATA_VERSION,
        "cohort_summary": {"n": int(len(pop)), "outcome": OUTCOME,
                           "prevalence": round(float(pop[OUTCOME].mean()), 4),
                           "provenance": prov},
        "risk_a": risk_a,
        "risk_b_mia_agg": mia_agg,
        "risk_b_mia_seed0": mia0,
        "gate3": extras["gate3_reproducibility"],
        "threshold_sweep": extras["threshold_sweep"],
        "cost": extras["cost_model"],
        "dp_sweep_supplementary": dp_table.to_dict("records"),
    }
    if disclosure is not None:                       # koges self-synthesis only
        summary["synth_disclosure"] = disclosure
        summary["synth_dcr"] = dcr_summary
        tables = ["tableD0_synth_fidelity", "tableD0b_dcr"] + tables
        artifacts["figD2_dcr"] = os.path.basename(dcr_fig)
    artifacts["tables"] = tables
    summary["artifacts"] = artifacts
    save_json(summary, os.path.join(OUT, "summary.json"))
    print("\nDone. All exhibits written to:", OUT)


if __name__ == "__main__":
    main()
