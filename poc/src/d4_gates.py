"""
D4 - The 3-gate "Safe Model Output" pipeline -> Table D3.

Gate 1 (format):   convert pkl -> safe format (handled in d3; here we record it).
Gate 2 (leakage):  pre-export MIA/SDC audit; PASS iff attack AUC <= threshold.
Gate 3 (escrow):   sign+hash model + environment + data version; on-demand
                   re-run must reproduce metrics and hashes (reproducibility);
                   compare warm-keeping vs escrow cost.

Table D3 ties pre/post leakage, reproducibility (hash match), and cost to C4.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from config import (FEATURES, OUTCOME, MIA_AUC_PASS_THRESHOLD, DATA_VERSION, COST,
                    THRESHOLD_SWEEP)
from src.utils import (make_model, model_quality, sha256_file, sha256_df,
                       hmac_sign, capture_environment, save_json)
from src.d1_synthesize import generate_population, membership_split


# --------------------------------------------------------------------------- #
# Gate 2 - leakage audit
# --------------------------------------------------------------------------- #
def gate2_decision(mia_shadow_auc: float) -> dict:
    passed = mia_shadow_auc <= MIA_AUC_PASS_THRESHOLD
    return {
        "mia_shadow_auc": round(float(mia_shadow_auc), 4),
        "threshold": MIA_AUC_PASS_THRESHOLD,
        "decision": "PASS (release)" if passed else "BLOCK (re-train, regularised)",
        "passed": bool(passed),
    }


def threshold_sweep(leaky_auc: float, safe_auc: float, thresholds=None) -> pd.DataFrame:
    """Show that the BLOCK/PASS decision is stable across plausible thresholds:
    the leaky model is blocked and the safe model passes for every value in the
    sweep (or report the value at which a decision flips)."""
    thresholds = THRESHOLD_SWEEP if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        rows.append({
            "threshold": t,
            "leaky_decision": "PASS" if leaky_auc <= t else "BLOCK",
            "safe_decision": "PASS" if safe_auc <= t else "BLOCK",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Gate 3 - escrow + reproducibility + cost
# --------------------------------------------------------------------------- #
def build_escrow_manifest(model_artifact_path: str, data_df: pd.DataFrame,
                          seed: int, metrics: dict, out_dir: str) -> dict:
    env = capture_environment()
    manifest = {
        "data_version": DATA_VERSION,
        "master_seed": seed,
        "data_hash": sha256_df(data_df),
        "model_artifact": os.path.basename(model_artifact_path),
        "model_hash": sha256_file(model_artifact_path),
        "env_hash": env["env_hash"],
        "python": env["python"],
        "platform": env["platform"],
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
    }
    import hashlib, json
    manifest["manifest_digest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["escrow_signature"] = hmac_sign(manifest["manifest_digest"])
    save_json(manifest, os.path.join(out_dir, "escrow_manifest.json"))
    return manifest


def reproduce_and_verify(manifest: dict, seed: int, n_population: int,
                         model_kind: str) -> dict:
    """Re-run from the manifest's seed/data-version and check it reproduces."""
    df2 = generate_population(n_population, seed)
    members2, _, nonmembers2 = membership_split(df2, seed)
    model2 = make_model(model_kind, seed=seed + 7)
    model2.fit(members2[FEATURES], members2[OUTCOME])
    metrics2 = model_quality(model2, nonmembers2[FEATURES], nonmembers2[OUTCOME])

    data_hash2 = sha256_df(df2)
    data_match = (data_hash2 == manifest["data_hash"])
    # metric reproducibility (utility AUC) within tight tolerance
    util_delta = abs(metrics2["auc"] - manifest["metrics"].get("utility_auc",
                                                               metrics2["auc"]))
    return {
        "data_hash_match": bool(data_match),
        "rerun_utility_auc": round(metrics2["auc"], 6),
        "utility_auc_delta": round(float(util_delta), 8),
        "metric_tolerance": 1e-4,
        "reproducible": bool(data_match and util_delta < 1e-4),
    }


def cost_model(model_artifact_kb: float) -> dict:
    """Warm-keeping vs escrow cost over a revision window, on NCP list prices.

    Warm-keeping = keeping the analysis VM alive for the whole window (the cost a
    platform/researcher bears today because the model cannot be exported).
    Escrow = object storage of the signed artifact + a couple of on-demand
    re-runs. Native unit is KRW; USD-equivalent is reported via a documented FX.
    """
    c = COST
    months = c["revision_window_months"]
    fx = c["krw_per_usd"]

    def usd(x: float) -> float:
        return round(x / fx, 2)

    # warm-keeping: floor (bare NCP VM) to ceiling (managed TRE seat)
    warm_floor_krw = c["warm_monthly_krw"] * months
    seat_low_krw = c["tre_seat_usd_per_year_low"] * fx * months / 12.0
    seat_high_krw = c["tre_seat_usd_per_year_high"] * fx * months / 12.0

    # escrow: object storage of the signed artifact + on-demand re-runs
    gb = max(model_artifact_kb / (1024 * 1024), 1e-4)
    storage_krw = c["escrow_storage_krw_per_gb_month"] * gb * months
    rerun_krw = (c["rerun_hourly_krw"] * c["expected_rerun_hours"]
                 * c["n_expected_reruns"])
    escrow_krw = storage_krw + rerun_krw

    def savings_pct(warm: float) -> float:
        return round(100 * (warm - escrow_krw) / warm, 2)

    return {
        "currency": "KRW",
        "cloud": "Naver Cloud Platform (Standard 2vCPU/8GB + Object Storage)",
        "krw_per_usd": fx,
        "revision_window_months": months,
        # warm-keeping baseline as a floor-to-ceiling range
        "warm_floor_krw": round(warm_floor_krw, 0),          # bare NCP VM
        "warm_floor_usd": usd(warm_floor_krw),
        "warm_seat_low_krw": round(seat_low_krw, 0),         # managed TRE seat (low)
        "warm_seat_low_usd": usd(seat_low_krw),
        "warm_seat_high_krw": round(seat_high_krw, 0),       # managed TRE seat (high)
        "warm_seat_high_usd": usd(seat_high_krw),
        # escrow alternative
        "escrow_storage_krw": round(storage_krw, 2),
        "escrow_rerun_krw": round(rerun_krw, 0),
        "escrow_total_krw": round(escrow_krw, 2),
        "escrow_total_usd": usd(escrow_krw),
        # savings vs each warm scenario
        "savings_vs_floor_pct": savings_pct(warm_floor_krw),
        "savings_vs_seat_low_pct": savings_pct(seat_low_krw),
        "savings_vs_seat_high_pct": savings_pct(seat_high_krw),
        # backward-compatible aliases (floor = headline figure)
        "warm_keeping_krw": round(warm_floor_krw, 0),
        "warm_keeping_usd": usd(warm_floor_krw),
        "savings_pct": savings_pct(warm_floor_krw),
    }


def run(leaky_metrics, safe_metrics, members, full_df, seed,
        model_artifact_path, out_dir) -> pd.DataFrame:
    """Assemble Table D3 across the leaky (current practice) and the gated/safe
    conditions, plus a threshold sweep."""
    rows = []

    # leaky: would it pass Gate 2?
    g2_l = gate2_decision(leaky_metrics["mia_shadow_auc"])
    rows.append({
        "condition": "leaky pickle (current practice)",
        "gate1_format": "pickle (code-exec on load)",
        "gate2_mia_auc": g2_l["mia_shadow_auc"],
        "gate2_decision": g2_l["decision"],
        "gate3_reproducible": "n/a (no escrow)",
        "utility_auc": leaky_metrics["utility_auc"],
    })

    # gated/safe path (mitigation after a BLOCK = regularised re-train)
    g2_s = gate2_decision(safe_metrics["mia_shadow_auc"])
    manifest = build_escrow_manifest(
        model_artifact_path, full_df, seed,
        {"utility_auc": safe_metrics["utility_auc"]}, out_dir)
    repro = reproduce_and_verify(
        manifest, seed, len(full_df), safe_metrics["model_kind"])
    cost = cost_model(os.path.getsize(model_artifact_path) / 1024)

    rows.append({
        "condition": "3-gate (safe format + audit + escrow)",
        "gate1_format": "ONNX/skops (no code-exec)",
        "gate2_mia_auc": g2_s["mia_shadow_auc"],
        "gate2_decision": g2_s["decision"],
        "gate3_reproducible": "yes (hash+metric match)" if repro["reproducible"] else "no",
        "utility_auc": safe_metrics["utility_auc"],
    })

    table = pd.DataFrame(rows)
    sweep = threshold_sweep(leaky_metrics["mia_shadow_auc"],
                            safe_metrics["mia_shadow_auc"])
    extras = {"gate3_reproducibility": repro, "cost_model": cost,
              "escrow_manifest_digest": manifest["manifest_digest"],
              "threshold_sweep": sweep.to_dict("records")}
    return table, extras, sweep
