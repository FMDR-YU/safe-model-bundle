"""
MIMIC-IV-ED -> one row per ED stay + binary outcome (hospital admission).

A third public substrate alongside SynPUF and Synthea. MIMIC-IV-ED is a real
(de-identified) emergency-department record from Beth Israel Deaconess, so the
data-generating process is external to the authors in the same sense SynPUF is
(R1#2), but unlike SynPUF it is *not* synthetic: the full release is credentialed
and carries a DUA forbidding redistribution. See data/mimic-iv-ed/README.md.

  demo-2.2  64 patients / 222 stays, open (ODbL). Wiring and smoke tests only --
            far too small for the membership-inference protocol, which needs a
            300-row training set plus a disjoint shadow pool.
  full-2.2  ~425,000 stays, credentialed. Same schema, so the loader is unchanged.

Task
----
Predict hospital admission from the ED (`disposition == 'ADMITTED'`) using only
what is known at triage. This is the standard MIMIC-IV-ED benchmark task and it
keeps the label clinically meaningful rather than an artifact of coding.

What is deliberately NOT used, and why
--------------------------------------
Three of the six tables would leak the outcome and are excluded:

  diagnosis  ICD codes assigned by trained coders *after* discharge. Using them
             to predict the disposition of that same stay is circular.
  pyxis      medications dispensed *during* the stay; what a patient is given in
             the ED is downstream of the decision to admit.
  vitalsign  aperiodic vitals charted *through* the stay, including after the
             admission decision. Triage vitals are used instead.

`outtime` is likewise excluded: ED length of stay is only known once the stay
ends, so a model using it would not be deployable at triage.

medrecon is used, but only as a count of the medications the patient was already
taking on arrival; the reconciliation is performed at intake, so it is available
at the time the prediction would be made.
"""
from __future__ import annotations

import os

import pandas as pd

# Triage vitals, plus arrival timing and prior-medication burden.
CONTINUOUS_VARS = [
    "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
    "pain", "n_prior_meds", "arrival_hour",
]
# acuity is ordinal 1 (most severe) to 5; kept as an integer rather than one-hot.
CATEGORICAL_VARS = [
    "acuity", "gender", "arrival_ambulance", "arrival_walkin",
    "race_group", "arrival_weekend", "triage_incomplete",
]
OUTCOME = "admitted"
FEATURES = CONTINUOUS_VARS + CATEGORICAL_VARS

_TRIAGE_VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]

# MIMIC race strings are many and sparse; collapse to coarse buckets so the
# column does not become a near-identifier in small cohorts.
_RACE_GROUPS = {
    "WHITE": 1, "BLACK": 2, "HISPANIC": 3, "ASIAN": 4,
    "AMERICAN INDIAN": 5, "NATIVE HAWAIIAN": 5,
}


def _race_group(s: pd.Series) -> pd.Series:
    up = s.fillna("").astype(str).str.upper()
    out = pd.Series(0, index=s.index, dtype=int)  # 0 = other/unknown/declined
    for prefix, code in _RACE_GROUPS.items():
        out = out.mask(up.str.startswith(prefix) & (out == 0), code)
    return out


def load_mimic_ed_cohort(ed_dir: str) -> pd.DataFrame:
    """Return a per-ED-stay table with FEATURES + [OUTCOME]=admitted.

    `ed_dir` is the directory holding the six `*.csv.gz` tables (the `ed/`
    subdirectory of a PhysioNet MIMIC-IV-ED release). df.attrs['provenance']
    carries source, n, and the admission rate.
    """
    def _read(name):
        path = os.path.join(ed_dir, name + ".csv.gz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Fetch a MIMIC-IV-ED release first "
                f"(see data/mimic-iv-ed/README.md)."
            )
        return pd.read_csv(path, low_memory=False)

    stays = _read("edstays")
    triage = _read("triage")
    medrecon = _read("medrecon")

    df = pd.DataFrame(index=stays.index)
    df["stay_id"] = stays["stay_id"].values

    intime = pd.to_datetime(stays["intime"], errors="coerce")
    df["arrival_hour"] = intime.dt.hour.astype(float).values
    df["arrival_weekend"] = (intime.dt.dayofweek >= 5).astype(int).values

    df["gender"] = (stays["gender"].astype(str).str.upper().str.strip() == "M").astype(int).values
    transport = stays["arrival_transport"].astype(str).str.upper().str.strip()
    df["arrival_ambulance"] = transport.isin(["AMBULANCE", "HELICOPTER"]).astype(int).values
    df["arrival_walkin"] = (transport == "WALK IN").astype(int).values
    df["race_group"] = _race_group(stays["race"]).values

    # --- triage ------------------------------------------------------------
    tri = triage.drop_duplicates("stay_id").set_index("stay_id")
    df = df.join(tri[_TRIAGE_VITALS + ["acuity"]], on="stay_id")
    # `pain` is free text from v2.0 ("UA" for unable to assess, and similar).
    df["pain"] = pd.to_numeric(
        tri["pain"].reindex(df["stay_id"]).values, errors="coerce")
    df.loc[(df["pain"] < 0) | (df["pain"] > 10), "pain"] = pd.NA

    # The six triage vitals are largely missing or present together, so one flag
    # records that the triage assessment was incomplete rather than six.
    df["triage_incomplete"] = df[_TRIAGE_VITALS].isna().any(axis=1).astype(int)

    # --- prior medication burden -------------------------------------------
    # medrecon repeats a drug once per ontology class (etc_rn), so count distinct
    # drug names, not rows.
    per_stay = medrecon.groupby("stay_id")["name"].nunique()
    df["n_prior_meds"] = df["stay_id"].map(per_stay).fillna(0).astype(float)

    # --- outcome ------------------------------------------------------------
    df[OUTCOME] = (stays["disposition"].astype(str).str.upper().str.strip()
                   == "ADMITTED").astype(int).values

    # --- imputation ---------------------------------------------------------
    # XGBoost handles NaN, but the linear and neural families in the extension
    # experiments do not; impute so every model family sees the same matrix.
    for c in CONTINUOUS_VARS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    df["acuity"] = pd.to_numeric(df["acuity"], errors="coerce").fillna(3).astype(int)
    for c in CATEGORICAL_VARS:
        df[c] = df[c].astype(int)

    out = df[FEATURES + [OUTCOME]].reset_index(drop=True)
    out.attrs["provenance"] = {
        "source": ed_dir,
        "dataset": "MIMIC-IV-ED (BIDMC emergency department)",
        "unit": "ED stay",
        "n": int(len(out)),
        "n_features": len(FEATURES),
        "outcome": OUTCOME,
        "admission_rate": round(float(out[OUTCOME].mean()), 4),
        "excluded_tables": ["diagnosis", "pyxis", "vitalsign"],
        "exclusion_reason": "coded after discharge / recorded during the stay",
    }
    return out
