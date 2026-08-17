"""
OMOP-CDM -> patient-level feature table loader (T4: public-data substrate).

Replaces the bespoke `d1.generate_population` (KoGES-schema DGP) with a loader
that reads a *standard, public* OMOP Common Data Model dataset and emits a
person-level table of metabolic features + an incident-T2DM outcome, in the
same column contract the downstream PoC (MIA, gates, formats) already consumes.

Why this matters (reviewer R1#2, "tautology"): the data-generating process is no
longer authored here. We target STANDARD OMOP concept_ids, so the identical
loader runs on any OMOP-CDM release (Synthea, SynPUF, real CDM) without change.

NOTE: this loader needs measurement/lab values (glucose, BMI, BP, lipids). It
therefore fits EHR-style OMOP (e.g. Synthea). Claims-only OMOP (e.g. CMS
DE-SynPUF) carries no MEASUREMENT labs; a claims variant would build features
from conditions/drugs instead (see `load_omop_cohort(..., feature_set=...)`,
not yet implemented for claims).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Concept map (standard OMOP concept_ids; portable across OMOP-CDM datasets)
# --------------------------------------------------------------------------- #
# Continuous features <- MEASUREMENT.value_as_number, aggregated per person.
MEASUREMENT_CONCEPTS: dict[str, int] = {
    "bmi":             3038553,  # Body mass index (BMI) [Ratio]
    "sbp":             3004249,  # Systolic blood pressure
    "dbp":             3012888,  # Diastolic blood pressure
    "fasting_glucose": 3000483,  # Glucose [Mass/volume] in Blood
    "hba1c":           3004410,  # Hemoglobin A1c / Hemoglobin.total in Blood
    "total_chol":      3027114,  # Cholesterol [Mass/volume] in Serum or Plasma
    "hdl":             3007070,  # Cholesterol in HDL
    "triglyceride":    3022192,  # Triglyceride [Mass/volume]
}
GENDER_CONCEPTS = {8507: 1, 8532: 0}   # 8507 MALE -> 1, 8532 FEMALE -> 0
T2DM_ROOT_CONCEPT = 201826             # Type 2 diabetes mellitus (SNOMED-standard)

CONTINUOUS_VARS = ["age"] + list(MEASUREMENT_CONCEPTS.keys())
CATEGORICAL_VARS = ["sex"]
OUTCOME = "t2dm"
FEATURES = CONTINUOUS_VARS + CATEGORICAL_VARS


def _read(data_dir: str, table: str) -> pd.DataFrame:
    """Read an OMOP table case-insensitively (Synthea ships UPPERCASE .csv)."""
    for name in (table.upper(), table.lower(), table):
        p = os.path.join(data_dir, name + ".csv")
        if os.path.exists(p):
            return pd.read_csv(p, low_memory=False)
    raise FileNotFoundError(f"{table}.csv not found in {data_dir}")


def _t2dm_concept_set(data_dir: str) -> set[int]:
    """Root T2DM concept plus all its descendants (via CONCEPT_ANCESTOR if present),
    so ICD/SNOMED-coded diagnoses across datasets all count as the outcome."""
    concepts = {T2DM_ROOT_CONCEPT}
    try:
        anc = _read(data_dir, "CONCEPT_ANCESTOR")
        kids = anc.loc[anc["ancestor_concept_id"] == T2DM_ROOT_CONCEPT,
                       "descendant_concept_id"]
        concepts |= set(int(c) for c in kids.dropna().unique())
    except FileNotFoundError:
        pass
    return concepts


def load_omop_cohort(data_dir: str, reference_year: int | None = None,
                     min_nonmissing: int = 3) -> pd.DataFrame:
    """Build a person-level metabolic feature table + incident-T2DM outcome.

    - age: reference_year (default: latest measurement year in the data) minus
      year_of_birth.
    - continuous features: per-person MEAN of value_as_number for each concept
      (deterministic; robust to repeated measures).
    - sex: from PERSON.gender_concept_id.
    - t2dm: 1 if the person has any CONDITION_OCCURRENCE in the T2DM concept set.
    - persons with fewer than `min_nonmissing` of the measurement features present
      are dropped; remaining gaps are median-imputed (imputation flagged in the
      returned attrs).
    Returns a DataFrame with columns FEATURES + [OUTCOME]; df.attrs carries
    provenance (source dir, n dropped, per-column missing rate before imputation).
    """
    person = _read(data_dir, "PERSON")
    meas = _read(data_dir, "MEASUREMENT")
    cond = _read(data_dir, "CONDITION_OCCURRENCE")

    df = pd.DataFrame({"person_id": person["person_id"].astype(int)})
    df["sex"] = person["gender_concept_id"].map(GENDER_CONCEPTS)
    df["_yob"] = person["year_of_birth"].astype(float)

    # reference year for age = latest measurement year observed
    myear = pd.to_datetime(meas["measurement_date"], errors="coerce").dt.year
    ref = int(reference_year if reference_year is not None
              else np.nanmax(myear.to_numpy()))
    df["age"] = (ref - df["_yob"]).clip(lower=0)

    # per-person mean of each target measurement concept
    m = meas[["person_id", "measurement_concept_id", "value_as_number"]].dropna(
        subset=["value_as_number"])
    for feat, cid in MEASUREMENT_CONCEPTS.items():
        s = (m[m["measurement_concept_id"] == cid]
             .groupby("person_id")["value_as_number"].mean())
        df[feat] = df["person_id"].map(s)

    # outcome: any T2DM (root + descendants) condition
    t2dm_set = _t2dm_concept_set(data_dir)
    has = cond.loc[cond["condition_concept_id"].isin(t2dm_set), "person_id"].unique()
    df[OUTCOME] = df["person_id"].isin(has).astype(int)

    # drop persons with too few measured features; record missingness
    meas_cols = list(MEASUREMENT_CONCEPTS.keys())
    present = df[meas_cols].notna().sum(axis=1)
    n_before = len(df)
    df = df[present >= min_nonmissing].copy()
    miss_rate = df[meas_cols].isna().mean().round(4).to_dict()

    # median-impute remaining gaps (deterministic)
    for c in meas_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    df["sex"] = df["sex"].fillna(0).astype(int)

    out = df[FEATURES + [OUTCOME]].reset_index(drop=True).round(3)
    out.attrs["provenance"] = {
        "source_dir": data_dir,
        "reference_year": ref,
        "n_persons_raw": int(n_before),
        "n_persons_kept": int(len(out)),
        "t2dm_prevalence": round(float(out[OUTCOME].mean()), 4),
        "n_t2dm_positive": int(out[OUTCOME].sum()),
        "missing_rate_pre_impute": miss_rate,
        "t2dm_concepts_n": len(t2dm_set),
    }
    return out
