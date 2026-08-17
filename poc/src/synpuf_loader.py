"""
CMS DE-SynPUF (claims) -> patient-level feature table + binary outcome.

T4 substrate (Yu decision 2026-08-14): a fixed, public, citable dataset whose
data-generating process is external to the authors — the CMS 2008-2010 Data
Entrepreneurs' Synthetic Public Use File (DE-SynPUF), Sample 1 Beneficiary
Summary. This cuts the reviewer R1#2 "tautology" objection (we neither designed
the DGP nor generated the instance) while giving real scale (~116k beneficiaries).

Unlike Synthea (EHR-style, has labs), SynPUF is CLAIMS data with no lab values,
so the prediction task is re-cast in claims terms: predict the CMS diabetes
chronic-condition flag from the other comorbidity flags + demographics +
utilization/reimbursement. No OMOP ETL or vocabulary is required — the
Beneficiary Summary CSV is already a clean per-beneficiary table.

Source file: DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv
  http://www.cms.gov/.../synpufs/downloads/de1_0_2008_beneficiary_summary_file_sample_1.zip
"""
from __future__ import annotations
import pandas as pd

REFERENCE_YEAR = 2008  # the beneficiary-summary year (age = ref - birth year)

# CMS chronic-condition flags (coded 1=Yes, 2=No). SP_DIABETES is the outcome.
CHRONIC_FLAGS = [
    "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
    "SP_DEPRESSN", "SP_ISCHMCHT", "SP_OSTEOPRS", "SP_RA_OA", "SP_STRKETIA",
]  # 10 comorbidity predictors (SP_DIABETES excluded -> it is the label)
OUTCOME_FLAG = "SP_DIABETES"

COVERAGE_COLS = ["BENE_HI_CVRAGE_TOT_MONS", "BENE_SMI_CVRAGE_TOT_MONS",
                 "BENE_HMO_CVRAGE_TOT_MONS", "PLAN_CVRG_MOS_NUM"]
REIMB_COLS = ["MEDREIMB_IP", "BENRES_IP", "PPPYMT_IP",
              "MEDREIMB_OP", "BENRES_OP", "PPPYMT_OP",
              "MEDREIMB_CAR", "BENRES_CAR", "PPPYMT_CAR"]

CONTINUOUS_VARS = ["age"] + COVERAGE_COLS + REIMB_COLS
CATEGORICAL_VARS = ["sex", "race", "esrd"] + [f.lower() for f in CHRONIC_FLAGS]
OUTCOME = "diabetes"
FEATURES = CONTINUOUS_VARS + CATEGORICAL_VARS


def _flag01(s: pd.Series) -> pd.Series:
    """CMS chronic-condition coding 1=Yes / 2=No -> 1 / 0."""
    return (s.astype(str).str.strip() == "1").astype(int)


def load_synpuf_cohort(csv_path: str) -> pd.DataFrame:
    """Return a per-beneficiary table with FEATURES + [OUTCOME]=diabetes.

    df.attrs['provenance'] carries source, n, and diabetes prevalence.
    """
    raw = pd.read_csv(csv_path, dtype=str)

    out = pd.DataFrame(index=range(len(raw)))
    out["age"] = REFERENCE_YEAR - raw["BENE_BIRTH_DT"].str[:4].astype(int)
    out["sex"] = (raw["BENE_SEX_IDENT_CD"].astype(str).str.strip() == "1").astype(int)  # 1=male
    out["race"] = pd.to_numeric(raw["BENE_RACE_CD"], errors="coerce").fillna(0).astype(int)
    out["esrd"] = (raw["BENE_ESRD_IND"].astype(str).str.strip().str.upper() == "Y").astype(int)

    for c in COVERAGE_COLS + REIMB_COLS:
        out[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)

    for f in CHRONIC_FLAGS:
        out[f.lower()] = _flag01(raw[f])

    out[OUTCOME] = _flag01(raw[OUTCOME_FLAG])

    out = out[FEATURES + [OUTCOME]].reset_index(drop=True)
    out.attrs["provenance"] = {
        "source": csv_path,
        "dataset": "CMS DE-SynPUF Sample 1, 2008 Beneficiary Summary",
        "n": int(len(out)),
        "n_features": len(FEATURES),
        "outcome": OUTCOME,
        "diabetes_prevalence": round(float(out[OUTCOME].mean()), 4),
    }
    return out
