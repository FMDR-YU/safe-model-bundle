"""
Central configuration for the KoGES model-export proof-of-concept.

Everything that affects the experiment is fixed here so that a re-run on a
different machine reproduces the same numbers (the paper's own
reproducibility claim, demonstrated on itself).

NOTE on data: NO real KoGES distributed records are used anywhere. We
synthesise a KoGES-*schema* population from a documented generative model
(see src/d1_synthesize.py). See the paper's ethics section.
"""
from __future__ import annotations
import os

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
MASTER_SEED = 20260601          # single global seed; every stage derives from it
DATA_VERSION = "koges-synth-v1"  # bump if the generative model changes

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
ART = os.path.join(OUT, "artifacts")   # serialized models, manifests
os.makedirs(OUT, exist_ok=True)
os.makedirs(ART, exist_ok=True)

# ----------------------------------------------------------------------------
# Synthetic cohort
# ----------------------------------------------------------------------------
N_POPULATION = 8000             # total synthetic individuals
N_TRAIN_MEMBERS = 300           # target (and each shadow) training-set size.
                                # Small-cohort regime: a high-capacity default model
                                # fit to a modest sample overfits and leaks -- the
                                # common bad practice the leakage gate is meant to catch.
TARGET_TRAIN_FRAC = 0.08        # (legacy; superseded by N_TRAIN_MEMBERS)
ATTACKER_FRAC = 0.50            # pool the adversary uses to train shadow models
# remainder = held-out non-members for evaluation

# KoGES-schema variables (continuous + categorical). Order matters for the
# sequential CART synthesiser (each var is generated conditional on predecessors).
CONTINUOUS_VARS = [
    "age", "bmi", "waist", "sbp", "dbp",
    "fasting_glucose", "total_chol", "hdl", "triglyceride",
]
CATEGORICAL_VARS = [
    "sex", "smoking", "alcohol", "phys_activity", "family_history_dm",
]
OUTCOME = "t2dm"
FEATURES = CONTINUOUS_VARS + CATEGORICAL_VARS

# ----------------------------------------------------------------------------
# Data substrate (T4: standard PUBLIC benchmark replaces the KoGES-schema DGP,
# so the data-generating process is external to the authors -- cuts reviewer
# R1#2 "tautology"). Select with env var POC_SUBSTRATE.
#   "synpuf"   = CMS DE-SynPUF 2008 Beneficiary Summary (~116k, claims) -- MAIN
#   "synthea"  = OHDSI Synthea27Nj OMOP (metabolic labs) -- 27-person demo, smoke test only
#   "mimic_ed" = MIMIC-IV-ED, one row per ED stay, outcome = hospital admission.
#                The open demo release is 222 stays (smoke test only); the full
#                ~425k release is credentialed and must not be redistributed.
#   "koges"    = legacy self-authored KoGES-schema synthetic DGP (pre-pivot)
# ----------------------------------------------------------------------------
SUBSTRATE = os.environ.get("POC_SUBSTRATE", "synpuf")


def _resolve_data_root() -> str:
    """Locate the public CDM data directory so the repo runs both standalone and
    inside the paper's project layout. Priority: POC_DATA_ROOT env var, then a
    repo-relative `data/cdm` (recommended for a fresh clone), then the project
    layout `../../data/cdm`. Returns the last candidate as the default target
    even if it does not yet exist (so the error message names where to put data).
    """
    env = os.environ.get("POC_DATA_ROOT")
    candidates = ([env] if env else []) + [
        os.path.join(ROOT, "..", "data", "cdm"),        # repo-relative (clone)
        os.path.join(ROOT, "..", "..", "data", "cdm"),  # project layout
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[-1])


_DATA_ROOT = _resolve_data_root()
SYNPUF_CSV = os.path.join(_DATA_ROOT, "synpuf",
                          "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv")
SYNTHEA_DIR = os.path.join(_DATA_ROOT, "Synthea27Nj")

# MIMIC-IV-ED sits beside data/cdm/, not inside it: it is not an OMOP CDM
# release. POC_MIMIC_ED_DIR overrides; otherwise prefer the full release and
# fall back to the open demo.
MIMIC_ED_DIR = os.environ.get("POC_MIMIC_ED_DIR") or next(
    (p for p in (
        os.path.join(_DATA_ROOT, "..", "mimic-iv-ed", "full-2.2", "ed"),
        os.path.join(_DATA_ROOT, "..", "mimic-iv-ed", "demo-2.2", "ed"),
    ) if os.path.isdir(p)),
    os.path.join(_DATA_ROOT, "..", "mimic-iv-ed", "demo-2.2", "ed"),
)
MIMIC_ED_DIR = os.path.abspath(MIMIC_ED_DIR)

if SUBSTRATE == "synpuf":
    from src.synpuf_loader import (CONTINUOUS_VARS, CATEGORICAL_VARS,  # noqa: F811,E402
                                   OUTCOME, FEATURES)
    DATA_VERSION = "cms-desynpuf-2008-sample1"
elif SUBSTRATE == "synthea":
    from src.omop_loader import (CONTINUOUS_VARS, CATEGORICAL_VARS,    # noqa: F811,E402
                                 OUTCOME, FEATURES)
    DATA_VERSION = "ohdsi-synthea27nj-omop"
elif SUBSTRATE == "mimic_ed":
    from src.mimic_ed_loader import (CONTINUOUS_VARS, CATEGORICAL_VARS,  # noqa: F811,E402
                                     OUTCOME, FEATURES)
    DATA_VERSION = ("mimic-iv-ed-2.2-full" if "full-2.2" in MIMIC_ED_DIR
                    else "mimic-iv-ed-demo-2.2")
# else "koges": keep the KoGES-schema defaults above (DATA_VERSION=koges-synth-v1)

# ----------------------------------------------------------------------------
# Membership-inference attack (Risk B)
# ----------------------------------------------------------------------------
N_SHADOW_MODELS = 16            # Shokri-style shadow models
MIA_AUC_PASS_THRESHOLD = 0.55   # Gate 2 leakage threshold (RQ5); >0.55 => block
THRESHOLD_SWEEP = [0.50, 0.52, 0.55, 0.58, 0.60]  # report BLOCK/PASS stability vs threshold

# Statistical robustness: repeat the whole synthesise->split->train->attack
# pipeline over multiple data seeds and report mean +/- 95% CI (not a single draw).
SEED_LIST = [MASTER_SEED + i for i in range(10)]   # seed 0 (=MASTER_SEED) is the
                                                   # representative draw for Fig 3

# ----------------------------------------------------------------------------
# Differential privacy -- SUPPLEMENTARY only (formal DP collapses utility at small
# epsilon, so it is reported as an epsilon-sweep in supp_dp.py, not in the main
# text). The defended/safe path in the body uses regularisation, not DP.
# ----------------------------------------------------------------------------
DP_EPSILON = 1.0                          # legacy single point (kept for reference)
DP_EPSILON_SWEEP = [0.5, 1.0, 2.0, 4.0, 8.0]   # supplementary epsilon-sweep

# ----------------------------------------------------------------------------
# Gate-3 escrow cost model -- Naver Cloud Platform (NCP) published list prices.
# NCP is a Korean domestic cloud; its published list prices are an illustrative
# unit-cost basis for a Korean deployment. Prices are in
# KRW, retrieved 2026-06-05 from ncloud.com/charge (Standard g3 2vCPU/8GB server
# and Object Storage, KR region). Figures are illustrative, not a quote.
# ----------------------------------------------------------------------------
COST = {
    "currency": "KRW",
    "krw_per_usd": 1350.0,              # illustrative FX for USD-equivalent reporting (2026)
    "revision_window_months": 6,        # submission -> final revision
    # Warm-keeping baseline is reported as a floor-to-ceiling range, because the
    # cost of keeping an environment alive depends on what is being kept:
    #  - FLOOR  = a bare Standard 2vCPU/8GB analysis VM on NCP's monthly committed
    #             rate (the domestic, public-institution-relevant unit price).
    #  - CEILING= a managed TRE/SDE *seat*, which bundles secure desktop, support,
    #             output review, storage and licensing. We anchor this on the only
    #             openly published seat tariff we found, the US CMS Virtual Research
    #             Data Center: USD 20,000 first year / 13,000 renewal year per seat
    #             (retrieved 2026-06-05). International basis, used for an upper bound.
    "warm_monthly_krw": 69000.0,        # NCP Standard g3 2vCPU/8GB, per month (floor)
    "tre_seat_usd_per_year_low": 13000.0,   # CMS VRDC renewal-year seat (ceiling, low)
    "tre_seat_usd_per_year_high": 20000.0,  # CMS VRDC first-year seat (ceiling, high)
    # escrow path: object storage of the signed artifact + on-demand re-runs.
    "escrow_storage_krw_per_gb_month": 30.0,   # NCP Object Storage, standard tier
    "rerun_hourly_krw": 250.0,          # NCP Standard g3 2vCPU/8GB, per hour (on-demand)
    "expected_rerun_hours": 1.0,        # one revision re-run
    "n_expected_reruns": 2,             # reviewer rounds
}
