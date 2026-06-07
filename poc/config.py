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
# NCP is the CSAP-certified domestic cloud for an illustrative Korean deployment,
# so it is the relevant unit-cost basis for a Korean deployment. Prices are in
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
