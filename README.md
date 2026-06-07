# Safe Model Bundle — reproducibility package

Companion code and synthetic data for:

> **Safe and Reproducible Model Export from Closed Clinical Data Platforms: A Governance Specification and Synthetic-Data Proof of Concept.** Hojun Yu (2026).

This repository demonstrates, on **synthetic data only**, that a trained model can be exported from a closed clinical-data platform in a form that is simultaneously (1) **safe to release** — no code execution on load, audited for membership-inference leakage — and (2) **sufficient to reproduce** the analysis on demand. No real patient records are used anywhere.

## Contents
- `poc/` — the pipeline and its modules
  - `run_all.py` — orchestrates D1→D4 + the supplementary DP sweep
  - `src/d1_synthesize.py` — synthetic KoGES-schema cohort (logistic DGP + synthpop-style CART), fidelity + DCR disclosure checks
  - `src/d2_risk_a_pickle.py` — pickle deserialization code-execution demo (Risk A)
  - `src/d2_risk_b_mia.py` — shadow-model + loss-threshold membership inference (Risk B), leaky vs safe, multi-seed CI
  - `src/d3_formats.py` — pickle → skops/ONNX conversion, code-exec & inference-parity check (Gate 1)
  - `src/d4_gates.py` — leakage audit + threshold sweep (Gate 2), signed escrow + reproduction + cost (Gate 3)
  - `src/supp_dp.py` — supplementary differential-privacy ε-sweep
  - `config.py` — all fixed parameters (seed, cohort, thresholds, NCP cost model)
- `results/` — canonical exhibits (figures, tables, `summary.json`, `escrow_manifest.json`)

## Reproduce
```bat
cd poc
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python run_all.py
```
The pipeline is **deterministic**: numeric backends are pinned to a single thread and the tree models use XGBoost's exact split-finder, so reruns are bit-identical — including the multi-seed shadow-model attack. Regenerated outputs match those in `results/`.

Environment note: developed on Python 3.12 with `pandas==2.2.3`, `numpy==2.2.6`, `scikit-learn==1.6.1` (pinned to avoid Windows wheel issues); other versions per `poc/requirements.txt`.

## Key results (synthetic; mechanism demonstration, not real-world risk estimates)
- **Gate 1 (format):** skops and ONNX load with no code execution and reproduce predictions exactly (max |Δprob| = 0.0).
- **Gate 3 (escrow):** on-demand re-run reproduces the data hash exactly and held-out utility within ~3×10⁻⁵.
- **Gate 2 (leakage):** a leaky model (membership-inference AUC ≈0.58, 95% CI 0.567–0.594) is blocked while a regularized model of equal-or-better utility (≈0.53, CI 0.516–0.536) passes; decision stable over a 0.50–0.60 threshold sweep.
- **Cost:** escrow-plus-rerun is roughly three to four orders of magnitude cheaper than keeping the analysis environment open.

## Data
All data are synthetic, generated from a documented model of the KoGES *variable schema* (no real KoGES records). The generator and the fixed seed are included.

## License
- Code: **MIT** (see `LICENSE`).
- Data and documentation: **CC-BY-4.0**.

## Citation
Yu H. Safe and Reproducible Model Export from Closed Clinical Data Platforms: A Governance Specification and Synthetic-Data Proof of Concept. 2026. (DOI assigned on release.)
