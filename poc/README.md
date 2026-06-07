# KoGES Model-Export Proof-of-Concept

Reproducibility package for the demo (D1–D4) behind the paper
*"Model-Object Export Restrictions as a Reproducibility Gap in Closed Clinical
Data Platforms"* (target: *Healthcare Informatics Research*).

The package **demonstrates the paper's own claims on itself**: every number is
derived from a single seed (`config.MASTER_SEED`), and Gate 3 re-runs from that
seed to verify hash + metric reproducibility.

> **Ethics / data.** No real KoGES distributed records (2023 CSV) are used
> anywhere. The cohort is synthesised from a documented generative model that
> only mimics the KoGES *variable schema* (age, sex, BMI, BP, fasting glucose,
> lipids, lifestyle, family history → incident T2DM). See `src/d1_synthesize.py`.

## Quick start
```bash
pip install -r requirements.txt
python run_all.py          # writes every exhibit to ./outputs/
```
Runtime ≈ 1–2 min on a laptop (single-threaded for bit-exact results).

## What maps to what

| Stage | Module | Exhibit | Contribution |
|-------|--------|---------|--------------|
| D1 synthesis | `src/d1_synthesize.py` | `synthetic_population.csv`, `tableD0_synth_fidelity` | data realism |
| D2-A Risk A (ingress code-exec) | `src/d2_risk_a_pickle.py` | part of `tableD1` / `summary.json` | C3 |
| D2-B Risk B (egress leakage / MIA) | `src/d2_risk_b_mia.py` | **`fig3_mia_roc.png`**, `tableD1_risk_decomposition` | C3·C4 |
| D3 format comparison (Gate 1) | `src/d3_formats.py` | `tableD2_format_comparison` | C3 |
| D4 three-gate (audit+escrow+cost) | `src/d4_gates.py` | **`tableD3_three_gate_results`**, `tableD3_extras.json` | C4 |

Core exhibits for the paper: **Table 1** (policy matrix — already done, not in this
repo), **Fig 3** (`fig3_mia_roc.png`), **Table 3** (`tableD3_three_gate_results`).

## The three risks/gates, concretely

- **Risk A (ingress).** A crafted pickle whose `__reduce__` runs code at load
  time — payload is a **harmless marker only**. `picklescan` flags it; skops/ONNX
  remove the attack surface. Scanning *mitigates but cannot guarantee* (bypassable
  by novel gadget chains), which is the point: blocking the *format* is the wrong
  lever for either risk.
- **Risk B (egress).** Shadow-model MIA (Shokri-style) + loss-threshold MIA
  (Yeom-style) over two regimes of one task: **leaky** (high-capacity default fit
  to a small `N_TRAIN_MEMBERS` cohort) vs **safe** (regularised). Reported as
  **attack AUC** and **TPR @ low FPR**, plus train-test gap, as **mean ± 95% CI**
  over `SEED_LIST` (10 data seeds). Leakage is a property of the **model**, not the
  format. Formal DP is a separate **supplementary** ε-sweep (`src/supp_dp.py`).
- **Gate 1 (format).** pkl → skops / ONNX; measured: code-exec on load, language
  neutrality, portability, **inference parity** vs the pickle baseline.
- **Gate 2 (leakage audit).** Pre-export MIA; PASS iff shadow AUC ≤
  `MIA_AUC_PASS_THRESHOLD` (default 0.55); on BLOCK, mitigation = regularised
  re-train (the `safe` model). `THRESHOLD_SWEEP` reports decision stability over
  0.50–0.60 (Table D3b). Maps to All of Us's leakage-based logic — not a format rule.
- **Gate 3 (escrow + reproducibility + cost).** Sign+hash model + environment +
  data version; on-demand re-run must reproduce the data hash exactly and metrics
  within tolerance (1e-4). Cost compares warm-keeping vs escrow.

## Knobs the team will likely tune (`config.py`)
- `N_TRAIN_MEMBERS` — target (and per-shadow) training size; smaller ⇒ more overfit
  ⇒ stronger leaky MIA. Default 300 (seed-0: 24 positives) gives leaky shadow AUC
  ≈ 0.58 (CI > 0.55) vs safe ≈ 0.53 (CI < 0.55).
- `SEED_LIST` — data seeds for the mean ± 95% CI (default 10).
- `MIA_AUC_PASS_THRESHOLD` / `THRESHOLD_SWEEP` — Gate-2 cut and stability sweep.
- `DP_EPSILON_SWEEP`, model hyper-params in `src/utils.make_model`.
- `COST` — NCP list prices + CMS-VRDC seat ceiling; replace with contract rates.

## Design notes / honest caveats
- **Synthesiser:** transparent synthpop-style sequential CART (recommended 1st
  choice). `cart_synthesize()` is swappable for CTGAN as a sensitivity comparison.
- **leaky vs safe** are two regimes of the *same* task (library-default high-capacity
  fit to a small cohort vs a regularised fit), not a crippled model against a good
  one — the safe model matches or beats the leaky model's held-out utility.
- **Formal DP** (diffprivlib) is reported only in the supplementary ε-sweep; in this
  small-cohort tabular setting it collapses utility without beating regularisation on
  leakage. DP is randomised by design, seeded only for a stable figure.
- Determinism: numeric backends pinned to one thread + XGBoost `tree_method="exact"`
  ⇒ bit-exact reruns, **including the multi-seed shadow-MIA**.
- Numbers are **illustrative** (synthetic data, mechanism demonstration), not
  real-world risk estimates.

## AI-use disclosure (HIR requires this)
Note in the manuscript's Methods/Acknowledgments which AI tools assisted code or
text, with name/version/model/source, per HIR instructions for authors.
