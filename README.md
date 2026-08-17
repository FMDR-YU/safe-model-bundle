# safe-model-bundle

Reference implementation for **"Safe and Reproducible Model Export from Closed Clinical Data Platforms."**
A *Safe Model Bundle* is a checkable export contract for a trained model that composes three independent gates:

1. **Gate 1 — format:** convert the model to a code-free serialization format (skops / ONNX) that carries no executable payload, with lossless inference parity against pickle.
2. **Gate 2 — leakage audit:** release only if a membership-inference audit stays at or below a threshold (a lower bound against the audited attack).
3. **Gate 3 — escrow:** bind the model to its environment (container, dependency lock, seed, data version) under a signed hash, so a revision re-run reproduces the analysis.

`run_all.py` runs the whole pipeline end-to-end on a **public benchmark** and regenerates every exhibit deterministically. No real patient data are used.

## Data (required before running)

The default substrate is the **CMS DE-SynPUF** 2008 Beneficiary Summary, a public, freely downloadable synthetic-claims file (no real beneficiaries).

1. Download **DE-SynPUF Sample 1 — 2008 Beneficiary Summary** from CMS:
   <https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf>
2. Unzip and place the CSV at **`data/cdm/synpuf/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv`** (repo-relative).
   - Alternatively point the loader elsewhere with `POC_DATA_ROOT=/path/to/cdm` (the loader expects `<root>/synpuf/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv`).
3. (Optional) The OHDSI `Synthea27Nj` OMOP dataset can be placed at `data/cdm/Synthea27Nj/` for the `POC_SUBSTRATE=synthea` smoke test — note this is a 27-person toy set, not an analysis substrate.

Data files are **not** committed to this repository.

## Run

```bash
cd poc
pip install -r requirements.txt          # Python 3.12 recommended
python run_all.py                        # POC_SUBSTRATE defaults to "synpuf"
```

Substrate is selected with the `POC_SUBSTRATE` env var: `synpuf` (default, ~116k claims), `synthea` (OMOP smoke test), or `koges` (legacy self-authored synthetic DGP, no external data needed).

## Outputs

`run_all.py` writes exhibits to `poc/outputs/`; the canonical shared copies are in **`results/`**:
`summary.json`, `fig3_mia_roc.png`, `figS1_dp_sweep.png`, `figS2_cost.png`, `escrow_manifest.json`, and the tables `tableD1`–`tableD3b`, `tableS1`.

## Policy-audit inter-coder reliability

`poc/kappa_policy_audit.py` computes Cohen's κ (overall and per dimension, with a bootstrap CI) from two independent codings of the six-platform policy matrix:

```bash
python poc/kappa_policy_audit.py --second-coder-sheet   # write the blank second-coder sheet
python poc/kappa_policy_audit.py                  # compute κ once both codings are filled
```

The second coder fills `poc/policy_audit_coder_JL.csv` blind; the first coder's sheet is kept private until the codings are merged (so κ stays valid).

## Reproducibility notes

All randomness derives from `config.MASTER_SEED`; numeric backends are pinned to a single thread and tree models use an exact split-finder, so results are bit-reproducible. On Windows, if the newest wheels fail to load, pin `pandas==2.2.3`, `numpy==2.2.6`, `scikit-learn==1.6.1`; `scipy<1.15` is required for `diffprivlib` compatibility (see `requirements.txt`).

## License

Code: MIT. Documentation: CC-BY-4.0.
