"""
Inter-coder reliability for the Table 1 policy audit (rec 2 / R1#7).

Computes Cohen's kappa (overall + per dimension) from TWO independent human
codings of the 6-platform x 5-dimension matrix (30 cells). No values are
fabricated: the script reads the two coders' actual category assignments from
`policy_audit_codings.csv`. If that file is absent it writes a blank template
and exits, so the humans fill it in before any kappa is produced.

Usage:
    python kappa_policy_audit.py            # compute from policy_audit_codings.csv
    python kappa_policy_audit.py --template # (re)write the blank template

Coders: first coder = H.Y., second coder (independent, blind) = J.L. (the second coder).
The second coder must code from the S5 primary documents WITHOUT viewing Table 1
or the first coder's assignments, or the kappa is invalid.
"""
from __future__ import annotations
import csv
import os
import sys

# numpy / scikit-learn are imported lazily inside the compute paths so that
# writing the blank template works even before the analysis env is installed.

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "policy_audit_codings.csv")            # merged (Yu's copy; both coders)
SECOND_CODER_CSV = os.path.join(HERE, "policy_audit_coder_JL.csv")  # 2nd coder's blank sheet (coder_JL only)

PLATFORMS = [
    "UK Biobank RAP",
    "All of Us",
    "EHDS Secure Processing Environment",
    "CODA",
    "K-CURE",
    "Mayo Clinic Platform",
]
DIMENSIONS = ["D1", "D2", "D3", "D4", "D5"]

# Closed category lists per dimension (codebook; see S1 File S5b). Used only to
# validate that entered labels are in-vocabulary -- NOT to auto-assign codes.
CATEGORIES = {
    "D1": ["aggregate stats/tables/figures only", "+ de-identified result files",
           "+ conditional model release", "finished algorithm only", "not stated"],
    "D2": ["not addressed", "conditional - leakage basis",
           "algorithm-out (platform-internal)", "format/reproducibility basis"],
    "D3": ["automated egress alerts", "human output-checking committee",
           "platform-controlled build & validate", "SDC rules"],
    "D4": ["persists", "time-limited / decommissioned",
           "workspace persists (with credits)"],
    "D5": ["access fee / credits", "subscription / seat",
           "partnership / contract", "free"],
}


def write_template() -> None:
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["platform", "dimension", "coder_HY", "coder_JL", "note"])
        for p in PLATFORMS:
            for d in DIMENSIONS:
                w.writerow([p, d, "", "", ""])
    print(f"Wrote blank template: {CSV_PATH}")
    print("Fill coder_HY and coder_JL with one codebook category each (S1 File S5b),")
    print("code independently/blind, then re-run this script. Leave a cell blank to")
    print("exclude an as-yet-uncoded pair (kappa uses complete pairs only).")


def _norm(s: str) -> str:
    """Normalize a category label so trivial variants (case, spacing, dash type)
    count as the same code when comparing the two coders -- prevents a spurious
    disagreement from '–' vs '-' or stray whitespace."""
    s = str(s).strip().lower().replace("–", "-").replace("—", "-")
    return " ".join(s.split())


def write_second_coder_sheet() -> None:
    """Second-coder sheet: coder_JL only, NO coder_HY (blind independence).
    Give this file to the second coder; never hand over a sheet containing the
    first coder's (coder_HY) assignments -- that would invalidate the kappa."""
    with open(SECOND_CODER_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["platform", "dimension", "coder_JL",
                    "confidence", "evidence", "source", "note"])
        for p in PLATFORMS:
            for d in DIMENSIONS:
                w.writerow([p, d, "", "", "", "", ""])
    print(f"Wrote second-coder sheet (coder_JL only, no coder_HY): {SECOND_CODER_CSV}")
    print("Hand THIS file to the second coder. Keep policy_audit_codings.csv")
    print("(which holds coder_HY) private until both codings are merged.")


def _kappa_ci(a, b, n_boot=2000, seed=20260813):
    """Bootstrap 95% CI for Cohen's kappa over paired items."""
    import numpy as np
    from sklearn.metrics import cohen_kappa_score
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) < 2 or len(set(a.tolist()) | set(b.tolist())) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = []
    idx = np.arange(len(a))
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        try:
            stats.append(cohen_kappa_score(a[s], b[s]))
        except Exception:
            continue
    if not stats:
        return (float("nan"), float("nan"))
    return (float(np.nanpercentile(stats, 2.5)),
            float(np.nanpercentile(stats, 97.5)))


def main() -> None:
    if "--second-coder-sheet" in sys.argv:
        write_second_coder_sheet()
        return

    if "--template" in sys.argv or not os.path.exists(CSV_PATH):
        if not os.path.exists(CSV_PATH):
            print("No coding file found.")
        write_template()
        return

    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            hy = (r.get("coder_HY") or "").strip()
            jl = (r.get("coder_JL") or "").strip()
            if hy and jl:                      # complete pairs only
                rows.append((r["dimension"].strip(), hy, jl))

    if len(rows) < 2:
        print("Fewer than 2 complete coder pairs -- coding not finished. "
              "Kappa withheld (no fabricated value).")
        return

    from sklearn.metrics import cohen_kappa_score
    coded = len(rows)
    print(f"Complete coded cells: {coded}/30\n")
    print(f"{'Scope':<34}{'Cohen kappa':>12}{'95% CI':>22}{'n':>5}")
    print("-" * 73)

    hy_all = [_norm(r[1]) for r in rows]
    jl_all = [_norm(r[2]) for r in rows]
    k = cohen_kappa_score(hy_all, jl_all)
    lo, hi = _kappa_ci(hy_all, jl_all)
    print(f"{'Overall (all dimensions)':<34}{k:>12.3f}"
          f"{f'[{lo:.3f}, {hi:.3f}]':>22}{coded:>5}")

    for d in DIMENSIONS:
        hy = [_norm(r[1]) for r in rows if r[0] == d]
        jl = [_norm(r[2]) for r in rows if r[0] == d]
        if len(hy) >= 2 and len(set(hy) | set(jl)) >= 2:
            kd = cohen_kappa_score(hy, jl)
            print(f"{d:<34}{kd:>12.3f}{'':>22}{len(hy):>5}")
        else:
            # kappa undefined when all labels identical or <2 pairs
            agree = sum(int(x == y) for x, y in zip(hy, jl))
            print(f"{d:<34}{'n/a':>12}{f'agree {agree}/{len(hy)}':>22}{len(hy):>5}")

    print("\nPaste the Overall/per-dimension values into S1 File S5b (replace [PENDING]).")


if __name__ == "__main__":
    main()
