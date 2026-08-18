#!/usr/bin/env python3
"""
Extended cohort-size sweep (N up to 20,000) — back-pocket evidence for the
reviewer question "did you try much larger N?". NOT a main-text exhibit: the
paper's R1#3 answer is the complexity sweep (experiments_ext.py, Fig S1b);
this script exists only to answer, in a rebuttal if asked, what happens at
large N. Reuses experiments_ext primitives. Run from the poc/ directory.

Finding (CMS DE-SynPUF, 3 seeds): the fixed high-capacity default model keeps
fitting its training data perfectly (train AUC = 1.000) and leaks
(membership-inference AUC ~0.78-0.79) all the way to N = 20,000 REAL records;
the members are a random subset of the 116,352-row public benchmark, not
synthesized. The textbook decline in membership inference with dataset size
requires the generalization gap to close, which does not happen here because
the model over-memorizes throughout this range (cf. Zhang et al. 2017;
Feldman 2020). Results -> ../results/sweep_largeN.json.
"""
import os, json, statistics as st
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import experiments_ext as ex                      # K_SHADOW=16 unless POC_EXT_SMOKE
from config import FEATURES, OUTCOME
from src.d1_synthesize import get_population, membership_split
from src.utils import make_model, model_quality

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
SIZES = [100, 300, 1000, 3000, 10000, 20000]
SEEDS = [20260601 + i for i in range(3)]


def main():
    rows = []
    for N in SIZES:
        mia, tr, te, gap = [], [], [], []
        for seed in SEEDS:
            df = get_population(seed)
            members, pool, nonmembers = membership_split(df, seed, n_members=N)
            nm = ex._eval_nonmembers(nonmembers, seed)
            tgt = make_model("leaky", seed=seed + 7)
            tgt.fit(members[FEATURES], members[OUTCOME])
            mia.append(ex.shadow_mia(lambda s: make_model("leaky", seed=s),
                                     tgt, members, pool, nm, seed, N))
            trauc = model_quality(tgt, members[FEATURES], members[OUTCOME])["auc"]
            teauc = model_quality(tgt, nm[FEATURES], nm[OUTCOME])["auc"]
            tr.append(trauc); te.append(teauc); gap.append(trauc - teauc)
        r = dict(N=N, mia=round(st.mean(mia), 4),
                 mia_sd=round(st.pstdev(mia), 4) if len(mia) > 1 else 0.0,
                 train_auc=round(st.mean(tr), 4), test_auc=round(st.mean(te), 4),
                 gap=round(st.mean(gap), 4))
        rows.append(r)
        print(f"N={N:6d}  MIA={r['mia']:.3f}(+/-{r['mia_sd']:.3f})  "
              f"train={r['train_auc']:.3f} test={r['test_auc']:.3f} gap={r['gap']:.3f}",
              flush=True)
    out = os.path.join(RESULTS, "sweep_largeN.json")
    with open(out, "w") as f:
        json.dump({"seeds": SEEDS, "k_shadow": ex.K_SHADOW, "rows": rows}, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
