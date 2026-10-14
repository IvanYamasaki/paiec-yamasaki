"""TODO item 8: does J2 score better locally when more pairs per benchmark are co-sampled in `labeled`?

Compares J2 in the benchmark scope with 10, 20 and 40 pairs per benchmark, seeds 0-2.
The pair sets are nested, so besides the full-set score (which also changes which pairs are scored)
it reports the mean over the same pairs scored at 10 (LOBO folds only, stress folds excluded).
"""
import json
import os
import statistics as st

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
STRESS = ("stress:matharena", "stress:mmdocrag")


def path(m, s, kind):
    if m == 10:
        return f"{RES}/J_J2_b10_s{s}" + ("_pairs.csv" if kind == "csv" else ".json")
    return f"{HERE}/J2_b{m}_s{s}" + ("_pairs.csv" if kind == "csv" else ".json")


out = {}
for m in (10, 20, 40):
    full, same, npairs = [], [], None
    for s in range(3):
        d = json.load(open(path(m, s, "json")))
        full.append(d["brier_alc"])
        npairs = d["n_pairs"]
        base = pd.read_csv(path(10, s, "csv"))
        base = base[~base["fold"].isin(STRESS)][["fold", "subject_id"]]
        cur = pd.read_csv(path(m, s, "csv"))
        j = base.merge(cur, on=["fold", "subject_id"])
        assert len(j) == len(base), (m, s, len(j), len(base))
        same.append(j["alc"].mean())
    out[m] = dict(n_pairs=npairs, full=full, same31=same)
    print(f"bench/{m}: pairs scored {npairs}; full {st.mean(full):.4f} ± {st.stdev(full):.4f}; "
          f"same 31 pairs {st.mean(same):.4f} ± {st.stdev(same):.4f}; per seed same {[round(x, 4) for x in same]}")
json.dump(out, open(os.path.join(HERE, "item8_summary.json"), "w"), indent=1)
