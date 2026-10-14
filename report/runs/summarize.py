"""Table 2 rows for the runs added on 2026-10-13/14 (seed 0 of J3/J4 pair and v6 bench/10 seeds 0, 2 come from results/)."""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")

SETS = {
    "J3 pair": [f"{RES}/J_J3_hib_pair_s0.json", f"{HERE}/J3_pair_s1.json", f"{HERE}/J3_pair_s2.json"],
    "J4 pair": [f"{RES}/J_J4_rasch_pair_s0.json", f"{HERE}/J4_pair_s1.json", f"{HERE}/J4_pair_s2.json"],
    "J3 bench/10": [f"{HERE}/J3_b10_s{s}.json" for s in range(3)],
    "J4 bench/10": [f"{HERE}/J4_b10_s{s}.json" for s in range(3)],
    "v6 bench/10": [f"{RES}/A_v6n1_b10_seed0.json", f"{HERE}/v6_b10_s1.json", f"{RES}/A_v6n1_b10_seed2.json"],
    "J2 bench/10": [f"{RES}/J_J2_b10_s{s}.json" for s in range(3)],
    "J2 bench/20": [f"{HERE}/J2_b20_s{s}.json" for s in range(3)],
    "J2 bench/40": [f"{HERE}/J2_b40_s{s}.json" for s in range(3)],
}


def fmt(v):
    return f"{st.mean(v):.4f} ± {st.stdev(v):.4f}" if len(v) > 1 else f"{v[0]:.4f} (1 seed)"


for name, files in SETS.items():
    ds = [json.load(open(f)) for f in files if os.path.exists(f)]
    if not ds:
        print(name, "no runs yet")
        continue
    alc = [d["brier_alc"] for d in ds]
    b0 = [d["B0"] for d in ds]
    sm = [d["stress"]["folds"]["stress:matharena"]["brier_alc"] for d in ds]
    sd = [d["stress"]["folds"]["stress:mmdocrag"]["brier_alc"] for d in ds]
    print(f"{name}: n_seeds={len(ds)} pairs={ds[0]['n_pairs']} alc={fmt(alc)} B0={st.mean(b0):.3f} "
          f"stress math={st.mean(sm):.4f} mmdoc={st.mean(sd):.4f} strict={all(d['strict_ok'] for d in ds)} "
          f"seeds={[d['seed'] for d in ds]} per_seed={[round(x, 4) for x in alc]}")
