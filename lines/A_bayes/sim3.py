"""Run a real model dir (model.py) on cached trajectories, with labeled holding the
pair's labels plus the k labels of n_other other pairs of the same benchmark.
usage: sim3.py MODEL_DIR N_OTHER(all|int) SEEDS(0,1,2)"""
import sys, os, importlib.util, uuid, numpy as np, pandas as pd, sim
from eval_local import BUDGETS, WEIGHTS
import bench_data

def load(path):
    spec = importlib.util.spec_from_file_location("m_" + uuid.uuid4().hex[:6], os.path.join(path, "model.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

OVR = {}
def run(model_dir, n_other, seed, stress=False, max_pairs=None):
    """stress: test folds are matharena and mmdocrag (binarized); train = the other 5."""
    T = sim.trajectories(seed, all_benches=stress); obs = T["obs"]; S, I = T["subjects"], T["items"]
    mk = lambda s, i: bench_data.make_input(S, I, s, i, "")
    rng = np.random.default_rng(seed + 99); recs = []
    benches = ["matharena", "mmdocrag"] if stress else sorted({p["bench"] for p in T["pairs"]})
    for b in benches:
        M = load(model_dir)
        for k, v in OVR.items(): setattr(M, k, v)
        if hasattr(M, "local_fit"): M.local_fit(obs[obs.bench != b].reset_index(drop=True), S, I)
        P = [p for p in T["pairs"] if p["bench"] == b]
        if max_pairs and len(P) > max_pairs:
            P = [P[t] for t in np.random.default_rng(seed).choice(len(P), max_pairs, replace=False)]
        for p in P:
            others = [q for q in P if q is not p]
            if n_other is not None and len(others) > n_other:
                others = [others[t] for t in rng.choice(len(others), n_other, replace=False)]
            ev = [mk(p["subject_id"], i) for i in p["eval_items"]]
            r = {"bench": b}
            for k in BUDGETS:
                lab = [[mk(p["subject_id"], i), l] for i, l in p["acq"][:k]]
                lab += [[mk(q["subject_id"], i), l] for q in others for i, l in q["acq"][:k]]
                pr = np.array(M.predict_batch(ev, lab))
                r[f"B{k}"] = float(np.mean((pr - p["y"]) ** 2))
            r["alc"] = sum(WEIGHTS[k] * r[f"B{k}"] for k in BUDGETS); recs.append(r)
    df = pd.DataFrame(recs)
    return {"brier_alc": df.alc.mean(), **{f"B{k}": df[f"B{k}"].mean() for k in BUDGETS},
            "per_bench": df.groupby("bench").alc.mean().round(4).to_dict()}

if __name__ == "__main__":
    md, no = sys.argv[1], (None if sys.argv[2] == "all" else (0 if sys.argv[2] == "pair" else int(sys.argv[2])))
    stress = len(sys.argv) > 4 and sys.argv[4] == "stress"
    for kv in (sys.argv[5].split(",") if len(sys.argv) > 5 else []):
        k, v = kv.split("="); OVR[k] = None if v == "None" else float(v)
    rs = [run(md, no, int(s), stress, 30 if stress else None) for s in sys.argv[3].split(",")]
    a = [r["brier_alc"] for r in rs]
    print(md, OVR, "stress" if stress else "lobo4", "n_other", no, f"{np.mean(a):.4f} +- {np.std(a):.4f}", [round(x, 4) for x in a],
          " ".join(f"B{b}={np.mean([r[f'B{b}'] for r in rs]):.4f}" for b in BUDGETS),
          {k: round(np.mean([r['per_bench'][k] for r in rs]), 4) for k in rs[0]['per_bench']})
