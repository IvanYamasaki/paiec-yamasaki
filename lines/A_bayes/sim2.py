"""labeled-scope simulation: at checkpoint k, labeled = first k labels of every
pair in the same benchmark (or of a random subset of n_other pairs)."""
import numpy as np, pandas as pd, sim
from eval_local import BUDGETS, WEIGHTS

def score(predict_fn, traj, fold_fit=None, n_other=None, seed=0):
    """predict_fn(state, pair, own_labels, others) where others = list of (subject_id, item_id, y)."""
    obs = traj["obs"]; recs = []
    rng = np.random.default_rng(seed + 99)
    for b in sorted({p["bench"] for p in traj["pairs"]}):
        state = fold_fit(obs[obs.bench != b].reset_index(drop=True), traj["subjects"], traj["items"], b) if fold_fit else None
        P = [p for p in traj["pairs"] if p["bench"] == b]
        for j, p in enumerate(P):
            others = [q for q in P if q is not p]
            if n_other is not None and len(others) > n_other:
                others = [others[t] for t in rng.choice(len(others), n_other, replace=False)]
            r = {"bench": b, "subject_id": p["subject_id"]}
            for k in BUDGETS:
                oth = [(q["subject_id"], it, l) for q in others for it, l in q["acq"][:k]]
                pr = np.asarray(predict_fn(state, p, p["acq"][:k], oth), float)
                r[f"B{k}"] = float(np.mean((pr - p["y"]) ** 2))
            r["alc"] = sum(WEIGHTS[k] * r[f"B{k}"] for k in BUDGETS)
            recs.append(r)
    df = pd.DataFrame(recs)
    res = {"brier_alc": df.alc.mean(), **{f"B{k}": df[f"B{k}"].mean() for k in BUDGETS}}
    res["per_bench"] = df.groupby("bench").alc.mean().to_dict()
    return res, df
