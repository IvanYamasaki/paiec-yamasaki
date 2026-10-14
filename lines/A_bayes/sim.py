"""Fast replica of eval_local.py (default random acquisition) for line A.

Trajectories under the default policy do not depend on the model, so we cache,
per seed and pair: eval item ids + labels, acquired item ids + labels in order.
Then any model f(pair_ctx, eval_items, labeled) can be scored quickly.
Must match eval_local.run_pair exactly (checked against beta_pair).
"""
import os, sys, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import bench_data
from eval_local import eligible_pairs, pair_rng, BUDGETS, WEIGHTS, MAX_LABELS

CACHE = os.path.join(HERE, "cache")


def trajectories(seed, data=os.path.join(ROOT, "data", "measurement-db"), all_benches=False):
    """all_benches: also matharena/mmdocrag binarized (stress folds)."""
    f = os.path.join(CACHE, f"traj_seed{seed}{'_all' if all_benches else ''}.pkl")
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    obs, subjects, items, meta = bench_data.load(data, seed=seed, eligible_only=not all_benches)
    pairs = eligible_pairs(obs, 80)
    grouped = {k: g for k, g in obs.groupby(["subject_id", "bench"])}
    out = []
    for (sid, b) in sorted(pairs, key=lambda x: (x[1], x[0])):
        rng = pair_rng(seed, sid, b)
        rows = grouped[(sid, b)].sample(frac=1.0, random_state=int(rng.integers(2**31))).reset_index(drop=True)
        half = len(rows) // 2
        acq, ev = rows.iloc[:half], rows.iloc[half:]
        acq_list = list(acq.itertuples())
        lab = []
        for i, r in enumerate(acq_list):
            if len(lab) >= MAX_LABELS:
                break
            if rng.random() < min(1.0, (MAX_LABELS - len(lab)) / (len(acq_list) - i)):
                lab.append((r.item_id, int(r.label)))
        out.append({"bench": b, "subject_id": sid, "eval_items": ev["item_id"].tolist(),
                    "y": ev["label"].to_numpy(), "acq": lab})
    os.makedirs(CACHE, exist_ok=True)
    res = {"pairs": out, "obs": obs, "subjects": subjects, "items": items}
    pickle.dump(res, open(f, "wb"))
    return res


def score(predict_fn, traj, fold_fit=None, benches=None):
    """predict_fn(model_state, pair, labeled_list) -> array of probs for pair['eval_items'].
    fold_fit(train_obs, subjects, items, test_bench) -> model_state."""
    import pandas as pd
    obs = traj["obs"]
    recs = []
    for b in sorted({p["bench"] for p in traj["pairs"]}):
        if benches and b not in benches:
            continue
        state = fold_fit(obs[obs.bench != b].reset_index(drop=True), traj["subjects"], traj["items"], b) if fold_fit else None
        for p in traj["pairs"]:
            if p["bench"] != b:
                continue
            r = {"bench": b, "subject_id": p["subject_id"]}
            for k in BUDGETS:
                pr = np.asarray(predict_fn(state, p, p["acq"][:k]), float)
                r[f"B{k}"] = float(np.mean((pr - p["y"]) ** 2))
            r["alc"] = sum(WEIGHTS[k] * r[f"B{k}"] for k in BUDGETS)
            recs.append(r)
    df = pd.DataFrame(recs)
    res = {"brier_alc": df.alc.mean(), **{f"B{k}": df[f"B{k}"].mean() for k in BUDGETS}}
    res["per_bench"] = df.groupby("bench").alc.mean().to_dict()
    return res, df
