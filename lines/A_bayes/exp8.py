"""Heteroscedastic prior variance: residual var by (name known, year known)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "v2")
import model as M2
import sim, extdata
CFG = {}

def fit(train, subjects, items, b, EXT=None, EXS=None):
    subs = dict(subjects); subs.update(EXS)
    obs = pd.concat([train[["subject_id", "item_id", "bench", "label"]], EXT], ignore_index=True)
    P = M2.fit_params(obs, subs, sorted(train.bench.unique()))
    pt = M2._pair_table(obs, subs)
    G, R = [], []
    for bb in pt.bench.unique():
        tg = pt[pt.bench == bb]; ns = M2._name_scores(pt, bb)
        X = np.array([M2.features(subs[s], ns, P["yc"]) for s in tg.subject_id])
        z = tg.z.to_numpy()
        res = (z - z.mean()) - (X - X.mean(0)) @ np.asarray(P["beta"])
        g = [grp(subs[s], ns) for s in tg.subject_id]
        G += g; R += list(res)
    G, R = np.array(G), np.array(R)
    P["rg"] = {g: float(np.mean(R[G == g] ** 2)) for g in set(G)}
    return P

def grp(s, ns):
    if not CFG.get("het"): return 0
    return int(M2.sname(s) in ns) * 2 + int(not np.isnan(M2.year(s)))

def pred(P, p, lab):
    s = SUBJ[p["subject_id"]]
    x = M2.features(s, P["ns"], P["yc"])
    mean = P["m"] + P["sd"] * float(x @ np.asarray(P["beta"]))
    r = P["rg"].get(grp(s, P["ns"]), P["r"])
    var = P["s2mu"] + P["sd"] ** 2 * r
    k = sum(l for _, l in lab); n = len(lab)
    return np.full(len(p["y"]), M2._post_mean(mean, var, k, n))

res = {}
for seed in (0, 1, 2):
    T = sim.trajectories(seed); EXT, EXS = extdata.load_ext(0); SUBJ = T["subjects"]
    for k, c in {"homo": {}, "het": {"het": True}}.items():
        CFG.clear(); CFG.update(c)
        r, _ = sim.score(pred, T, lambda *a: fit(*a, EXT=EXT, EXS=EXS)); res.setdefault(k, []).append(r)
for k, rs in res.items():
    a = [r["brier_alc"] for r in rs]
    print(f"{k:6s} {np.mean(a):.4f} +- {np.std(a):.4f}", " ".join(f"B{b}={np.mean([r[f'B{b}'] for r in rs]):.4f}" for b in (0, 1, 3, 7, 15, 31)))
