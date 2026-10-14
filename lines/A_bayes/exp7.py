"""Prior shape: normal vs empirical residuals (KDE) convolved with bench offset vs student-t."""
import sys, numpy as np, pandas as pd, importlib
sys.path.insert(0, "v2")
import model as M2
from scipy.special import expit
from scipy.stats import t as tdist
import sim, extdata
TH = M2.TH; GR = M2.GR; LG, L1G = M2.LG, M2.L1G
CFG = {}

def fit(train, subjects, items, b, EXT=None, EXS=None):
    subs = dict(subjects); subs.update(EXS)
    obs = pd.concat([train[["subject_id", "item_id", "bench", "label"]], EXT], ignore_index=True)
    P = M2.fit_params(obs, subs, sorted(train.bench.unique()))
    # residuals within binary training benches (logit scale)
    pt = M2._pair_table(obs, subs)
    pt = pt[pt.bench.isin(train.bench.unique()) | CFG.get("res_all", False)]
    X = np.array([M2.features(subs[s], P["ns"], P["yc"]) for s in pt.subject_id])
    mu_b = pt.groupby("bench").y.transform("mean").to_numpy()
    sd_b = pt.groupby("bench").y.transform("std").to_numpy()
    res = (pt.y.to_numpy() - mu_b) / sd_b - (X - X.mean(0)) @ np.asarray(P["beta"])
    P["res"] = res * P["sd"]
    return P

def prior_logpdf(P, mean):
    kind = CFG.get("kind", "normal")
    var = P["s2mu"] + P["sd"] ** 2 * P["r"]
    if kind == "normal":
        return -0.5 * (TH - mean) ** 2 / var
    if kind == "t":
        return tdist.logpdf((TH - mean) / np.sqrt(var), CFG.get("df", 5))
    if kind == "kde":
        bw2 = P["s2mu"] + CFG.get("bw", 0.3) ** 2
        r = P["res"] - P["res"].mean()
        d = (TH[:, None] - mean - r[None, :]) ** 2
        return np.log(np.exp(-0.5 * d / bw2).mean(1) + 1e-300)

def pred(P, p, lab):
    s = sim_subjects[p["subject_id"]]
    x = M2.features(s, P["ns"], P["yc"])
    mean = P["m"] + P["sd"] * float(x @ np.asarray(P["beta"]))
    k = sum(l for _, l in lab); n = len(lab)
    lp = prior_logpdf(P, mean) + k * LG + (n - k) * L1G
    w = np.exp(lp - lp.max())
    return np.full(len(p["y"]), float((w * GR).sum() / w.sum()))

configs = {"normal": dict(kind="normal"), "t5": dict(kind="t", df=5), "t3": dict(kind="t", df=3),
           "kde.3": dict(kind="kde", bw=0.3), "kde.6": dict(kind="kde", bw=0.6), "kde.3all": dict(kind="kde", bw=0.3, res_all=True)}
res = {}
for seed in (0, 1, 2):
    T = sim.trajectories(seed); EXT, EXS = extdata.load_ext(0); sim_subjects = T["subjects"]
    for k, c in configs.items():
        CFG.clear(); CFG.update(c)
        r, _ = sim.score(pred, T, lambda *a: fit(*a, EXT=EXT, EXS=EXS))
        res.setdefault(k, []).append(r)
for k, rs in res.items():
    a = [r["brier_alc"] for r in rs]
    print(f"{k:10s} {np.mean(a):.4f} +- {np.std(a):.4f}", " ".join(f"B{b}={np.mean([r[f'B{b}'] for r in rs]):.4f}" for b in (0, 1, 3, 7, 15, 31)), {kk[:6]: round(np.mean([r['per_bench'][kk] for r in rs]), 4) for kk in rs[0]['per_bench']})
