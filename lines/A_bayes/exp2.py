import sys, numpy as np, pandas as pd, sim, extdata, popmodel as pm
from scipy.special import expit
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
T = sim.trajectories(seed)
EXT, EXS = extdata.load_ext(seed)
TH = np.linspace(-8, 8, 641); GR = expit(TH); LG, L1G = np.log(GR), np.log1p(-GR)

def gpost(mean, var, lab):
    k = sum(l for _, l in lab); n = len(lab)
    lp = -0.5 * (TH - mean) ** 2 / var + k * LG + (n - k) * L1G
    w = np.exp(lp - lp.max()); return float((w * GR).sum() / w.sum())

def make_fit(use_ext_a, lam, vscale, mean_mode):
    def fit(train, subjects, items, b):
        subs = dict(subjects); subs.update(EXS)
        tr = pd.concat([train[["subject_id", "item_id", "bench", "label"]], EXT]) if use_ext_a else train
        pt = pm.pair_table(tr, subs)
        M = pm.fit(pt, subs)
        elig = [i for i, bb in enumerate(M["benches"]) if bb not in extdata.EXT]
        mus = M["mu"][elig]; nb = M["bench_n"][elig]
        if mean_mode == "pairw":
            m = np.average(mus, weights=nb)
        else:
            m = mus.mean()
        s2mu = max(0.1, float(np.average((mus - m) ** 2, weights=nb)))
        amap = dict(zip(M["names"], zip(M["a"], M["a_var"])))
        return dict(m=m, s2mu=s2mu, amap=amap, M=M, subs=subs)
    return fit

def make_pred(lam, vscale, use_a=True):
    def pred(st, p, lab):
        nm = pm.sname(T["subjects"][p["subject_id"]])
        M = st["M"]
        if use_a and nm in st["amap"]:
            a, av = st["amap"][nm]
        else:
            a, av = 0.0, M["s2a"]
        mean = st["m"] + lam * a
        var = vscale * (st["s2mu"] + lam ** 2 * av + M["s2e"])
        return np.full(len(p["y"]), gpost(mean, var, lab))
    return pred

for ue in (False, True):
    for mm in ("pairw",):
        fit = make_fit(ue, 1, 1, mm)
        for lam, vs, ua in [(1, 1, False), (1, 1, True), (1, 0.7, True), (1, 1.4, True), (0.7, 1, True)]:
            r, df = sim.score(make_pred(lam, vs, ua), T, fit)
            print(f"ext={ue} {mm} lam={lam} vs={vs} useA={ua}", {k: round(v, 4) for k, v in r.items() if k != "per_bench"}, {k[:6]: round(v, 4) for k, v in r["per_bench"].items()})
