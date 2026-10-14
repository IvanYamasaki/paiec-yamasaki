import sys, numpy as np, pandas as pd, sim, extdata, popmodel as pm
from scipy.special import expit
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
T = sim.trajectories(seed)
EXT, EXS = extdata.load_ext(seed)
SUBS = dict(T["subjects"]); SUBS.update(EXS)
TH = np.linspace(-8, 8, 641); GR = expit(TH); LG, L1G = np.log(GR), np.log1p(-GR)

def gpost(mean, var, lab):
    k = sum(l for _, l in lab); n = len(lab)
    lp = -0.5 * (TH - mean) ** 2 / var + k * LG + (n - k) * L1G
    w = np.exp(lp - lp.max()); return float((w * GR).sum() / w.sum())

def zscores(pt):
    pt = pt[pt.groupby("bench").y.transform("size") >= 5]
    g = pt.groupby("bench").y
    pt = pt.copy(); pt["z"] = (pt.y - g.transform("mean")) / g.transform("std")
    return pt

def name_score(ptz, exclude_bench):
    q = ptz[ptz.bench != exclude_bench]
    nb = q.groupby(["name", "bench"]).z.mean().reset_index()   # one value per (name, bench)
    return nb.groupby("name").z.agg(["mean", "size"])

def fit(train, subjects, items, b, use_ext=True, minn=20):
    tr = pd.concat([train[["subject_id", "item_id", "bench", "label"]], EXT]) if use_ext else train
    pt = zscores(pm.pair_table(tr, SUBS, minn))
    elig = pt[~pt.bench.isin(extdata.EXT)]
    # inner LOBO calibration of t = beta * S over all training benches as targets
    X, Y = [], []
    for bb in pt.bench.unique():
        ns = name_score(pt, bb)
        tgt = pt[pt.bench == bb]
        S = tgt.name.map(ns["mean"]); C = tgt.name.map(ns["size"]).fillna(0)
        k = S.notna().to_numpy(); s_ = S.fillna(0).to_numpy().copy(); z_ = tgt.z.to_numpy().copy()
        if CFG.get("fe", True) and k.sum() >= 3:
            s_[k] -= s_[k].mean()
        X.append(np.c_[s_, k]); Y.append(z_)
    X, Y = np.vstack(X), np.concatenate(Y)
    kn = X[:, 1] > 0
    beta = float((X[kn, 0] @ Y[kn]) / (X[kn, 0] @ X[kn, 0]))
    r_kn = float(np.mean((Y[kn] - beta * X[kn, 0]) ** 2)); r_un = float(np.mean(Y[~kn] ** 2))
    # benchmark-level stats from eligible (binary) training benches, pair-weighted
    bs = elig.groupby("bench").y.agg(["mean", "std", "size"])
    m = float(np.average(bs["mean"], weights=bs["size"]))
    s2mu = float(np.average((bs["mean"] - m) ** 2, weights=bs["size"]))
    sd = float(np.sqrt(np.average(bs["std"].fillna(bs["std"].mean()) ** 2, weights=bs["size"])))
    ns = name_score(pt, None)
    return dict(m=m, s2mu=s2mu, sd=sd, beta=beta, r_kn=r_kn, r_un=r_un, ns=ns)

CFG = {}
def pred(st, p, lab):
    nm = pm.sname(T["subjects"][p["subject_id"]])
    sc = CFG.get("bscale", 1.0)
    if CFG.get("useA", True) and nm in st["ns"].index:
        mean = st["m"] + st["sd"] * st["beta"] * sc * st["ns"].loc[nm, "mean"]; r = st["r_kn"]
    else:
        mean = st["m"]; r = st["r_un"]
    var = CFG.get("vscale", 1.0) * (st["s2mu"] + st["sd"] ** 2 * r)
    return np.full(len(p["y"]), gpost(mean, var, lab))

def run(tag, fitkw={}, **cfg):
    CFG.clear(); CFG.update(cfg)
    r, df = sim.score(pred, T, lambda *a: fit(*a, **fitkw))
    print(tag, {k: round(v, 4) for k, v in r.items() if k != "per_bench"}, {k[:6]: round(v, 4) for k, v in r["per_bench"].items()}, flush=True)
    return r

if __name__ == "__main__":
    for bb in ["multi_swebench", "real_webagents", "researchcodebench"]:
        st = fit(T["obs"][T["obs"].bench != bb], None, None, bb)
        print(bb, {k: (round(v, 3) if isinstance(v, float) else '') for k, v in st.items()})
    run("noA", useA=False)
    run("A ext nofe", fe=False)
    run("A ext", useA=True)
    run("A noext", {"use_ext": False}, useA=True)
    run("A ext b.7", bscale=0.7)
    run("A ext b1.3", bscale=1.3)
    run("A ext v.8", vscale=0.8)
    run("A ext v1.25", vscale=1.25)
