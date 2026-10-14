import sys, numpy as np, pandas as pd, sim, extdata, popmodel as pm, feats
from scipy.special import expit
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
T = sim.trajectories(seed)
EXT, EXS = extdata.load_ext(seed)
SUBS = dict(T["subjects"]); SUBS.update(EXS)
TH = np.linspace(-8, 8, 641); GR = expit(TH); LG, L1G = np.log(GR), np.log1p(-GR)
CFG = {}

def gpost(mean, var, lab):
    k = sum(l for _, l in lab); n = len(lab)
    lp = -0.5 * (TH - mean) ** 2 / var + k * LG + (n - k) * L1G
    w = np.exp(lp - lp.max()); return float((w * GR).sum() / w.sum())

def zscores(pt):
    pt = pt[pt.groupby("bench").y.transform("size") >= 5].copy()
    g = pt.groupby("bench").y
    pt["z"] = (pt.y - g.transform("mean")) / g.transform("std"); return pt

def name_score(pt, excl):
    q = pt[pt.bench != excl]
    return q.groupby(["name", "bench"]).z.mean().groupby("name").mean()

def design(pt_rows, ns, yc, use):
    X = []
    for sid, nm in zip(pt_rows.subject_id, pt_rows.name):
        f = feats.subj_feats(SUBS[sid], yc)
        s = ns.get(nm, np.nan)
        row = [0.0 if np.isnan(s) else s, float(not np.isnan(s))] + list(f)
        X.append(row)
    X = np.array(X)
    return X[:, use]

def fit(train, subjects, items, b):
    use = CFG["use"]
    tr = pd.concat([train[["subject_id", "item_id", "bench", "label"]], EXT])
    pt = zscores(pm.pair_table(tr, SUBS, 20))
    yc = float(np.nanmean([feats.year(SUBS[s]) for s in pt.subject_id]))
    Xs, Ys = [], []
    for bb in pt.bench.unique():
        tg = pt[pt.bench == bb]
        X = design(tg, name_score(pt, bb), yc, use); z = tg.z.to_numpy()
        Xs.append(X - X.mean(0)); Ys.append(z - z.mean())   # bench fixed effects
    X, Y = np.vstack(Xs), np.concatenate(Ys)
    lam = CFG.get("ridge", 5.0)
    beta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y) if len(use) else np.zeros(0)
    r = float(np.mean((Y - X @ beta) ** 2)) if len(use) else float(np.mean(Y ** 2))
    elig = pt[~pt.bench.isin(extdata.EXT)]
    bs = elig.groupby("bench").y.agg(["mean", "std", "size"])
    m = float(np.average(bs["mean"], weights=bs["size"]))
    s2mu = float(np.average((bs["mean"] - m) ** 2, weights=bs["size"]))
    sd = float(np.sqrt(np.average(bs["std"] ** 2, weights=bs["size"])))
    return dict(m=m, s2mu=max(s2mu, CFG.get("s2mu_min", 0.0)), sd=sd, beta=beta, r=r, ns=name_score(pt, None), yc=yc)

def pred(st, p, lab):
    sid = p["subject_id"]
    nm = pm.sname(T["subjects"][sid])
    if len(CFG["use"]):
        x = design(pd.DataFrame({"subject_id": [sid], "name": [nm]}), st["ns"], st["yc"], CFG["use"])[0]
        mean = st["m"] + st["sd"] * float(x @ st["beta"])
    else:
        mean = st["m"]
    var = CFG.get("vscale", 1.0) * (st["s2mu"] + st["sd"] ** 2 * st["r"])
    return np.full(len(p["y"]), gpost(mean, var, lab))

def run(tag, **cfg):
    CFG.clear(); CFG.update(cfg)
    r, df = sim.score(pred, T, fit)
    print(tag, {k: round(v, 4) for k, v in r.items() if k != "per_bench"}, {k[:6]: round(v, 4) for k, v in r["per_bench"].items()}, flush=True)
    return r, df

if __name__ == "__main__":
    # columns: 0 S_name, 1 known, 2 year, 3 year_missing, 4 reasoning, 5 agentic
    run("none", use=[])
    run("name", use=[0])
    run("year", use=[2, 3])
    run("year+reas+ag", use=[2, 3, 4, 5])
    run("all", use=[0, 2, 3, 4, 5])
    run("all s2mu>=.3", use=[0, 2, 3, 4, 5], s2mu_min=0.3)
