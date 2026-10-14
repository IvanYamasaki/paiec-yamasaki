import sys, numpy as np, pandas as pd, sim, extdata
from scipy.special import expit, logit
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
T = sim.trajectories(seed)
EXT, _ = extdata.load_ext(seed)
G = expit(np.linspace(-7, 7, 401))
LG, L1G = np.log(G), np.log1p(-G)

def pair_counts(train, minn=80):
    a = train.groupby(["subject_id", "bench"]).label.agg(["sum", "size"]).reset_index()
    return a[a["size"] >= minn]

def npmle(k, n, wpair, iters=500):
    LL = k[:, None] * LG + (n - k)[:, None] * L1G
    L = np.exp(LL - LL.max(1, keepdims=True))  # pairs x grid
    w = np.full(len(G), 1 / len(G))
    for _ in range(iters):
        P = L * w; P /= P.sum(1, keepdims=True)
        w = (wpair[:, None] * P).sum(0) / wpair.sum()
    return w

def post_mean(w, lab):
    k = sum(l for _, l in lab); n = len(lab)
    p = w * np.exp(k * LG + (n - k) * L1G); return float((p * G).sum() / p.sum())

def make_fit(use_ext, bench_w, minn=80):
    def fit(train, subjects, items, b):
        tr = pd.concat([train[["subject_id", "item_id", "bench", "label"]], EXT]) if use_ext else train
        pc = pair_counts(tr, minn)
        k, n = pc["sum"].to_numpy(float), pc["size"].to_numpy(float)
        if bench_w:
            cnt = pc.groupby("bench").bench.transform("size").to_numpy(float); wp = 1 / cnt
        else:
            wp = np.ones(len(pc))
        return npmle(k, n, wp)
    return fit

def pred(w, p, lab): return np.full(len(p["y"]), post_mean(w, lab))

def oracle(_, p, lab): return np.full(len(p["y"]), p["y"].mean())
#, {k: round(v, 4) for k, v in r.items() if k != "per_bench"})
for ue in (False, True):
    for bw in (False, True):
        for mn in (80, 20):
            r, df = sim.score(pred, T, make_fit(ue, bw, mn))
            print(f"npmle ext={ue} benchw={bw} minn={mn}", {k: round(v, 4) for k, v in r.items() if k != "per_bench"}, {k[:6]: round(v, 4) for k, v in r["per_bench"].items()})
