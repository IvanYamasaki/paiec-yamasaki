import sys, numpy as np, pandas as pd, sim, gp
from sklearn.feature_extraction.text import HashingVectorizer
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
T = sim.trajectories(seed); items = T["items"]
HV = HashingVectorizer(n_features=2**18, ngram_range=(1, 1), alternate_sign=False, norm="l2", binary=True)
_vec = {}
def vec(i):
    if i not in _vec: _vec[i] = HV.transform([items[i]["item_content"][:20000]])
    return _vec[i]
def cat(i): return items[i]["item_features"]

def make(m0, v0, se, sc, st):
    def pred(_, p, lab):
        E = p["eval_items"]; L = [i for i, _ in lab]; y = np.array([l for _, l in lab], float)
        css = np.full(len(E), v0 + sc + st + se)
        if not L:
            return gp.probit_mean(np.full(len(E), m0), css)
        cl = np.array([cat(i) for i in L]); ce = np.array([cat(i) for i in E])
        CLL = v0 + sc * (cl[:, None] == cl[None, :]) + se * np.eye(len(L))
        cLs = v0 + sc * (cl[:, None] == ce[None, :])
        if st > 0:
            import scipy.sparse as sp
            VL = sp.vstack([vec(i) for i in L]); VE = sp.vstack([vec(i) for i in E])
            CLL = CLL + st * (VL @ VL.T).toarray(); cLs = cLs + st * (VL @ VE.T).toarray()
        return gp.laplace_predict(m0, CLL, cLs, css, y)
    return pred
cfgs = [(-1.4, 1.5, 3.0, 0, 0), (-1.4, 1.5, 3.0, 1.0, 0), (-1.4, 1.5, 3.0, 0, 1.5), (-1.4, 1.5, 3.0, 0.7, 1.0), (-1.4, 1.5, 2.0, 0, 0), (-1.4, 1.5, 4.5, 0, 0)]
for c in cfgs:
    r, df = sim.score(make(*c), T)
    print(c, {k: round(v, 4) for k, v in r.items() if k != "per_bench"}, {k[:6]: round(v, 4) for k, v in r["per_bench"].items()}, flush=True)
