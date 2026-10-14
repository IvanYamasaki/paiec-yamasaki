"""Ceilings: item difficulty oracle (Rasch fit on the full test bench, includes the test labels: NOT a model, only a bound)."""
import sys, numpy as np, pandas as pd, sim, rasch
from scipy.special import expit
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
T = sim.trajectories(seed); obs = T["obs"]
TH = np.linspace(-8, 8, 321); 
D = {}
for b in obs.bench.unique():
    th, d = rasch.fit_rasch(obs[obs.bench == b]); D.update(d.to_dict())
    print(b, "sd(d)=%.2f" % d.std())
def make(mode, m=-1.0, v=1.8):
    def pred(_, p, lab):
        de = np.array([D[i] for i in p["eval_items"]])
        if mode == "dshrink": de = de * 0.7
        dl = np.array([D[i] for i, _ in lab]); yl = np.array([l for _, l in lab])
        lp = -0.5 * (TH - m) ** 2 / v
        if len(lab):
            P = expit(TH[:, None] + dl[None, :])
            lp = lp + (yl * np.log(P) + (1 - yl) * np.log1p(-P)).sum(1)
        w = np.exp(lp - lp.max()); w /= w.sum()
        return (w[:, None] * expit(TH[:, None] + de[None, :])).sum(0)
    return pred
for mode in ("d", "dshrink"):
    r, df = sim.score(make(mode), T); print(mode, {k: round(v, 4) for k, v in r.items() if k != "per_bench"}, {k[:6]: round(v, 4) for k, v in r["per_bench"].items()})
