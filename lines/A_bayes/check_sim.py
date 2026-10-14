import sys, numpy as np, sim
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
T = sim.trajectories(seed)
def fit(train, subjects, items, b):
    m = float(train["label"].mean())
    rates = train.groupby(["subject_id", "bench"])["label"].agg(["mean", "size"])
    rates = rates[rates["size"] >= 20]
    var = float(np.var(rates["mean"])) - float(np.mean(m * (1 - m) / rates["size"]))
    return m, float(np.clip(m * (1 - m) / max(var, 1e-6) - 1, 0.5, 100))
def pred(st, p, lab):
    m, k = st; s = sum(l for _, l in lab)
    return np.full(len(p["y"]), (k * m + s) / (k + len(lab)))
res, df = sim.score(pred, T, fit)
print(res)
