import sys, numpy as np, importlib
res = {}
configs = {"none": dict(use=[]), "yra": dict(use=[2, 3, 4, 5]), "all": dict(use=[0, 2, 3, 4, 5]),
           "yra_s2mu.3": dict(use=[2, 3, 4, 5], s2mu_min=0.3), "yra_v1.2": dict(use=[2, 3, 4, 5], vscale=1.2),
           "yra_r1": dict(use=[2, 3, 4, 5], ridge=1.0), "yra_r20": dict(use=[2, 3, 4, 5], ridge=20.0)}
for seed in (0, 1, 2):
    sys.argv = ["x", str(seed)]
    import exp6; importlib.reload(exp6)
    for k, c in configs.items():
        exp6.CFG.clear(); exp6.CFG.update(c)
        r, _ = exp6.sim.score(exp6.pred, exp6.T, exp6.fit)
        res.setdefault(k, []).append(r)
for k, rs in res.items():
    a = [r["brier_alc"] for r in rs]
    print(f"{k:12s} {np.mean(a):.4f} +- {np.std(a):.4f}", " ".join(f"B{b}={np.mean([r[f'B{b}'] for r in rs]):.4f}" for b in (0, 1, 3, 7, 15, 31)))
