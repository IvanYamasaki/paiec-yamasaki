"""Population model for line A: logit pair rate y_sb = mu_b + a_name + e_sb.

Fitted on pair-level counts (training benchmarks only). a_name gets a ridge
prior centered on a metadata regression (release year, provider). Used to set
the prior of a new pair's logit rate."""
import numpy as np, pandas as pd


def sname(subj):
    n = subj.get("normalized_name", "")
    if not n:
        ex = subj.get("subject_features_extra", "")
        for part in ex.split(";"):
            if part.startswith("source_model_name="):
                n = "src:" + part.split("=", 1)[1].split(" (")[0].strip().lower()
    return n


def year(subj):
    d = subj.get("release_date", "")
    try:
        return int(d[:4]) + (int(d[5:7]) - 1) / 12
    except Exception:
        return np.nan


def pair_table(obs, subjects, minn=20):
    a = obs.groupby(["subject_id", "bench"]).label.agg(["sum", "size"]).reset_index()
    a = a[a["size"] >= minn].copy()
    k, n = a["sum"].to_numpy(float), a["size"].to_numpy(float)
    p = (k + 0.5) / (n + 1)
    a["y"] = np.log(p / (1 - p))
    a["v"] = 1 / ((n + 1) * p * (1 - p))
    a["name"] = a.subject_id.map(lambda s: sname(subjects[s]))
    return a


def fit(pt, subjects, iters=30, s2a=1.0, s2e=0.3):
    """Alternating GLS for mu_b (fixed), a_name (random). Returns dict."""
    benches = sorted(pt.bench.unique()); names = sorted(pt.name.unique())
    bi = pt.bench.map({b: i for i, b in enumerate(benches)}).to_numpy()
    ni = pt.name.map({n: i for i, n in enumerate(names)}).to_numpy()
    y, v = pt.y.to_numpy(), pt.v.to_numpy()
    mu = np.array([np.average(y[bi == j]) for j in range(len(benches))])
    a = np.zeros(len(names))
    for _ in range(iters):
        w = 1 / (v + s2e)
        r = y - a[ni]
        mu = np.array([np.average(r[bi == j], weights=w[bi == j]) for j in range(len(benches))])
        r = y - mu[bi]
        num = np.bincount(ni, w * r, len(names)); den = np.bincount(ni, w, len(names))
        a = num / (den + 1 / s2a)
        pa = 1 / (den + 1 / s2a)
        res = y - mu[bi] - a[ni]
        s2e = max(0.02, float(np.mean(res ** 2 - v)))
        s2a = max(0.02, float(np.mean(a ** 2 + pa)))
    return {"benches": benches, "mu": mu, "names": names, "a": a, "a_var": pa,
            "s2a": s2a, "s2e": s2e, "bench_n": np.bincount(bi, minlength=len(benches))}
