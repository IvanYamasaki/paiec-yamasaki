import sys, numpy as np, pandas as pd
sys.path.insert(0, "v2")
import model as M2
import sim, sim2, extdata, joint, rasch
CFG = {}

def fit(train, subjects, items, b, EXT=None, EXS=None):
    subs = dict(subjects); subs.update(EXS)
    obs = pd.concat([train[["subject_id", "item_id", "bench", "label"]], EXT], ignore_index=True)
    P = M2.fit_params(obs, subs, sorted(train.bench.unique()))
    dv = []
    for bb, g in train.groupby("bench"):
        if g.subject_id.nunique() < 5: continue
        th, d = rasch.fit_rasch(g)
        cnt = g.groupby("item_id").size().reindex(d.index)
        dv.append(d[cnt >= 5].var())
    P["sd2"] = float(np.mean(dv)) * CFG.get("sd2_scale", 1.0)
    return P

def pred(P, p, own, others):
    s = SUBJ[p["subject_id"]]
    k = sum(l for _, l in own); n = len(own)
    mean, var = M2._prior(s) if False else prior(P, s)
    if not others or CFG.get("v2only"):
        return np.full(len(p["y"]), M2._post_mean(mean, var, k, n))
    c = np.sqrt(1 + np.pi * P["sd2"] / 8) * CFG.get("cfac", 1.0)
    if CFG.get("hybrid"):
        return hybrid(P, p, own, others, c)
    subj_ids = [p["subject_id"]] + sorted({o[0] for o in others})
    sidx = {x: j for j, x in enumerate(subj_ids)}
    labs = [(p["subject_id"], it, l) for it, l in own] + list(others)
    item_ids = sorted({it for _, it, _ in labs} | set(p["eval_items"]))
    iidx = {x: j for j, x in enumerate(item_ids)}
    a = np.array([c * (prior(P, SUBJ[x])[0] - P["m"]) for x in subj_ids])
    tau2 = c ** 2 * P["sd"] ** 2 * P["r"]
    th, vth, d, vd, mu, vmu = joint.fit(np.array([sidx[x] for x, _, _ in labs]), np.array([iidx[i] for _, i, _ in labs]),
                                        np.array([l for _, _, l in labs]), a, c * P["m"], c ** 2 * P["s2mu"], tau2, P["sd2"], len(item_ids))
    e = np.array([iidx[i] for i in p["eval_items"]])
    return joint.expected_sigmoid(th[0] + d[e], vth[0] + vd[e])

TH = np.linspace(-14, 14, 561)
def hybrid_cat(P, p, own, others, c):
    rho = CFG["rho"]
    subj_ids = sorted({o[0] for o in others}); sidx = {x: j for j, x in enumerate(subj_ids)}
    item_ids = sorted({it for _, it, _ in others}); iidx = {x: j for j, x in enumerate(item_ids)}
    cats = {}
    allc = [ITEMS[i]["item_features"] for i in item_ids]
    for cc in allc:
        if cc: cats.setdefault(cc, len(cats))
    coi = np.array([cats.get(cc, -1) if cc else -1 for cc in allc])
    a = np.array([c * (prior(P, SUBJ[x])[0] - P["m"]) for x in subj_ids])
    tau2 = c ** 2 * P["sd"] ** 2 * P["r"]
    th, vth, d, vd, mu, vmu, g, vg, ve = joint.fit_cat(np.array([sidx[x] for x, _, _ in others]), np.array([iidx[i] for _, i, _ in others]),
        np.array([l for _, _, l in others], float), a, c * P["m"], c ** 2 * P["s2mu"], tau2, P["sd2"], len(item_ids), coi, len(cats), rho)
    sd2 = P["sd2"]
    def dstat(i):
        j = iidx.get(i)
        if j is not None: return (d[j], vd[j])
        cc = ITEMS[i]["item_features"]
        k = cats.get(cc) if cc else None
        return (0.0, sd2) if k is None else (g[k], vg[k] + ve)
    m_t = mu + c * (prior(P, SUBJ[p["subject_id"]])[0] - P["m"]); v_t = tau2 + vmu
    lp = -0.5 * (TH - m_t) ** 2 / v_t
    for it, l in own:
        dm, dv = dstat(it); q = joint.expected_sigmoid(TH + dm, np.full(len(TH), dv))
        lp = lp + (np.log(q) if l else np.log1p(-q))
    w = np.exp(lp - lp.max()); w /= w.sum()
    st = np.array([dstat(i) for i in p["eval_items"]])
    uniq, inv = np.unique(st, axis=0, return_inverse=True)
    Q = joint.expected_sigmoid(TH[:, None] + uniq[None, :, 0], np.broadcast_to(uniq[None, :, 1], (len(TH), len(uniq))))
    return (w @ Q)[inv.ravel()]

def hybrid(P, p, own, others, c):
    if CFG.get("rho"):
        return hybrid_cat(P, p, own, others, c)
    subj_ids = sorted({o[0] for o in others})
    sidx = {x: j for j, x in enumerate(subj_ids)}
    item_ids = sorted({it for _, it, _ in others})
    iidx = {x: j for j, x in enumerate(item_ids)}
    a = np.array([c * (prior(P, SUBJ[x])[0] - P["m"]) for x in subj_ids])
    tau2 = c ** 2 * P["sd"] ** 2 * P["r"] * CFG.get("tau_scale", 1.0)
    th, vth, d, vd, mu, vmu = joint.fit(np.array([sidx[x] for x, _, _ in others]), np.array([iidx[i] for _, i, _ in others]),
                                        np.array([l for _, _, l in others], float), a, c * P["m"], c ** 2 * P["s2mu"], tau2, P["sd2"], len(item_ids))
    sd2 = P["sd2"]
    def dstat(i):
        j = iidx.get(i)
        return (0.0, sd2) if j is None else (d[j], vd[j])
    m_t = mu + c * (prior(P, SUBJ[p["subject_id"]])[0] - P["m"])
    v_t = tau2 + vmu
    if CFG.get("eb"):
        u = th - mu - a
        nj = len(u); k0 = CFG.get("eb_k0", 5.0)
        t_emp = float(np.mean(u ** 2 + vth))
        v_t = (k0 * tau2 + nj * t_emp) / (k0 + nj) + vmu
    lp = -0.5 * (TH - m_t) ** 2 / v_t
    for it, l in own:
        dm, dv = dstat(it)
        q = joint.expected_sigmoid(TH + dm, np.full(len(TH), dv))
        lp = lp + (np.log(q) if l else np.log1p(-q))
    w = np.exp(lp - lp.max()); w /= w.sum()
    st = np.array([dstat(i) for i in p["eval_items"]])
    uniq, inv = np.unique(st, axis=0, return_inverse=True)
    Q = joint.expected_sigmoid(TH[:, None] + uniq[None, :, 0], np.broadcast_to(uniq[None, :, 1], (len(TH), len(uniq))))
    return (w @ Q)[inv.ravel()]

def prior(P, s):
    x = M2.features(s, P["ns"], P["yc"])
    return P["m"] + P["sd"] * float(x @ np.asarray(P["beta"])), P["s2mu"] + P["sd"] ** 2 * P["r"]

configs = {k: v for k, v in {"v2": dict(v2only=True), "v3": {}, "v3_sd2x.7": dict(sd2_scale=0.7), "v3_c.8": dict(cfac=0.8), "v3h": dict(hybrid=True), "v3h_sd2x.7": dict(hybrid=True, sd2_scale=0.7), "v3h_t1.5": dict(hybrid=True, tau_scale=1.5), "v3h_t.7": dict(hybrid=True, tau_scale=0.7), "v3h_sd2x1.3": dict(hybrid=True, sd2_scale=1.3), "v3h_eb": dict(hybrid=True, eb=True), "cat.1": dict(hybrid=True, rho=0.1), "cat.25": dict(hybrid=True, rho=0.25), "cat.4": dict(hybrid=True, rho=0.4)}.items() if k in sys.argv[2].split(",")}
n_other = None if sys.argv[1] == "all" else int(sys.argv[1])
res = {}
for seed in (0, 1, 2):
    T = sim.trajectories(seed); EXT, EXS = extdata.load_ext(0); SUBJ = T["subjects"]; ITEMS = T["items"]
    for k, cfg in configs.items():
        CFG.clear(); CFG.update(cfg)
        r, _ = sim2.score(pred, T, lambda *a: fit(*a, EXT=EXT, EXS=EXS), n_other=n_other, seed=seed); res.setdefault(k, []).append(r)
        print(seed, k, round(r["brier_alc"], 4), flush=True)
for k, rs in res.items():
    a = [r["brier_alc"] for r in rs]
    print(f"n_other={n_other} {k:10s} {np.mean(a):.4f} +- {np.std(a):.4f}", " ".join(f"B{b}={np.mean([r[f'B{b}'] for r in rs]):.4f}" for b in (0, 1, 3, 7, 15, 31)), {kk[:6]: round(np.mean([r['per_bench'][kk] for r in rs]), 4) for kk in rs[0]['per_bench']})
