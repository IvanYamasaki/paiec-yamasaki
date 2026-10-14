"""Line A v6 (unified base model) = v5 + item effect from an LLM success note.

Item easiness prior: d_i ~ N(g_cat(i) + BETA * (x_i - xc), SDR2), x_i = logit(p_i/100)
of the LLM note (line B module), xc = benchmark centre: mean x of the items visible
in labeled for this benchmark, shrunk toward the training mean with N0 pseudo-items
(N0 = 0: purely relative note; nothing visible -> no note effect). Items without a
note: d_i ~ N(g_cat, SDR2 + BETA^2 VX). BETA, SDR2, VX are fitted on the training
benchmarks (Rasch d regressed on the within-benchmark centred note). With the note
on, the pair-scope path is the Rasch grid too (item priors differ per item).

v5 (unified base model). Changes over v4:
  * benchmark-level prior: mean M0 and variance >= S2MU_MIN (wider, chosen by the
    worst stress fold, high-rate benchmarks matharena/mmdocrag binarized);
  * item category = item_features without id-like tokens (as in line D v8);
  * per-process memo of the item-independent part, keyed by (subject,
    benchmark_id, len(labeled), label sum, last entry) -> same output, faster;
  * local_fit never reads a benchmark that is not in train_obs when train_obs
    already holds a non-binary benchmark (stress folds).

Line A v4: v3 plus item category effects (item_features) learned on the new
benchmark from the other subjects' labels.

Line A v3: v2 plus a joint Rasch fit on the other subjects' labels of the
same benchmark found in `labeled` (the official site says labeled carries
entries across sampled subject-benchmark pairs).

Without other subjects of the target benchmark in labeled, v3 == v2:

Prior: theta_pair ~ N(m + sd * x'beta, s2mu + sd^2 * r), where x are features of
the visible subject dict (release year, missing year, reasoning effort, agentic
scaffold, cross-benchmark name score) and beta is a within-benchmark ridge
regression calibrated by leave-one-benchmark-out inside the training data.
Posterior on a grid; prediction = posterior mean of sigmoid(theta) (not the
sigmoid of the mean). No item effects. predict is a pure function of
(input, labeled, fitted params).

local_fit (local evaluator only) refits params on the fold's training rows plus
the non-binary public benchmarks (matharena, mmdocrag), which are never a test
benchmark. For submission, params.json holds the fit on all public data.

With other subjects' labels on the target benchmark (rate logit scale -> Rasch
scale by c = sqrt(1 + pi*sd2/8)):
  1. joint MAP of mu (benchmark level), theta_j (other subjects) and d_i (item
     easiness) from the OTHER subjects' labels only;
  2. target theta on an exact grid with prior N(mu + c*offset, tau2 + var(mu))
     and the target's own labels, each with its item's d_i posterior;
  3. p = E[sigmoid(theta + d_item)], d_item ~ N(d_i, v_i) or N(0, sd2) if unseen.
v4: d_i ~ N(g_cat(i), (1-rho) sd2), g_c ~ N(0, rho sd2); category = the
item_features string (or its first key=value token when full strings are
mostly unique). rho = share of item-difficulty variance explained by
categories in the training benchmarks.
No global cache: predict_batch only shares the step-1/2 fit inside one call,
through the same code path as predict."""
import csv, hashlib, json, os, re
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
TH = np.linspace(-8, 8, 641)
GR = 1 / (1 + np.exp(-TH))
LG, L1G = np.log(GR), np.log1p(-GR)
REAS = {"minimal": -1.5, "low": -1.0, "medium": 0.0, "high": 1.0, "xhigh": 2.0, "max": 2.0}
AGENTIC = ("agent scaffold", "community")
EXT_BENCHES = ("matharena", "mmdocrag")
PUBLIC_BINARY = ("multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")
RIDGE, MIN_PAIR = 5.0, 20
S2MU_MIN = 1.0      # floor of the benchmark-level variance (rate-logit scale)
M0 = None           # None: pair-weighted mean of training levels; float: fixed prior mean
M0_W = 0.0          # shrink the data mean toward M0 (0 = data only)
ID_LIKE = re.compile(r"(idx|_id|^id)=|=\d+$")
_MEMO, _MEMO_MAX = {}, 64
N0 = 1.0            # pseudo-items pulling the note centre toward the training mean
USE_NOTE = True
NOTE_FILES = [os.path.join(_HERE, "notes.csv"),
              os.path.join(_HERE, "..", "..", "B_conteudo", "module", "scores_sonnet.csv")]

P = {"nbeta": 0.0, "sdr2": 4.0, "vx": 1.0, "xbar": 0.0, "rho": 0.2, "sd2": 4.0, "m": -1.0, "s2mu": 0.5, "sd": 1.2, "beta": [0, 0, 0, 0, 0], "r": 1.0, "ns": {}, "yc": 2025.0}
if os.path.exists(os.path.join(_HERE, "params.json")):
    P = json.load(open(os.path.join(_HERE, "params.json")))


def _p_to_x(p):
    q = min(max(float(p) / 100.0, 0.03), 0.97)
    return float(np.log(q / (1 - q)))


def _load_notes():
    for f in NOTE_FILES:
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                return {r["key"]: _p_to_x(r["p"]) for r in csv.DictReader(fh)}
    return {}


NOTES = _load_notes()  # sha1(item_content) -> x = logit(p/100)


def note(item):
    """x = logit of the LLM success note for this item, or None."""
    if not USE_NOTE:
        return None
    return NOTES.get(hashlib.sha1((item.get("item_content", "") or "").encode("utf-8")).hexdigest())


# ---------- subject features ----------
def sname(s):
    n = s.get("normalized_name", "")
    if not n:
        for part in s.get("subject_features_extra", "").split(";"):
            if part.startswith("source_model_name="):
                n = "src:" + part.split("=", 1)[1].split(" (")[0].strip().lower()
    return n


def year(s):
    d = s.get("release_date", "")
    try:
        return int(d[:4]) + (int(d[5:7]) - 1) / 12
    except Exception:
        return np.nan


def features(s, ns, yc):
    y = year(s)
    sc = ns.get(sname(s), np.nan)
    reas = REAS.get(s.get("reasoning_effort", "").strip().lower(), 0.0)
    ag = float(s.get("normalized_name", "").startswith("Agent:")
               or s.get("provider", "").strip().lower() in AGENTIC)
    return np.array([0.0 if np.isnan(sc) else sc,
                     0.0 if np.isnan(y) else y - yc, float(np.isnan(y)), reas, ag])


# ---------- prediction ----------
GH_X, GH_W = np.polynomial.hermite.hermgauss(24)
TH3 = np.linspace(-14, 14, 561)


def expected_sigmoid(mean, var):
    mean = np.asarray(mean, float)[..., None]
    sd = np.sqrt(2 * np.asarray(var, float))[..., None]
    return (GH_W * (1 / (1 + np.exp(-(mean + sd * GH_X))))).sum(-1) / np.sqrt(np.pi)


def _skey(s):
    return tuple(sorted(s.items()))


def _ikey(it):
    return (it.get("benchmark_id", ""), it.get("item_content", ""), it.get("item_features", ""))


def _prior(subject):
    x = features(subject, P["ns"], P["yc"])
    mean = P["m"] + P["sd"] * float(x @ np.asarray(P["beta"]))
    var = P["s2mu"] + P["sd"] ** 2 * P["r"]
    return mean, var


def _post_mean(mean, var, k, n):
    lp = -0.5 * (TH - mean) ** 2 / var + k * LG + (n - k) * L1G
    w = np.exp(lp - lp.max())
    return float((w * GR).sum() / w.sum())


def _joint_fit(s_idx, i_idx, y, a, m, s2mu, tau2, n_items, coi, n_cats, rho, dm0, dv0, iters=80):
    """MAP by alternating Newton steps. Item prior: d_i ~ N(g_cat + dm0_i, dv0_i),
    g_c ~ N(0, rho * sd2). coi: category index per item (-1 = none)."""
    ns = len(a)
    mu, th, d, g = m, m + a.copy(), dm0.copy(), np.zeros(max(n_cats, 1))
    has = coi >= 0
    cidx = np.where(has, coi, 0)
    vg0 = rho * P["sd2"]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(th[s_idx] + d[i_idx])))
        gr, h = y - p, p * (1 - p)
        th_old = th
        th = th + (np.bincount(s_idx, gr, ns) - (th - mu - a) / tau2) / (np.bincount(s_idx, h, ns) + 1 / tau2)
        p = 1 / (1 + np.exp(-(th[s_idx] + d[i_idx])))
        gr, h = y - p, p * (1 - p)
        pm = dm0 + np.where(has, g[cidx], 0.0)
        d = d + (np.bincount(i_idx, gr, n_items) - (d - pm) / dv0) / (np.bincount(i_idx, h, n_items) + 1 / dv0)
        if n_cats:
            r = (d - dm0)[has] / dv0[has]
            g = np.bincount(cidx[has], r, n_cats) / (np.bincount(cidx[has], 1 / dv0[has], n_cats) + 1 / vg0)
        mu = (m / s2mu + (th - a).sum() / tau2) / (1 / s2mu + ns / tau2)
        if np.max(np.abs(th - th_old)) < 1e-6:
            break
    p = 1 / (1 + np.exp(-(th[s_idx] + d[i_idx])))
    h = p * (1 - p)
    v_mu = 1 / (1 / s2mu + ns / tau2)
    Hi = np.bincount(i_idx, h, n_items)
    v_d = 1 / (Hi + 1 / dv0)
    v_g = None
    if n_cats:
        w = np.where(Hi > 0, 1 / (dv0 + 1 / np.maximum(Hi, 1e-12)), 0.0)
        v_g = 1 / (np.bincount(cidx[has], w[has], n_cats) + 1 / vg0)
    return d, v_d, g, v_g, mu, v_mu


def _cat(feat):
    """item_features stratum: drop id-like key=value tokens."""
    if not feat:
        return ""
    toks = [t.strip() for t in feat.split(";") if t.strip() and not ID_LIKE.search(t.strip())]
    return ";".join(toks)


def _core(subject, bench, labeled):
    """Everything that does not depend on the target item. Returns a function item -> p."""
    sk = _skey(subject)
    own, others, xs = [], [], {}
    for (s, it), y in labeled or []:
        if it.get("benchmark_id") != bench:
            continue
        ik = _ikey(it)
        if ik not in xs:
            xs[ik] = note(it)
        if _skey(s) == sk:
            own.append((ik, int(y)))
        else:
            others.append((s, ik, int(y)))
    mean, var = _prior(subject)
    vis = [x for x in xs.values() if x is not None]
    use_x = USE_NOTE and P["nbeta"] != 0 and (len(vis) + N0) > 0
    if not others and not use_x:
        p = _post_mean(mean, var, sum(y for _, y in own), len(own))
        return lambda item: p
    xc = (N0 * P["xbar"] + sum(vis)) / (N0 + len(vis)) if use_x else 0.0
    sd2, sdr2, beta = P["sd2"], P["sdr2"], P["nbeta"]
    c = np.sqrt(1 + np.pi * sd2 / 8)

    def prior_d(x):
        if not use_x:
            return 0.0, sd2
        if x is None:
            return 0.0, sdr2 + beta ** 2 * P["vx"]
        return beta * (x - xc), sdr2

    rho = P["rho"]
    tau2 = c ** 2 * P["sd"] ** 2 * P["r"]
    cats, iidx = {}, {}
    if others:
        subj, sidx = [], {}
        for s, _, _ in others:
            k = _skey(s)
            if k not in sidx:
                sidx[k] = len(subj)
                subj.append(s)
        ifeat = []
        for _, ik, _ in others:
            if ik not in iidx:
                iidx[ik] = len(iidx)
                ifeat.append(ik[2])
        for f in ifeat:
            cf = _cat(f)
            if cf:
                cats.setdefault(cf, len(cats))
        coi = np.array([cats.get(_cat(f), -1) for f in ifeat])
        pri = np.array([prior_d(xs[ik]) for ik in iidx])
        a = np.array([c * (_prior(s)[0] - P["m"]) for s in subj])
        d, v_d, g, v_g, mu, v_mu = _joint_fit(np.array([sidx[_skey(s)] for s, _, _ in others]),
                                              np.array([iidx[ik] for _, ik, _ in others]),
                                              np.array([y for _, _, y in others], float),
                                              a, c * P["m"], c ** 2 * P["s2mu"], tau2, len(iidx),
                                              coi, len(cats), rho, pri[:, 0].copy(), pri[:, 1].copy())
    else:
        mu, v_mu = c * P["m"], c ** 2 * P["s2mu"]

    def dstat(ik, x):
        j = iidx.get(ik)
        if j is not None:
            return (float(d[j]), float(v_d[j]))
        pm, pv = prior_d(x)
        k = cats.get(_cat(ik[2])) if cats else None
        if k is None:
            return (pm, pv)
        return (pm + float(g[k]), pv + float(v_g[k]))

    m_t, v_t = mu + c * (mean - P["m"]), tau2 + v_mu
    lp = -0.5 * (TH3 - m_t) ** 2 / v_t
    for ik, y in own:
        dm, dv = dstat(ik, xs[ik])
        q = expected_sigmoid(TH3 + dm, np.full(len(TH3), dv))
        lp = lp + (np.log(q) if y else np.log1p(-q))
    w = np.exp(lp - lp.max())
    w /= w.sum()

    def finish(item):
        ik = _ikey(item)
        dm, dv = dstat(ik, xs[ik] if ik in xs else note(item))
        return float(w @ expected_sigmoid(TH3 + dm, np.full(len(TH3), dv)))
    return finish


def _memo_core(subject, bench, labeled):
    lab = labeled or []
    last = lab[-1] if lab else None
    key = (_skey(subject), bench, len(lab), sum(int(y) for _, y in lab),
           None if last is None else (_skey(last[0][0]), _ikey(last[0][1]), int(last[1])))
    f = _MEMO.get(key)
    if f is None:
        if len(_MEMO) >= _MEMO_MAX:
            _MEMO.clear()
        f = _MEMO[key] = _core(subject, bench, labeled)
    return f


def predict(input, labeled=None):
    subject, item = input
    return _memo_core(subject, item.get("benchmark_id"), labeled)(item)


def predict_batch(inputs, labeled):
    cores, out = {}, []
    for subject, item in inputs:
        key = (_skey(subject), item.get("benchmark_id"))
        if key not in cores:
            cores[key] = _memo_core(subject, item.get("benchmark_id"), labeled)
        out.append(cores[key](item))
    return out


# ---------- fitting ----------
def _pair_table(obs, subjects):
    a = obs.groupby(["subject_id", "bench"])["label"].agg(["sum", "size"]).reset_index()
    a = a[a["size"] >= MIN_PAIR].copy()
    k, n = a["sum"].to_numpy(float), a["size"].to_numpy(float)
    p = (k + 0.5) / (n + 1)
    a["y"] = np.log(p / (1 - p))
    a["name"] = a["subject_id"].map(lambda s: sname(subjects[s]))
    a = a[a.groupby("bench")["y"].transform("size") >= 5].copy()
    g = a.groupby("bench")["y"]
    a["z"] = (a["y"] - g.transform("mean")) / g.transform("std")
    return a


def _name_scores(pt, exclude=None):
    q = pt[pt["bench"] != exclude]
    return q.groupby(["name", "bench"])["z"].mean().groupby("name").mean().to_dict()


def fit_params(obs, subjects, binary_benches):
    """obs: subject_id, item_id, bench, label (all training benchmarks).
    binary_benches: benchmarks whose rate level is used for the prior mean."""
    pt = _pair_table(obs, subjects)
    yc = float(np.nanmean([year(subjects[s]) for s in pt["subject_id"]]))
    Xs, Ys = [], []
    for b in pt["bench"].unique():
        tg = pt[pt["bench"] == b]
        ns = _name_scores(pt, b)
        X = np.array([features(subjects[s], ns, yc) for s in tg["subject_id"]])
        z = tg["z"].to_numpy()
        Xs.append(X - X.mean(0))
        Ys.append(z - z.mean())
    X, Y = np.vstack(Xs), np.concatenate(Ys)
    beta = np.linalg.solve(X.T @ X + RIDGE * np.eye(X.shape[1]), X.T @ Y)
    r = float(np.mean((Y - X @ beta) ** 2))
    bs = pt[pt["bench"].isin(binary_benches)].groupby("bench")["y"].agg(["mean", "std", "size"])
    m = float(np.average(bs["mean"], weights=bs["size"]))
    s2mu = float(np.average((bs["mean"] - m) ** 2, weights=bs["size"]))
    if M0 is not None:
        s2mu += M0_W * (m - M0) ** 2
        m = (1 - M0_W) * m + M0_W * M0
    sd = float(np.sqrt(np.average(bs["std"] ** 2, weights=bs["size"])))
    return {"m": m, "s2mu": max(s2mu, S2MU_MIN), "sd": sd, "beta": beta.tolist(), "r": r,
            "ns": _name_scores(pt), "yc": yc}


def load_ext(root, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed + 1000)
    parts, subjects = [], {}
    fields = ("normalized_name", "provider", "release_date", "access_date", "harness",
              "harness_version", "reasoning_effort", "subject_features_extra")
    for d in EXT_BENCHES:
        p = os.path.join(root, d)
        if not os.path.exists(os.path.join(p, "response.parquet")):
            continue
        for row in pd.read_parquet(os.path.join(p, "subjects.parquet")).to_dict("records"):
            subjects[row["subject_id"]] = {k: "" if row.get(k) is None or (isinstance(row.get(k), float) and np.isnan(row.get(k))) else str(row.get(k)) for k in fields}
        r = pd.read_parquet(os.path.join(p, "response.parquet"), columns=["subject_id", "item_id", "response"])
        r = r[r["response"].notna()].copy()
        r["label"] = (r["response"].astype(float) >= 0.5).astype(int)
        r["_u"] = rng.random(len(r))
        r = r.sort_values("_u").drop_duplicates(["subject_id", "item_id"])
        r["bench"] = d
        parts.append(r[["subject_id", "item_id", "bench", "label"]])
    return (pd.concat(parts, ignore_index=True) if parts else None), subjects


def local_fit(train_obs, subjects, items):
    import pandas as pd
    global P
    obs = train_obs[["subject_id", "item_id", "bench", "label"]]
    subs = dict(subjects)
    binary = sorted(obs["bench"].unique())
    held_binary = set(PUBLIC_BINARY) - set(binary)
    if held_binary and not set(binary) & set(EXT_BENCHES):  # normal LOBO: add the non-binary public benchmarks
        root = os.path.join(_HERE, "..", "..", "..", "data", "measurement-db")
        ext, ext_subj = load_ext(root)
        subs.update(ext_subj)
        if ext is not None:
            obs = pd.concat([obs, ext], ignore_index=True)
    P = fit_params(obs, subs, binary)
    P["sd2"], P["rho"], note_p = item_var(train_obs[["subject_id", "item_id", "bench", "label"]], items)
    P.update(note_p)


def _rasch_item_var(df, iters=200, l2_item=0.1, l2_subj=0.01, min_resp=5, return_d=False):
    s_codes, s_idx = np.unique(df["subject_id"].to_numpy(), return_inverse=True)
    i_codes, i_idx = np.unique(df["item_id"].to_numpy(), return_inverse=True)
    y = df["label"].to_numpy(float)
    th, d = np.zeros(len(s_codes)), np.zeros(len(i_codes))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(th[s_idx] + d[i_idx])))
        th += (np.bincount(s_idx, y - p, len(th)) - l2_subj * th) / (np.bincount(s_idx, p * (1 - p), len(th)) + l2_subj)
        p = 1 / (1 + np.exp(-(th[s_idx] + d[i_idx])))
        d += (np.bincount(i_idx, y - p, len(d)) - l2_item * d) / (np.bincount(i_idx, p * (1 - p), len(d)) + l2_item)
        mm = d.mean(); d -= mm; th += mm
    cnt = np.bincount(i_idx, minlength=len(d))
    keep = cnt >= min_resp
    if return_d:
        return float(np.var(d[keep], ddof=1)), d[keep], i_codes[keep]
    return float(np.var(d[keep], ddof=1))


def _icc(d, c):
    """One-way ANOVA intraclass correlation of d by category c (share of variance between)."""
    ks = sorted(set(c))
    n = np.array([np.sum(c == k) for k in ks], float)
    if len(ks) < 2 or len(d) - len(ks) < 2:
        return 0.0
    mk = np.array([d[c == k].mean() for k in ks])
    msb = float((n * (mk - d.mean()) ** 2).sum() / (len(ks) - 1))
    msw = float(sum(((d[c == k] - d[c == k].mean()) ** 2).sum() for k in ks) / (len(d) - len(ks)))
    n0 = (n.sum() - (n ** 2).sum() / n.sum()) / (len(ks) - 1)
    return max(0.0, (msb - msw) / (msb + (n0 - 1) * msw))


def item_var(obs, items=None, min_subjects=5):
    """Mean within-benchmark Rasch item variance, mean share explained by item_features,
    and the note regression: slope of d on the within-benchmark centred note."""
    v, r2, betas, res, vxs, xm = [], [], [], [], [], []
    for _, g in obs.groupby("bench"):
        if g["subject_id"].nunique() < min_subjects:
            continue
        var, d, ids = _rasch_item_var(g, return_d=True)
        v.append(var)
        if items is not None:
            xx = np.array([np.nan if note(items[i]) is None else note(items[i]) for i in ids])
            ok = ~np.isnan(xx)
            if ok.sum() >= 30:
                xcn = xx[ok] - xx[ok].mean()
                dc = d[ok] - d[ok].mean()
                b = float(xcn @ dc / (xcn @ xcn))
                betas.append(b)
                res.append(float(np.var(dc - b * xcn, ddof=2)))
                vxs.append(float(np.var(xcn)))
                xm.append(float(xx[ok].mean()))
        if items is not None:
            f = [items[i].get("item_features", "") for i in ids]
            if sum(1 for x in f if x) >= 0.5 * len(f) and len(set(f)) > 1:
                c = np.array([_cat(x) for x in f])
                r2.append(_icc(d, c))
    sd2 = float(np.mean(v)) if v else 4.0
    rho = float(np.clip(np.mean(r2), 0.05, 0.5)) if r2 else 0.2
    if betas:
        note_p = {"nbeta": float(np.mean(betas)), "sdr2": float(np.mean(res)),
                  "vx": float(np.mean(vxs)), "xbar": float(np.mean(xm))}
    else:
        note_p = {"nbeta": 0.0, "sdr2": sd2, "vx": 1.0, "xbar": 0.0}
    return sd2, rho, note_p
