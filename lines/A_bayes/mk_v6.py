"""Builds v6/model.py from v5/model.py (LLM note as item effect)."""
s = open("v5/model.py", encoding="utf-8").read()


def rep(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new)


rep('''"""Line A v5 (unified base model). Changes over v4:''', '''"""Line A v6 (unified base model) = v5 + item effect from an LLM success note.

Item easiness prior: d_i ~ N(g_cat(i) + BETA * (x_i - xc), SDR2), x_i = logit(p_i/100)
of the LLM note (line B module), xc = benchmark centre: mean x of the items visible
in labeled for this benchmark, shrunk toward the training mean with N0 pseudo-items
(N0 = 0: purely relative note; nothing visible -> no note effect). Items without a
note: d_i ~ N(g_cat, SDR2 + BETA^2 VX). BETA, SDR2, VX are fitted on the training
benchmarks (Rasch d regressed on the within-benchmark centred note). With the note
on, the pair-scope path is the Rasch grid too (item priors differ per item).

v5 (unified base model). Changes over v4:''')
rep("import json, os, re\n", "import csv, hashlib, json, os, re\n")
rep("S2MU_MIN = 0.3 ", "S2MU_MIN = 1.0 ")
rep('''_MEMO, _MEMO_MAX = {}, 64''', '''_MEMO, _MEMO_MAX = {}, 64
N0 = 1.0            # pseudo-items pulling the note centre toward the training mean
USE_NOTE = True
NOTE_FILES = [os.path.join(_HERE, "notes.csv"),
              os.path.join(_HERE, "..", "..", "B_conteudo", "module", "scores_sonnet.csv")]''')
rep('P = {"rho": 0.2,', 'P = {"nbeta": 0.0, "sdr2": 4.0, "vx": 1.0, "xbar": 0.0, "rho": 0.2,')
rep('''if os.path.exists(os.path.join(_HERE, "params.json")):
    P = json.load(open(os.path.join(_HERE, "params.json")))''', '''if os.path.exists(os.path.join(_HERE, "params.json")):
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
    return NOTES.get(hashlib.sha1((item.get("item_content", "") or "").encode("utf-8")).hexdigest())''')

a0 = s.index("def _joint_fit(")
a1 = s.index("def _memo_core(")
new_core = '''def _joint_fit(s_idx, i_idx, y, a, m, s2mu, tau2, n_items, coi, n_cats, rho, dm0, dv0, iters=80):
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


'''
s = s[:a0] + new_core + s[a1:]

rep('''def item_var(obs, items=None, min_subjects=5):
    """Mean within-benchmark Rasch item variance, and mean share explained by item_features."""
    v, r2 = [], []''', '''def item_var(obs, items=None, min_subjects=5):
    """Mean within-benchmark Rasch item variance, mean share explained by item_features,
    and the note regression: slope of d on the within-benchmark centred note."""
    v, r2, betas, res, vxs, xm = [], [], [], [], [], []''')
rep('''        var, d, ids = _rasch_item_var(g, return_d=True)
        v.append(var)''', '''        var, d, ids = _rasch_item_var(g, return_d=True)
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
                xm.append(float(xx[ok].mean()))''')
rep('''    sd2 = float(np.mean(v)) if v else 4.0
    rho = float(np.clip(np.mean(r2), 0.05, 0.5)) if r2 else 0.2
    return sd2, rho''', '''    sd2 = float(np.mean(v)) if v else 4.0
    rho = float(np.clip(np.mean(r2), 0.05, 0.5)) if r2 else 0.2
    if betas:
        note_p = {"nbeta": float(np.mean(betas)), "sdr2": float(np.mean(res)),
                  "vx": float(np.mean(vxs)), "xbar": float(np.mean(xm))}
    else:
        note_p = {"nbeta": 0.0, "sdr2": sd2, "vx": 1.0, "xbar": 0.0}
    return sd2, rho, note_p''')
rep('''    P["sd2"], P["rho"] = item_var(train_obs[["subject_id", "item_id", "bench", "label"]], items)''',
    '''    P["sd2"], P["rho"], note_p = item_var(train_obs[["subject_id", "item_id", "bench", "label"]], items)
    P.update(note_p)''')
open("v6/model.py", "w", encoding="utf-8").write(s)
print("ok")

# stress-fold guard: when every public binary benchmark is in train_obs, the test
# benchmark is one of the non-binary ones -> never read them from disk.
s = open("v6/model.py", encoding="utf-8").read()
old = '''    if not set(binary) & set(EXT_BENCHES):  # normal LOBO: add the non-binary public benchmarks'''
new = '''    held_binary = set(PUBLIC_BINARY) - set(binary)
    if held_binary and not set(binary) & set(EXT_BENCHES):  # normal LOBO: add the non-binary public benchmarks'''
assert old in s
s = s.replace(old, new)
s = s.replace('''EXT_BENCHES = ("matharena", "mmdocrag")''', '''EXT_BENCHES = ("matharena", "mmdocrag")
PUBLIC_BINARY = ("multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")''')
open("v6/model.py", "w", encoding="utf-8").write(s)
print("guard ok")
