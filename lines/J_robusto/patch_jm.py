"""Turn the copied A v7 model into the J model: CFG knobs for level robustness."""
s = open("jm/model.py", encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:90]
    s = s.replace(old, new)

rep('"""Line A v7 =', '''"""Line J (robust to the hidden-test level) = line A v7 with knobs in CFG (params.json "cfg"):
  lam, lev_lo, lev_hi : mixture weight and range of the spread-levels component;
  m_shrink           : pull the training level mean toward logit 0 (p = 0.5);
  varb_mult          : multiply the benchmark-level variance s2mu;
  varp_mult          : multiply the subject-deviation variance sd^2 r (faster per-pair updates);
  temp_a, temp_k0, temp_nb : logit temperature t = 1 - temp_a / (1 + (k + n_oth/temp_nb)/temp_k0)
                       applied around 0.5 (k = own labels, n_oth = other subjects' labels);
  use_others         : 0 ignores other subjects' labels (pair path only).
Line A v7 =''')
rep('''if os.path.exists(os.path.join(_HERE, "params.json")):
    P = json.load(open(os.path.join(_HERE, "params.json")))''', '''CFG = {"lam": 0.5, "lev_lo": 0.10, "lev_hi": 0.95, "m_shrink": 0.0, "varb_mult": 1.0, "varp_mult": 1.0,
       "temp_a": 0.0, "temp_k0": 3.0, "temp_nb": 30.0, "use_others": 1, "s2mu_min": 1.0}
if os.path.exists(os.path.join(_HERE, "params.json")):
    P = json.load(open(os.path.join(_HERE, "params.json")))
    CFG.update(P.get("cfg", {}))
P_RAW = dict(P)


def apply_cfg():
    """Effective params from the raw fit and CFG (call after changing CFG or refitting)."""
    global P, LAM, LEVELS
    P = dict(P_RAW)
    P["m"] = (1 - CFG["m_shrink"]) * P_RAW["m"]
    P["s2mu"] = max(P_RAW["s2mu"], CFG["s2mu_min"]) * CFG["varb_mult"]
    P["r"] = P_RAW["r"] * CFG["varp_mult"]
    LAM = CFG["lam"]
    q = np.linspace(CFG["lev_lo"], CFG["lev_hi"], 18)
    LEVELS = np.log(q / (1 - q))
    _MEMO.clear()


def _temper(p, k, n_oth):
    if CFG["temp_a"] <= 0:
        return p
    t = 1 - CFG["temp_a"] / (1 + (k + n_oth / CFG["temp_nb"]) / CFG["temp_k0"])
    p = min(max(p, 1e-9), 1 - 1e-9)
    return float(1 / (1 + np.exp(-t * np.log(p / (1 - p)))))''')
# apply after module constants are defined: append at the NOTES load line
rep('''NOTES = _load_notes()  # sha1(item_content) -> x = logit(p/100)''', '''NOTES = _load_notes()  # sha1(item_content) -> x = logit(p/100)
apply_cfg()''')
rep('''        if _skey(s) == sk:
            own.append((ik, int(y)))
        else:
            others.append((s, ik, int(y)))''', '''        if _skey(s) == sk:
            own.append((ik, int(y)))
        elif CFG["use_others"]:
            others.append((s, ik, int(y)))''')
rep('''        p = _post_mean(mean, var, sum(y for _, y in own), len(own))
        return lambda item: p''', '''        p = _temper(_post_mean(mean, var, sum(y for _, y in own), len(own)), len(own), 0)
        return lambda item: p''')
rep('''        return float(w @ expected_sigmoid(TH3 + dm, np.full(len(TH3), dv)))
    return finish''', '''        return _temper(float(w @ expected_sigmoid(TH3 + dm, np.full(len(TH3), dv))), len(own), len(others))
    return finish''')
rep('''    P = fit_params(obs, subs, binary)
    P["sd2"], P["rho"], note_p = item_var(train_obs[["subject_id", "item_id", "bench", "label"]], items)
    P.update(note_p)''', '''    global P_RAW
    P = fit_params(obs, subs, binary)
    P["sd2"], P["rho"], note_p = item_var(train_obs[["subject_id", "item_id", "bench", "label"]], items)
    P.update(note_p)
    P_RAW = dict(P)
    apply_cfg()''')
rep('''    return {"m": m, "s2mu": max(s2mu, S2MU_MIN),''', '''    return {"m": m, "s2mu": s2mu,''')
open("jm/model.py", "w", encoding="utf-8").write(s)
print("ok")
