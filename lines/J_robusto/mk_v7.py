"""Builds v7/model.py from v6/model.py: benchmark-level prior as a mixture.

Level prior of a pair on the rate-logit scale:
    (1 - LAM) * N(m + off, s2mu + sd^2 r)  +  LAM * mean_k N(L_k + off, sd^2 r)
with L_k = logit of 10%, 15%, ..., 95% (spread levels) and off = the subject
offset from metadata. LAM = 0 gives v6. In the joint (other subjects) path the
benchmark level mu gets the moment-matched Gaussian of the same mixture, so the
other subjects' labels move it quickly. USE_NOTE defaults to False (no API)."""
s = open("v6/model.py", encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    assert s.count(old) >= 1, old[:80]
    s = s.replace(old, new) if count is None else s.replace(old, new, count)


rep('''"""Line A v6 (unified base model)''', '''"""Line A v7 = v6 + mixture prior for the benchmark level (rates 10% to 95%), note off by default.

Level prior on the rate-logit scale: (1 - LAM) N(m + off, s2mu + sd^2 r) + LAM mean_k N(L_k + off, sd^2 r),
L_k = logit(0.10 ... 0.95). Joint path: mu ~ moment-matched Gaussian of that mixture.

Line A v6 (unified base model)''')
rep("USE_NOTE = True", "USE_NOTE = False")
rep('''_MEMO, _MEMO_MAX = {}, 64''', '''_MEMO, _MEMO_MAX = {}, 64
LAM = 0.5           # weight of the spread-levels component in the level prior
LEVELS = np.log(np.linspace(0.10, 0.95, 18) / (1 - np.linspace(0.10, 0.95, 18)))''')

# helpers for the mixture
rep('''def _post_mean(mean, var, k, n):
    lp = -0.5 * (TH - mean) ** 2 / var + k * LG + (n - k) * L1G''', '''def _level_logprior(grid, mean, var, scale=1.0):
    """log prior of the pair logit (rate scale times `scale`) under the level mixture."""
    wv = P["sd"] ** 2 * P["r"]
    a = np.log(max(1 - LAM, 1e-12)) - 0.5 * (grid - scale * mean) ** 2 / (scale ** 2 * var) - 0.5 * np.log(var)
    if LAM <= 0:
        return a
    off = mean - P["m"]
    L = scale * (LEVELS + off)
    comp = -0.5 * (grid[:, None] - L[None, :]) ** 2 / (scale ** 2 * wv) - 0.5 * np.log(wv)
    mx = comp.max(1, keepdims=True)
    b = np.log(LAM / len(LEVELS)) + mx[:, 0] + np.log(np.exp(comp - mx).sum(1))
    return np.logaddexp(a, b)


def _level_moments():
    """mean and variance of the benchmark level under the mixture (rate scale)."""
    m1 = (1 - LAM) * P["m"] + LAM * float(np.mean(LEVELS))
    m2 = (1 - LAM) * (P["s2mu"] + P["m"] ** 2) + LAM * float(np.mean(LEVELS ** 2))
    return m1, m2 - m1 ** 2


def _post_mean(mean, var, k, n):
    lp = _level_logprior(TH, mean, var) + k * LG + (n - k) * L1G''')

# joint path: mu prior from mixture moments; no-others note path: mixture on TH3
rep('''                                              a, c * P["m"], c ** 2 * P["s2mu"], tau2, len(iidx),''',
    '''                                              a, c * _level_moments()[0], c ** 2 * _level_moments()[1], tau2, len(iidx),''')
rep('''    else:
        mu, v_mu = c * P["m"], c ** 2 * P["s2mu"]''', '''    else:
        mu, v_mu = None, None''')
rep('''    m_t, v_t = mu + c * (mean - P["m"]), tau2 + v_mu
    lp = -0.5 * (TH3 - m_t) ** 2 / v_t''', '''    if mu is None:
        lp = _level_logprior(TH3, mean, var, scale=c)
    else:
        m_t, v_t = mu + c * (mean - P["m"]), tau2 + v_mu
        lp = -0.5 * (TH3 - m_t) ** 2 / v_t''')
# joint fit: a = offsets relative to m; mu prior mean shifted -> offsets stay relative to P["m"]
open("v7/model.py", "w", encoding="utf-8").write(s)
print("ok")
