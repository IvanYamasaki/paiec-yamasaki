"""Joint Rasch posterior on one benchmark from all labels in `labeled` (line A v3).

logit p(s, i) = theta_s + d_i;  theta_s = mu + a_s + u_s, u_s ~ N(0, tau2);
mu ~ N(m, s2mu); d_i ~ N(0, sd2). MAP by alternating 1-D Newton steps, with
diagonal Laplace variances. Prediction = E[sigmoid(theta_s + d_i)] by
Gauss-Hermite over the summed variance."""
import numpy as np

GH_X, GH_W = np.polynomial.hermite.hermgauss(24)


def expected_sigmoid(mean, var):
    mean = np.asarray(mean, float)[..., None]
    sd = np.sqrt(2 * np.asarray(var, float))[..., None]
    return (GH_W * (1 / (1 + np.exp(-(mean + sd * GH_X))))).sum(-1) / np.sqrt(np.pi)


def fit(s_idx, i_idx, y, a, m, s2mu, tau2, sd2, n_items, iters=60):
    """s_idx, i_idx: int arrays per label; a: prior offsets per subject (len n_subj).
    Returns theta mean/var per subject, d mean/var per item, mu mean/var."""
    ns = len(a)
    y = np.asarray(y, float)
    mu = m
    th = mu + a.copy()
    d = np.zeros(n_items)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(th[s_idx] + d[i_idx])))
        g, h = y - p, p * (1 - p)
        th_old = th
        gs = np.bincount(s_idx, g, ns) - (th - mu - a) / tau2
        hs = np.bincount(s_idx, h, ns) + 1 / tau2
        th = th + gs / hs
        p = 1 / (1 + np.exp(-(th[s_idx] + d[i_idx])))
        g, h = y - p, p * (1 - p)
        gi = np.bincount(i_idx, g, n_items) - d / sd2
        hi = np.bincount(i_idx, h, n_items) + 1 / sd2
        d = d + gi / hi
        prec = 1 / s2mu + ns / tau2
        mu = (m / s2mu + (th - a).sum() / tau2) / prec
        if np.max(np.abs(th - th_old)) < 1e-6:
            break
    p = 1 / (1 + np.exp(-(th[s_idx] + d[i_idx])))
    h = p * (1 - p)
    v_mu = 1 / (1 / s2mu + ns / tau2)
    v_th = 1 / (np.bincount(s_idx, h, ns) + 1 / (tau2 + v_mu))
    v_d = 1 / (np.bincount(i_idx, h, n_items) + 1 / sd2)
    return th, v_th, d, v_d, mu, v_mu


def fit_cat(s_idx, i_idx, y, a, m, s2mu, tau2, sd2, n_items, cat_of_item, n_cats, rho, iters=80):
    """As fit, with d_i ~ N(g_cat(i), (1-rho) sd2), g_c ~ N(0, rho sd2). cat_of_item: int array (-1 = none)."""
    ns = len(a)
    y = np.asarray(y, float)
    mu, th, d, g = m, m + a.copy(), np.zeros(n_items), np.zeros(max(n_cats, 1))
    ve = (1 - rho) * sd2 if n_cats else sd2
    has = cat_of_item >= 0
    cidx = np.where(has, cat_of_item, 0)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(th[s_idx] + d[i_idx])))
        gr, h = y - p, p * (1 - p)
        th_old = th
        th = th + (np.bincount(s_idx, gr, ns) - (th - mu - a) / tau2) / (np.bincount(s_idx, h, ns) + 1 / tau2)
        p = 1 / (1 + np.exp(-(th[s_idx] + d[i_idx])))
        gr, h = y - p, p * (1 - p)
        pm = np.where(has, g[cidx], 0.0)
        vi = np.where(has, ve, sd2)
        d = d + (np.bincount(i_idx, gr, n_items) - (d - pm) / vi) / (np.bincount(i_idx, h, n_items) + 1 / vi)
        if n_cats:
            num = np.bincount(cidx[has], d[has], n_cats) / ve
            den = np.bincount(cidx[has], minlength=n_cats) / ve + 1 / (rho * sd2)
            g = num / den
        mu = (m / s2mu + (th - a).sum() / tau2) / (1 / s2mu + ns / tau2)
        if np.max(np.abs(th - th_old)) < 1e-6:
            break
    p = 1 / (1 + np.exp(-(th[s_idx] + d[i_idx])))
    h = p * (1 - p)
    v_mu = 1 / (1 / s2mu + ns / tau2)
    v_th = 1 / (np.bincount(s_idx, h, ns) + 1 / (tau2 + v_mu))
    vi = np.where(has, ve, sd2)
    v_d = 1 / (np.bincount(i_idx, h, n_items) + 1 / vi)
    v_g = None
    if n_cats:
        # item-level info about g: each item contributes 1/(ve + v_lik) where v_lik = 1/H_i
        Hi = np.bincount(i_idx, h, n_items)
        w = np.where(Hi > 0, 1 / (ve + 1 / np.maximum(Hi, 1e-12)), 0.0)
        v_g = 1 / (np.bincount(cidx[has], w[has], n_cats) + 1 / (rho * sd2))
    return th, v_th, d, v_d, mu, v_mu, g, v_g, ve
