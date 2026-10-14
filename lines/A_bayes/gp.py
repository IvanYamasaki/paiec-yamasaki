"""Within-pair GP classification (Laplace) for line A.

latent f_i = theta + u_cat(i) + t(x_i) + e_i; cov C = v0 + sc*[same cat] + st*S_ij + se*[i=j].
Labeled items give the posterior; prediction p* = E[sigmoid(f*)] (probit approx)."""
import numpy as np

def laplace_predict(m0, CLL, cLs, css, y, iters=30):
    """CLL: (n,n) prior cov among labeled; cLs: (n, q) cross cov; css: (q,) prior var of targets.
    Returns predictive probabilities for q targets."""
    n = len(y)
    if n == 0:
        return probit_mean(np.full(len(css), m0), css)
    f = np.full(n, m0, float)
    a = np.zeros(n)
    I = np.eye(n)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-f)); W = p * (1 - p); sW = np.sqrt(W)
        B = I + sW[:, None] * CLL * sW[None, :]
        L = np.linalg.cholesky(B)
        b = W * (f - m0) + (y - p)
        z = np.linalg.solve(L.T, np.linalg.solve(L, sW * (CLL @ b)))
        a_new = b - sW * z
        f_new = m0 + CLL @ a_new
        if np.max(np.abs(f_new - f)) < 1e-6:
            f, a = f_new, a_new; break
        f, a = f_new, a_new
    p = 1 / (1 + np.exp(-f)); W = p * (1 - p); sW = np.sqrt(W)
    B = I + sW[:, None] * CLL * sW[None, :]; L = np.linalg.cholesky(B)
    mu = m0 + cLs.T @ (y - p)
    v = np.linalg.solve(L, sW[:, None] * cLs)
    var = css - (v * v).sum(0)
    return probit_mean(mu, np.maximum(var, 1e-9))

def probit_mean(mu, var):
    return 1 / (1 + np.exp(-mu / np.sqrt(1 + np.pi * var / 8)))
