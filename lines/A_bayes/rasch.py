import numpy as np, pandas as pd
from scipy.special import expit

def fit_rasch(df, iters=200, l2_item=0.1, l2_subj=0.01):
    """df: subject_id, item_id, label. Returns (theta: Series, d: Series) with logit p = theta_s - d_i... we use +d (easiness)."""
    s_codes, s_idx = np.unique(df.subject_id.to_numpy(), return_inverse=True)
    i_codes, i_idx = np.unique(df.item_id.to_numpy(), return_inverse=True)
    y = df.label.to_numpy(float)
    th = np.zeros(len(s_codes)); d = np.zeros(len(i_codes))
    for _ in range(iters):
        p = expit(th[s_idx] + d[i_idx]); g = y - p; h = p * (1 - p)
        th += (np.bincount(s_idx, g, len(th)) - l2_subj * th) / (np.bincount(s_idx, h, len(th)) + l2_subj)
        p = expit(th[s_idx] + d[i_idx]); g = y - p; h = p * (1 - p)
        d += (np.bincount(i_idx, g, len(d)) - l2_item * d) / (np.bincount(i_idx, h, len(d)) + l2_item)
        m = d.mean(); d -= m; th += m
    return pd.Series(th, s_codes), pd.Series(d, i_codes)
