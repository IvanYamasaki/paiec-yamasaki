"""Subject features for the population prior (only the visible subject dict)."""
import numpy as np
REAS = {"minimal": -1.5, "low": -1.0, "medium": 0.0, "high": 1.0, "xhigh": 2.0, "max": 2.0}
AGENTIC = ("agent scaffold", "community")

def year(s):
    d = s.get("release_date", "")
    try:
        return int(d[:4]) + (int(d[5:7]) - 1) / 12
    except Exception:
        return np.nan

def subj_feats(s, ycenter):
    y = year(s)
    reas = REAS.get(s.get("reasoning_effort", "").strip().lower(), 0.0)
    ag = float(s.get("normalized_name", "").startswith("Agent:") or s.get("provider", "").strip().lower() in AGENTIC)
    return np.array([0.0 if np.isnan(y) else y - ycenter, float(np.isnan(y)), reas, ag])
