"""Summarize results/J_sim_*.jsonl: per config, real LOBO (pair-weighted over 4 folds),
worst real stress fold, and level-shifted (mean over shifts of the pair-weighted mean over 6 folds)."""
import glob, json, os, sys
import pandas as pd
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
rows = [json.loads(l) for f in glob.glob(os.path.join(ROOT, "results", f"J_sim_{pat}.jsonl")) for l in open(f)]
df = pd.DataFrame(rows); df["shift"] = df["shift"].fillna(-1)
ELIG = ["multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
def wmean(g, c): return (g[c] * g.n).sum() / g.n.sum()
out = []
for (conf, scope), g in df.groupby(["conf", "scope"]):
    r = {"conf": conf, "scope": scope, "seeds": g.seed.nunique()}
    real = g[g["shift"] == -1]
    lobo = real[real.fold.isin(ELIG)]
    per_seed = lobo.groupby("seed").apply(lambda x: wmean(x, "alc"), include_groups=False)
    r["lobo"] = per_seed.mean(); r["lobo_B0"] = lobo.groupby("seed").apply(lambda x: wmean(x, "B0"), include_groups=False).mean()
    st = real[~real.fold.isin(ELIG)].groupby("fold").alc.mean()
    r["math"], r["mmdoc"] = st.get("matharena"), st.get("mmdocrag")
    sh = g[g["shift"] >= 0]
    if len(sh):
        by = sh.groupby(["shift", "seed"]).apply(lambda x: pd.Series({c: wmean(x, c) for c in ("alc", "B0", "B1", "B3")}), include_groups=False).groupby("shift").mean()
        r["shift_mean"] = by.alc.mean(); r["shift_worst"] = by.alc.max(); r["sh_B0"] = by.B0.mean(); r["sh_B0_worst"] = by.B0.max(); r["sh_B1"] = by.B1.mean()
        for q in (0.05, 0.5, 0.95):
            if q in by.index: r[f"q{int(q*100)}"] = by.alc[q]
    out.append(r)
t = pd.DataFrame(out).sort_values("shift_mean" if "shift_mean" in out[0] else "lobo")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(t.round(4).to_string(index=False))
