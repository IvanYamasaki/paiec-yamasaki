"""Quanto custa um prior de nivel estreito quando o benchmark de teste tem taxa alta?
Simula B_k (k=0,1,3,7,15,31) para pares reais de cada benchmark: posterior na grade do logit da taxa do par,
prior N(m, v); rotulos ~ Binomial(k, r). Brier esperado = (p_hat - r)^2 + r(1-r). 200 repeticoes por par."""
import os, numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "measurement-db")
rng = np.random.default_rng(0)
TH = np.linspace(-9, 9, 721); S = 1/(1+np.exp(-TH)); LS, LF = np.log(S), np.log1p(-S)
W = {0:.1,1:.2,3:.2,7:.2,15:.2,31:.1}
rates = {}
for d in sorted(os.listdir(ROOT)):
    f = os.path.join(ROOT, d, "response.parquet")
    if not os.path.exists(f): continue
    r = pd.read_parquet(f, columns=["subject_id","item_id","response"]); r = r[r.response.notna()]
    r["_u"] = rng.random(len(r)); r = r.sort_values("_u").drop_duplicates(["subject_id","item_id"])
    r["y"] = (r.response.astype(float)>=0.5).astype(int)
    g = r.groupby("subject_id").y.agg(["mean","size"]); rates[d] = g[g["size"]>=80]["mean"].to_numpy()
def post_mean(m, v, k, s):
    lp = -0.5*(TH-m)**2/v + s*LS + (k-s)*LF
    w = np.exp(lp-lp.max()); return float((w*S).sum()/w.sum())
def mix_post_mean(comps, k, s):  # mistura de normais (pesos iguais)
    num = den = 0.0
    for m, v in comps:
        lp = -0.5*(TH-m)**2/v + s*LS + (k-s)*LF - 0.5*np.log(v)
        w = np.exp(lp - 0); num += (w*S).sum(); den += w.sum()
    return float(num/den)
priors = {"estreito A (m=-1.2,v=1.0)": ("n", -1.2, 1.0), "medio (m=-0.6,v=2.0)": ("n", -0.6, 2.0),
          "largo (m=0,v=3.0)": ("n", 0.0, 3.0), "mistura 6 niveis (v=1.0)": ("mix", None, None)}
levels = [(-1.62,1.0),(-0.96,1.0),(-0.75,1.0),(-0.11,1.0),(0.65,1.0),(1.59,1.0)]
rows = []
for name,(kind,m,v) in priors.items():
    for d, rs in rates.items():
        res = {k:[] for k in W}
        for r in rs:
            for _ in range(60):
                for k in W:
                    s = rng.binomial(k, r) if k else 0
                    ph = post_mean(m, v, k, s) if kind=="n" else mix_post_mean(levels, k, s)
                    res[k].append((ph-r)**2 + r*(1-r))
        row = dict(prior=name, bench=d, **{f"B{k}": round(float(np.mean(res[k])),4) for k in W})
        row["ALC"] = round(sum(W[k]*np.mean(res[k]) for k in W),4); rows.append(row)
df = pd.DataFrame(rows); pd.set_option("display.width",250)
print(df.to_string(index=False))
print("\nALC medio nos 4 binarios (peso igual por benchmark) e nos 2 nao-binarios:")
for name in priors:
    a = df[(df.prior==name)&df.bench.isin(["multi_swebench","real_webagents","researchcodebench","swe_rebench"])].ALC.mean()
    b = df[(df.prior==name)&df.bench.isin(["matharena","mmdocrag"])].ALC.mean()
    print(f"  {name:32s} 4 bin: {a:.4f}   math+mmdoc: {b:.4f}   media 6: {(4*a+2*b)/6:.4f}")
