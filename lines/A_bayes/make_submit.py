"""Fit v6 on all public data and write a self-contained submission dir (no local_fit needed).
usage: make_submit.py SRC_VERSION OUT_DIR"""
import os, sys, json, shutil, csv
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import bench_data
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
shutil.copy(os.path.join(src, "model.py"), os.path.join(out, "model.py"))
shutil.copy(os.path.join(ROOT, "lines", "B_conteudo", "module", "scores_sonnet.csv"), os.path.join(out, "notes.csv"))
sys.path.insert(0, out)
import model as M   # loads notes.csv from out
obs, subjects, items, _ = bench_data.load(os.path.join(ROOT, "data", "measurement-db"), seed=0)
ext, ext_subj = M.load_ext(os.path.join(ROOT, "data", "measurement-db"))
subs = dict(subjects); subs.update(ext_subj)
o = obs[["subject_id", "item_id", "bench", "label"]]
P = M.fit_params(pd.concat([o, ext], ignore_index=True), subs, sorted(o.bench.unique()))
P["sd2"], P["rho"], note_p = M.item_var(o, items)
P.update(note_p)
json.dump(P, open(os.path.join(out, "params.json"), "w"))
print({k: (round(v, 3) if isinstance(v, float) else (len(v) if isinstance(v, (dict, list)) else v)) for k, v in P.items()})
