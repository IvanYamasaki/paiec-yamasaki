"""Self-contained submission dir (model.py + params.json, no local_fit run at test time).
usage: make_submit_j.py jm|jmin OUT_DIR 'CFG_JSON'
jm: raw params fitted on all public data (4 binary + matharena/mmdocrag for the level), cfg stored in params.json."""
import json, os, shutil, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import bench_data
src, out, cfg = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
os.makedirs(out, exist_ok=True)
shutil.copy(os.path.join(HERE, src, "model.py"), os.path.join(out, "model.py"))
if src == "jmin":
    json.dump({"cfg": cfg}, open(os.path.join(out, "params.json"), "w"))
    print("jmin", cfg); sys.exit()
sys.path.insert(0, os.path.join(HERE, src))
import model as M
obs, subjects, items, _ = bench_data.load(os.path.join(ROOT, "data", "measurement-db"), seed=0)
ext, ext_subj = M.load_ext(os.path.join(ROOT, "data", "measurement-db"))
subs = dict(subjects); subs.update(ext_subj)
o = obs[["subject_id", "item_id", "bench", "label"]]
P = M.fit_params(pd.concat([o, ext], ignore_index=True), subs, sorted(o.bench.unique()))
P["sd2"], P["rho"], note_p = M.item_var(o, items)
P.update(note_p)
P["cfg"] = cfg
json.dump(P, open(os.path.join(out, "params.json"), "w"))
print({k: (round(v, 3) if isinstance(v, float) else (len(v) if isinstance(v, (dict, list)) else v)) for k, v in P.items()})
