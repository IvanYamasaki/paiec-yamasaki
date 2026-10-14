"""Packaging check: does a model survive the hidden test's environment?

    paiec/.venv/Scripts/python paiec/lines/D_calib/pack/check_pack.py --model paiec/models/beta_pair

Builds cases with the full venv (bench_data), then runs worker.py with .venv_min
(numpy, pandas, pyarrow only) in a fresh subprocess. The worker imports model.py without
local_fit, calls predict(input, labeled) item by item, and reports import errors (with the
missing module and where), ms per call by (benchmark, N, k), range, purity and size.
Writes pack/report_<model name>.json.
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAIEC = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, PAIEC)
import bench_data  # noqa: E402

BENCHES = ("multi_swebench", "real_webagents")
KS = (0, 1, 7, 31)
NS = (9, 81)


def build_cases(data_root, seed, n_targets, n_items):
    obs, subjects, items, _ = bench_data.load(data_root, seed=seed)
    rng = np.random.default_rng(seed)
    cases = []
    for bench in BENCHES:
        sub = obs[obs["bench"] == bench]
        groups = {sid: g for sid, g in sub.groupby("subject_id") if g["item_id"].nunique() >= 80}
        pair_ids = sorted(groups)
        order = [pair_ids[i] for i in rng.permutation(len(pair_ids))]
        # one fixed shuffled trajectory per pair, so labels are nested across k
        traj = {}
        for sid in order:
            g = groups[sid].sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
            traj[sid] = g
        for t in range(min(n_targets, len(order))):
            target = order[t]
            tg = traj[target]
            others = order[:t] + order[t + 1:]
            ev = tg.iloc[31:31 + n_items]  # items never revealed as labels
            inputs = [bench_data.make_input(subjects, items, target, r.item_id, r.interactors)
                      for r in ev.itertuples()]
            for N in NS:
                n_use = min(N, len(others))
                for k in KS:
                    labeled = []
                    for sid in [target] + others[:n_use]:
                        for r in traj[sid].iloc[:k].itertuples():
                            labeled.append([bench_data.make_input(subjects, items, sid, r.item_id, r.interactors),
                                            int(r.label)])
                    perm = rng.permutation(len(labeled))
                    labeled = [labeled[i] for i in perm]
                    cases.append({"bench": bench, "N": N, "N_used": n_use, "k": k, "target": str(target),
                                  "inputs": inputs, "labeled": labeled})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(PAIEC, "data", "measurement-db"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", type=int, default=2, help="target pairs per benchmark")
    ap.add_argument("--items", type=int, default=20, help="evaluation items per target pair")
    ap.add_argument("--venv-min", default=os.path.join(HERE, ".venv_min"))
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model)
    name = os.path.basename(model_dir.rstrip("/\\"))
    py_min = os.path.join(args.venv_min, "Scripts", "python.exe")
    if not os.path.exists(py_min):
        py_min = os.path.join(args.venv_min, "bin", "python")
    if not os.path.exists(py_min):
        sys.exit(f"minimal venv not found at {args.venv_min}")

    cases = build_cases(args.data, args.seed, args.targets, args.items)
    tmp = tempfile.mkdtemp(prefix="pack_")
    cases_path, out_path = os.path.join(tmp, "cases.pkl"), os.path.join(tmp, "out.json")
    with open(cases_path, "wb") as fh:
        pickle.dump(cases, fh)

    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run([py_min, os.path.join(HERE, "worker.py"), model_dir, cases_path, out_path],
                          capture_output=True, text=True, env=env)
    if not os.path.exists(out_path):
        sys.exit(f"worker crashed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    rep = json.load(open(out_path))
    rep["model"] = name
    rep["worker_stderr_tail"] = proc.stderr[-1500:]

    rows = rep.get("cases", [])
    allms = [r["mean_ms"] for r in rows if r.get("mean_ms") is not None]
    mean_all = float(np.mean(allms)) if allms else None
    worst = max(rows, key=lambda r: r["mean_ms"] or 0) if rows else None
    rep["worst_case"] = worst
    rep["projected_hours_220k_calls"] = (mean_all * 220000 / 3.6e6) if mean_all is not None else None
    rep["projected_hours_220k_calls_worst_case"] = (worst["mean_ms"] * 220000 / 3.6e6) if worst else None
    rep["all_ok"] = bool(rep["ok_imports"] and rep.get("range_ok") and rep.get("purity_ok")
                         and rep.get("labeled_not_mutated", False) and not rep.get("runtime_errors"))
    out = os.path.join(args.out_dir, f"report_{name}.json")
    json.dump(rep, open(out, "w"), indent=1)

    print(f"== pack check: {name} ==")
    print(f"python worker {rep['python']} | size {rep['size_mb']:.3f} MB | import {rep['import_seconds']:.2f} s "
          f"| local_fit defined: {rep.get('has_local_fit')} (not called) | predict_batch: {rep.get('has_predict_batch')}")
    print(f"ok_imports: {rep['ok_imports']}")
    for e in rep["import_errors"]:
        print(f"  IMPORT ERROR in {e['file']}: {e['error']} at {e['where']}")
    for m in rep["missing_modules"]:
        print(f"  missing module: {m}")
    for e in rep.get("runtime_errors", []):
        print(f"  RUNTIME ERROR case {e['case']}: {e['error']} at {e['where']}")
    if rows:
        print(f"first call {rep['first_call_ms']:.2f} ms | range_ok {rep['range_ok']} | purity_ok {rep['purity_ok']} "
              f"| labeled untouched {rep['labeled_not_mutated']}")
        for key in ("range_bad", "purity_bad"):
            for s in rep.get(key, []):
                print(f"  {key}: {s}")
        print(f"{'benchmark':16s} {'N':>3s} {'k':>3s} {'labeled':>8s} {'p50':>8s} {'p95':>8s} {'mean':>8s}  (ms/call)")
        for r in rows:
            print(f"{r['bench']:16s} {r['N']:3d} {r['k']:3d} {r['n_labeled']:8d} "
                  f"{r['p50_ms']:8.3f} {r['p95_ms']:8.3f} {r['mean_ms']:8.3f}")
        print(f"worst case mean: {worst['mean_ms']:.3f} ms ({worst['bench']}, N={worst['N']}, k={worst['k']}) "
              f"-> {rep['projected_hours_220k_calls_worst_case']:.2f} h per 220k calls; "
              f"average over cases {rep['projected_hours_220k_calls']:.2f} h")
    print(f"ALL OK: {rep['all_ok']}   report: {out}")
    sys.exit(0 if rep["all_ok"] else 1)


if __name__ == "__main__":
    main()
