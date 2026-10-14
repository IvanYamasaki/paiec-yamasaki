"""Worker of check_pack.py. Run with the minimal venv (numpy, pandas, pyarrow only).

    .venv_min/Scripts/python worker.py MODEL_DIR CASES.pkl OUT.json

Imports MODEL_DIR/model.py (and labeling.py if present) WITHOUT local_fit, then calls
predict(input, labeled) one item at a time, like the hidden test.
"""

import copy
import importlib.util
import json
import math
import os
import pickle
import sys
import tempfile
import time
import traceback

import numpy as np


def dir_size_mb(path):
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1e6


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def describe_failure(exc, model_dir):
    tb = traceback.extract_tb(exc.__traceback__)
    where = None
    for fr in reversed(tb):  # innermost frame that lives in the candidate dir
        if os.path.abspath(fr.filename).startswith(os.path.abspath(model_dir)):
            where = f"{os.path.relpath(fr.filename, model_dir)}:{fr.lineno} ({fr.line})"
            break
    if where is None and tb:
        where = f"{tb[-1].filename}:{tb[-1].lineno}"
    return {"error": f"{type(exc).__name__}: {exc}", "where": where,
            "missing_module": getattr(exc, "name", None) if isinstance(exc, ImportError) else None}


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def main():
    model_dir, cases_path, out_path = sys.argv[1:4]
    model_dir = os.path.abspath(model_dir)
    res = {"ok_imports": False, "missing_modules": [], "import_errors": [], "import_seconds": None,
           "size_mb": dir_size_mb(model_dir), "python": sys.version.split()[0]}
    sys.path.insert(0, model_dir)
    # run from an empty dir: parameters must be found through __file__, not the cwd
    os.chdir(tempfile.mkdtemp(prefix="pack_cwd_"))
    t0 = time.perf_counter()
    model = None
    try:
        model = import_file(os.path.join(model_dir, "model.py"), "model_candidate")
        if not hasattr(model, "predict"):
            raise AttributeError("model.py has no predict")
    except BaseException as exc:  # noqa: BLE001
        info = describe_failure(exc, model_dir)
        info["file"] = "model.py"
        res["import_errors"].append(info)
        if info["missing_module"]:
            res["missing_modules"].append(f'{info["missing_module"]} (model.py, {info["where"]})')
    lab = os.path.join(model_dir, "labeling.py")
    if model is not None and os.path.exists(lab):
        try:
            m = import_file(lab, "labeling_candidate")
            if not hasattr(m, "acquisition_function"):
                raise AttributeError("labeling.py has no acquisition_function")
        except BaseException as exc:  # noqa: BLE001
            info = describe_failure(exc, model_dir)
            info["file"] = "labeling.py"
            res["import_errors"].append(info)
            if info["missing_module"]:
                res["missing_modules"].append(f'{info["missing_module"]} (labeling.py, {info["where"]})')
    res["import_seconds"] = time.perf_counter() - t0
    res["ok_imports"] = not res["import_errors"]
    res["has_local_fit"] = bool(model is not None and hasattr(model, "local_fit"))
    res["has_predict_batch"] = bool(model is not None and hasattr(model, "predict_batch"))
    if model is None:
        json.dump(res, open(out_path, "w"), indent=1)
        return

    with open(cases_path, "rb") as fh:
        cases = pickle.load(fh)

    first_call_ms = None
    range_ok = True
    range_bad = []
    purity_ok = True
    purity_bad = []
    mutation_ok = True
    runtime_errors = []
    rows = []
    for ci, case in enumerate(cases):
        labeled = case["labeled"]
        snapshot = copy.deepcopy(labeled)
        ms, vals = [], []
        try:
            for x in case["inputs"]:
                x = copy.deepcopy(x)
                t = time.perf_counter()
                v = model.predict(x, labeled)
                dt = (time.perf_counter() - t) * 1000
                if first_call_ms is None:
                    first_call_ms = dt
                    continue_ms = False
                else:
                    continue_ms = True
                if continue_ms:
                    ms.append(dt)
                vals.append(v)
                ok = isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool) \
                    and math.isfinite(float(v)) and 0.0 <= float(v) <= 1.0
                if not ok:
                    range_ok = False
                    if len(range_bad) < 5:
                        range_bad.append(f"case {ci}: {v!r} ({type(v).__name__})")
            # purity: same input again, interleaved with others (reverse order)
            for i in reversed(range(len(case["inputs"]))):
                if i >= len(vals):
                    break
                v2 = model.predict(copy.deepcopy(case["inputs"][i]), labeled)
                if not (float(v2) == float(vals[i])):
                    purity_ok = False
                    if len(purity_bad) < 5:
                        purity_bad.append(f"case {ci} item {i}: {vals[i]!r} then {v2!r}")
        except BaseException as exc:  # noqa: BLE001
            info = describe_failure(exc, model_dir)
            runtime_errors.append({"case": ci, **info})
            if info["missing_module"]:
                res["missing_modules"].append(f'{info["missing_module"]} (predict, {info["where"]})')
            break
        if labeled != snapshot:
            mutation_ok = False
        rows.append({"bench": case["bench"], "N": case["N"], "k": case["k"], "n_labeled": len(labeled),
                     "n_calls": len(ms), "p50_ms": pct(ms, 50), "p95_ms": pct(ms, 95),
                     "mean_ms": float(np.mean(ms)) if ms else None, "max_ms": max(ms) if ms else None})
    res.update(first_call_ms=first_call_ms, range_ok=range_ok, range_bad=range_bad, purity_ok=purity_ok,
               purity_bad=purity_bad, labeled_not_mutated=mutation_ok, runtime_errors=runtime_errors,
               cases=rows)
    if runtime_errors:
        res["ok_imports"] = res["ok_imports"] and not res["missing_modules"]
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
