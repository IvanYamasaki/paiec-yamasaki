"""Load measurement-db tables into the competition's input format.

Layout expected under the data root (download_data.py or make_synthetic.py):
    <root>/<benchmark_dir>/{benchmarks,subjects,items,response|responses}.parquet

Only binary, item-granularity benchmarks are kept, as in the official
prepare_data.py. Responses are reduced to one observation per (subject, item):
by default a seeded random trial, so repeated trials are not averaged into soft
labels (the hidden test scores single 0/1 outcomes).
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd

SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date",
                  "harness", "harness_version", "reasoning_effort", "subject_features_extra")


def _s(value):
    """Competition rule: every value is a plain string, missing is ''."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    return str(value)


def anonymous_benchmark_id(name, salt="paiec-local"):
    return "bench_" + hashlib.sha256(f"{salt}:{name}".encode()).hexdigest()[:12]


def benchmark_dirs(root):
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isfile(os.path.join(p, "benchmarks.parquet")):
            out.append(d)
    return out


def _read_responses(path):
    for name in ("response.parquet", "responses.parquet"):
        f = os.path.join(path, name)
        if os.path.exists(f):
            return pd.read_parquet(f)
    raise FileNotFoundError(f"no response table in {path}")


def load(root, seed=0, eligible_only=True):
    """Return (obs, subjects, items, meta).

    obs: one row per (subject_id, item_id) with columns
         subject_id, item_id, bench, label (0/1), interactors, n_trials
    subjects: subject_id -> dict of the 8 competition string fields
    items: item_id -> dict(item_content, item_features, benchmark_id)
    meta: per-benchmark info (name, response_type, granularity, counts)
    """
    rng = np.random.default_rng(seed)
    # Non-eligible benchmarks draw trials from their own stream, so loading them
    # (--stress) does not change the sampled trials of the eligible ones.
    other_rng = np.random.default_rng([seed, 1_000_003])
    obs_parts, subjects, items, meta = [], {}, {}, {}
    for d in benchmark_dirs(root):
        path = os.path.join(root, d)
        b = pd.read_parquet(os.path.join(path, "benchmarks.parquet")).iloc[0]
        ok = b.get("response_type") == "binary" and b.get("granularity") == "item"
        meta[d] = {"name": _s(b.get("name")), "response_type": _s(b.get("response_type")),
                   "granularity": _s(b.get("granularity")), "eligible": bool(ok),
                   "anon_id": anonymous_benchmark_id(d)}
        if eligible_only and not ok:
            continue
        subj = pd.read_parquet(os.path.join(path, "subjects.parquet"))
        for row in subj.to_dict("records"):
            subjects[row["subject_id"]] = {k: _s(row.get(k)) for k in SUBJECT_FIELDS}
        it = pd.read_parquet(os.path.join(path, "items.parquet"),
                             columns=None)
        anon = meta[d]["anon_id"]
        for row in it.to_dict("records"):
            items[row["item_id"]] = {"item_content": _s(row.get("content")),
                                     "item_features": _s(row.get("item_features")),
                                     "benchmark_id": anon}
        r = _read_responses(path)
        r = r[r["response"].notna()].copy()
        r["label"] = (r["response"].astype(float) >= 0.5).astype(int)
        r["interactors"] = r["interactors"].map(_s) if "interactors" in r else ""
        r["_u"] = (rng if ok else other_rng).random(len(r))
        n_trials = r.groupby(["subject_id", "item_id"]).size().rename("n_trials")
        r = r.sort_values("_u").drop_duplicates(["subject_id", "item_id"], keep="first")
        r = r.join(n_trials, on=["subject_id", "item_id"])
        r["bench"] = d
        obs_parts.append(r[["subject_id", "item_id", "bench", "label", "interactors", "n_trials"]])
        meta[d].update(n_items=int(len(it)), n_subjects=int(len(subj)),
                       n_response_rows=int(len(_read_responses(path))))
    obs = pd.concat(obs_parts, ignore_index=True)
    return obs, subjects, items, meta


def make_input(subjects, items, subject_id, item_id, interactors):
    item = dict(items[item_id])
    item["interactors"] = interactors
    return [dict(subjects[subject_id]), item]
