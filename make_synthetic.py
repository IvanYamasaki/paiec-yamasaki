"""Synthetic measurement-db lookalike to test the evaluator without the gated data.

2PL IRT ground truth: P(y=1) = sigmoid(a_i * (theta_s + delta_sb - b_i)).
    .venv/Scripts/python make_synthetic.py --out data/synthetic
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    names = [f"model-{i:02d}" for i in range(40)]
    theta = dict(zip(names, rng.normal(0, 1, len(names))))
    for bi, bench in enumerate(["bench_a", "bench_b", "bench_c", "bench_d", "bench_e", "bench_f"]):
        d = os.path.join(args.out, bench)
        os.makedirs(d, exist_ok=True)
        n_items = int(rng.integers(90, 400))
        level = rng.normal(0, 1.2)
        pd.DataFrame([{"benchmark_id": bench, "name": bench, "response_type": "binary",
                       "granularity": "item"}]).to_parquet(os.path.join(d, "benchmarks.parquet"))
        chosen = rng.choice(names, size=int(rng.integers(10, 30)), replace=False)
        subj = pd.DataFrame([{"subject_id": f"{bench}:{n}", "normalized_name": n, "provider": n[:7],
                              "release_date": f"2025-{1 + i % 12:02d}-01", "access_date": "",
                              "harness": "h" if i % 2 else "", "harness_version": "",
                              "reasoning_effort": ["", "low", "high"][i % 3],
                              "subject_features_extra": ""} for i, n in enumerate(chosen)])
        subj.to_parquet(os.path.join(d, "subjects.parquet"))
        b = rng.normal(level, 1.0, n_items)
        a = rng.lognormal(0, 0.3, n_items)
        items = pd.DataFrame({"item_id": [f"{bench}:{i}" for i in range(n_items)], "benchmark_id": bench,
                              "content": [f"Question {i} " + "x" * int(rng.integers(10, 500)) for i in range(n_items)],
                              "item_features": [json.dumps({"diff_hint": round(float(v + rng.normal(0, 1)), 2)}) for v in b]})
        items.to_parquet(os.path.join(d, "items.parquet"))
        rows = []
        for s in subj.itertuples():
            delta = rng.normal(0, 0.5)
            keep = rng.random(n_items) < 0.9
            p = 1 / (1 + np.exp(-a * (theta[s.normalized_name] + delta - b)))
            y = (rng.random(n_items) < p).astype(float)
            for i in np.flatnonzero(keep):
                rows.append({"subject_id": s.subject_id, "item_id": f"{bench}:{i}", "response_id": f"r{len(rows)}",
                             "benchmark_id": bench, "trial": 1, "test_condition": "", "interactors": "",
                             "response": y[i]})
        pd.DataFrame(rows).to_parquet(os.path.join(d, "response.parquet"))
    print("written", args.out)


if __name__ == "__main__":
    main()
