"""Describe the training data: sizes, base rates, field fill rates, text length, overlap.

    .venv/Scripts/python describe_data.py [--data data/measurement-db]
Writes results/data_summary.json and prints a readable summary.
"""

import argparse
import collections
import hashlib
import json
import os

import numpy as np
import pandas as pd

import bench_data

HERE = os.path.dirname(os.path.abspath(__file__))


def keys_of(js):
    try:
        v = json.loads(js)
        return list(v) if isinstance(v, dict) else [type(v).__name__]
    except Exception:
        return ["<non-json>"] if js else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "measurement-db"))
    ap.add_argument("--embed", default=os.path.join(HERE, "data", "measurement-db-embed"))
    args = ap.parse_args()
    obs, subjects, items, meta = bench_data.load(args.data, seed=0, eligible_only=False)
    out = {"benchmarks": {}, "totals": {}}
    name_benches = collections.defaultdict(set)
    for d, g in obs.groupby("bench"):
        its = [items[i] for i in g["item_id"].unique() if i in items]
        lens = np.array([len(x["item_content"]) for x in its])
        pair_n = g.groupby("subject_id")["item_id"].nunique()
        fkeys = collections.Counter(k for x in its for k in keys_of(x["item_features"]))
        subj_ids = g["subject_id"].unique()
        for s in subj_ids:
            name_benches[subjects[s]["normalized_name"]].add(d)
        fill = {f: float(np.mean([subjects[s][f] != "" for s in subj_ids])) for f in bench_data.SUBJECT_FIELDS}
        out["benchmarks"][d] = {
            **meta[d], "n_subjects_with_responses": int(len(subj_ids)),
            "n_items_with_responses": int(g["item_id"].nunique()),
            "n_subject_item_obs": int(len(g)), "multi_trial_share": float((g["n_trials"] > 1).mean()),
            "base_rate": float(g["label"].mean()),
            "pairs_ge_80_items": int((pair_n >= 80).sum()),
            "pair_rate_sd": float(g.groupby("subject_id")["label"].mean().std()),
            "content_chars": {"median": float(np.median(lens)), "p90": float(np.percentile(lens, 90)),
                              "max": int(lens.max()), "total_MB": float(lens.sum() / 1e6),
                              "empty_share": float(np.mean(lens == 0))},
            "item_features_nonempty": float(np.mean([x["item_features"] != "" for x in its])),
            "item_features_top_keys": fkeys.most_common(12),
            "interactors_nonempty": float((g["interactors"] != "").mean()),
            "interactors_examples": g.loc[g["interactors"] != "", "interactors"].drop_duplicates().head(3).tolist(),
            "subject_field_fill": fill,
        }
        emb = os.path.join(args.embed, f"{d}.parquet")
        if os.path.exists(emb):
            e = pd.read_parquet(emb, columns=["item_id", "content_sha1"])
            sha = {i: hashlib.sha1(items[i]["item_content"].encode()).hexdigest() for i in items}
            ok = e.apply(lambda r: sha.get(r.item_id) == r.content_sha1, axis=1)
            out["benchmarks"][d]["embeddings"] = {"rows": int(len(e)), "id_and_hash_match": float(ok.mean())}
    out["totals"] = {
        "benchmarks": len(meta), "eligible_benchmarks": int(sum(m["eligible"] for m in meta.values())),
        "subject_ids": int(obs["subject_id"].nunique()),
        "normalized_names": len(name_benches),
        "names_in_2plus_benchmarks": int(sum(len(v) >= 2 for v in name_benches.values())),
        "items": int(obs["item_id"].nunique()), "subject_item_obs": int(len(obs)),
        "response_rows": int(sum(m.get("n_response_rows", 0) for m in meta.values())),
        "base_rate": float(obs["label"].mean()),
        "eligible_pairs_ge_80": int(sum(b["pairs_ge_80_items"] for b in out["benchmarks"].values())),
    }
    out["name_benchmark_coverage"] = {k: sorted(v) for k, v in sorted(name_benches.items())}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "data_summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out["totals"], indent=2))
    for d, b in out["benchmarks"].items():
        print(f"\n== {d}: eligible={b['eligible']} subj={b['n_subjects_with_responses']} "
              f"items={b['n_items_with_responses']} obs={b['n_subject_item_obs']} rate={b['base_rate']:.3f} "
              f"pairs80={b['pairs_ge_80_items']} pair_sd={b['pair_rate_sd']:.3f} multi_trial={b['multi_trial_share']:.2f}")
        print("   content", b["content_chars"], "| features", round(b["item_features_nonempty"], 2),
              b["item_features_top_keys"][:6], "| interactors", round(b["interactors_nonempty"], 2))
        print("   subject fill", {k: round(v, 2) for k, v in b["subject_field_fill"].items()})
        if "embeddings" in b:
            print("   embeddings", b["embeddings"])


if __name__ == "__main__":
    main()
