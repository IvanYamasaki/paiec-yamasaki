"""Non-eligible public benchmarks (matharena, mmdocrag): never a test benchmark,
so their labels can be used as training data in every fold. Same reduction as
bench_data: one seeded random trial per (subject, item), label = response >= .5."""
import os, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EXT = ("matharena", "mmdocrag")
FIELDS = ("normalized_name", "provider", "release_date", "access_date",
          "harness", "harness_version", "reasoning_effort", "subject_features_extra")


def _s(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v)


def load_ext(seed=0, root=os.path.join(ROOT, "data", "measurement-db")):
    rng = np.random.default_rng(seed + 1000)
    parts, subjects = [], {}
    for d in EXT:
        p = os.path.join(root, d)
        if not os.path.exists(os.path.join(p, "response.parquet")):
            continue
        s = pd.read_parquet(os.path.join(p, "subjects.parquet"))
        for row in s.to_dict("records"):
            subjects[row["subject_id"]] = {k: _s(row.get(k)) for k in FIELDS}
        r = pd.read_parquet(os.path.join(p, "response.parquet"), columns=["subject_id", "item_id", "response"])
        r = r[r["response"].notna()].copy()
        r["label"] = (r["response"].astype(float) >= 0.5).astype(int)
        r["_u"] = rng.random(len(r))
        r = r.sort_values("_u").drop_duplicates(["subject_id", "item_id"])
        r["bench"] = d
        parts.append(r[["subject_id", "item_id", "bench", "label"]])
    return pd.concat(parts, ignore_index=True), subjects
