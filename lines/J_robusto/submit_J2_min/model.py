"""Line J, minimal arm: per-pair Beta posterior around a benchmark level from other subjects.

level p0 = (S_oth + A0 * 0.5) / (N_oth + A0) from the other subjects' labels of the same
benchmark_id in labeled (0.5 when there are none); prediction = (s + A * p0) / (n + A) from the
pair's own s successes in n labels. No metadata, no item effects, no fitted parameters."""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {"a": 2.0, "a0": 4.0}
if os.path.exists(os.path.join(_HERE, "params.json")):
    CFG.update(json.load(open(os.path.join(_HERE, "params.json"))).get("cfg", {}))


def _key(s):
    return tuple(sorted(s.items()))


def predict(input, labeled=None):
    subject, item = input
    bench, sk = item.get("benchmark_id"), _key(subject)
    s = n = so = no = 0
    for (sub, it), y in labeled or []:
        if it.get("benchmark_id") != bench:
            continue
        if _key(sub) == sk:
            s, n = s + int(y), n + 1
        else:
            so, no = so + int(y), no + 1
    p0 = (so + CFG["a0"] * 0.5) / (no + CFG["a0"])
    return float((s + CFG["a"] * p0) / (n + CFG["a"]))
