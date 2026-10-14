"""Builds the formative-phase ZIPs in formativo/zips/ from an explicit allowlist.

    paiec/.venv/Scripts/python paiec/formativo/build_zips.py

Each arm gets model.py at the ZIP root plus its params.json (and notes.csv for v6).
Nothing else goes in: no __pycache__, tokens, local_fit data or raw training files.
"""
import os
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

HERE = os.path.dirname(os.path.abspath(__file__))
PAIEC = os.path.dirname(HERE)
A = os.path.join(PAIEC, "lines", "A_bayes")
ARMS = {
    "v6": (os.path.join(A, "submit_v6"), ("model.py", "params.json", "notes.csv")),
    "l0": (os.path.join(A, "submit_v7_l0"), ("model.py", "params.json")),
    "l3": (os.path.join(A, "submit_v7_l3"), ("model.py", "params.json")),
    "l6": (os.path.join(A, "submit_v7_l6"), ("model.py", "params.json")),
    "J1": (os.path.join(PAIEC, "lines", "J_robusto", "submit_J1_rob"), ("model.py", "params.json")),
    "J2": (os.path.join(PAIEC, "lines", "J_robusto", "submit_J2_min"), ("model.py", "params.json")),
    "J3": (os.path.join(PAIEC, "lines", "J_robusto", "submit_J3_hib"), ("model.py", "params.json")),
    "J4": (os.path.join(PAIEC, "lines", "J_robusto", "submit_J4_rasch"), ("model.py", "params.json")),
    "mean": (os.path.join(PAIEC, "paiec_baseline", "empirical_mean"), ("model.py", "README.md")),
}
FORBIDDEN = ("token", "hf_token", ".env", "key", "parquet", "response")


def build(arm):
    src, names = ARMS[arm]
    out = os.path.join(HERE, "zips", f"{arm}.zip")
    with ZipFile(out, "w", compression=ZIP_DEFLATED) as z:
        for n in names:
            assert not any(f in n.lower() for f in FORBIDDEN), n
            e = ZipInfo(n)
            e.compress_type = ZIP_DEFLATED
            e.external_attr = 0o100644 << 16
            z.writestr(e, open(os.path.join(src, n), "rb").read())
    return out


if __name__ == "__main__":
    for arm in sys.argv[1:] or ARMS:
        p = build(arm)
        print(arm, p, os.path.getsize(p))
