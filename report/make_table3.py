"""Print Table 3 of draft.md (formative Brier ALC per arm) from formativo/RESULTADOS.md.

At the 25/10 cut: run this, paste the table over Table 3 and replace the other
[CORTE] numbers with the summary lines printed at the end.

    python report/make_table3.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "formativo", "RESULTADOS.md")

LABEL = {
    "J2": "J2 minimal",
    "J4": "J4 (J1 without metadata)",
    "J1": "J1 robust level",
    "J3": "J3 hybrid",
    "l0": "A v7 l0",
    "v6": "A v6 (note absent on test items)",
    "l3": "A v7 l3",
    "l6": "A v7 l6",
    "mean": "official empirical mean",
}


def load():
    rows, subs = {}, {}
    for line in open(SRC, encoding="utf8"):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 14 and c[0] in LABEL and c[2].isdigit():
            rows[c[0]] = dict(n=int(c[2]), mean=float(c[3]), sd=float(c[4]), ci=c[5], ece=float(c[6]),
                              b=[float(x) for x in c[7:13]])
        m = re.match(r"\| (\w+) \| \d+\*? \| [^|]+ \| ([0-9.]+) \|", line)
        if m and m.group(1) in LABEL:
            subs.setdefault(m.group(1), []).append(float(m.group(2)))
    return rows, subs


def main():
    rows, subs = load()
    order = sorted((k for k in rows if k != "mean"), key=lambda k: rows[k]["mean"]) + ["mean"]
    print("| arm | n | mean | 95% CI | B0 | B1 | B3 | B7 | B15 | B31 | ECE |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in order:
        r = rows[k]
        b = " | ".join(f"{x:.3f}" for x in r["b"])
        print(f"| {LABEL[k]} | {r['n']} | {r['mean']:.4f} | {r['ci']} | {b} | {r['ece']:.3f} |")
    ours = [k for k in order if k != "mean"]
    print()
    print("valid submissions:", sum(rows[k]["n"] for k in order))
    print("our arms: %.4f to %.4f; baseline %.4f" % (min(rows[k]["mean"] for k in ours),
                                                      max(rows[k]["mean"] for k in ours), rows["mean"]["mean"]))
    print("SD within arm: %.4f to %.4f" % (min(rows[k]["sd"] for k in order), max(rows[k]["sd"] for k in order)))
    for k in ("J2", "J4"):
        if k in subs:
            print(f"{k} single submissions: {min(subs[k]):.4f} to {max(subs[k]):.4f}")


if __name__ == "__main__":
    main()
