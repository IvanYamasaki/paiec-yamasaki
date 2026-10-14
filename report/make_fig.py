"""Figure 1 of draft.md: local bench/10 Brier ALC against the formative mean, one point per arm.

    python report/make_fig.py      (needs matplotlib; the project .venv does not have it)
"""
import json
import os
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
RUNS = os.path.join(HERE, "runs")
sys.path.insert(0, HERE)
from make_table3 import load  # noqa: E402

B10 = {
    "v6": [f"{RES}/A_v6n1_b10_seed0.json", f"{RUNS}/v6_b10_s1.json", f"{RES}/A_v6n1_b10_seed2.json"],
    "l0": [f"{RES}/A_v7l0_b10_seed{s}.json" for s in range(3)],
    "l3": [f"{RES}/A_v7l3_b10_seed{s}.json" for s in range(3)],
    "l6": [f"{RES}/A_v7l6_b10_seed{s}.json" for s in range(3)],
    "J1": [f"{RES}/J_J1_b10_s{s}.json" for s in range(3)],
    "J2": [f"{RES}/J_J2_b10_s{s}.json" for s in range(3)],
    "J3": [f"{RUNS}/J3_b10_s{s}.json" for s in range(3)],
    "J4": [f"{RUNS}/J4_b10_s{s}.json" for s in range(3)],
}
NAME = {"v6": "A v6", "l0": "A l0", "l3": "A l3", "l6": "A l6", "J1": "J1", "J2": "J2", "J3": "J3", "J4": "J4"}
OFF = {"J1": (3, 7), "J3": (-16, -10), "J4": (6, -10), "l0": (-28, 3)}
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#6b6b6b"


def main():
    rows, _ = load()
    fig, ax = plt.subplots(figsize=(5.4, 4.0), dpi=200)
    for k, files in B10.items():
        v = [json.load(open(f))["brier_alc"] for f in files if os.path.exists(f)]
        if len(v) < 2 or k not in rows:
            continue
        r = rows[k]
        lo, hi = (float(x) for x in r["ci"].strip("[]").split(","))
        x, y = st.mean(v), r["mean"]
        c = BLUE if k in ("v6", "l0", "l3", "l6") else ORANGE
        ax.errorbar(x, y, xerr=st.stdev(v), yerr=[[y - lo], [hi - y]], fmt="o", ms=5, color=c,
                    ecolor=c, elinewidth=1, capsize=0, markeredgecolor="white", markeredgewidth=1, zorder=3)
        ax.annotate(NAME[k], (x, y), xytext=OFF.get(k, (5, 4)), textcoords="offset points", fontsize=8, color=INK)
    lim = (0.155, 0.235)
    ax.plot(lim, lim, color=MUTED, lw=0.8, zorder=1)
    ax.axhline(rows["mean"]["mean"], color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.text(lim[0] + 0.001, rows["mean"]["mean"] + 0.001, "official empirical mean (formative)", fontsize=7, color=MUTED)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("local LOBO, bench/10 scope (Brier ALC)", fontsize=9, color=INK)
    ax.set_ylabel("formative hidden test (Brier ALC)", fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(color="#e6e6e3", lw=0.5, zorder=0)
    h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, label="line A (public-pool priors)"),
         plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="line J (robust variants)")]
    ax.legend(handles=h, fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    out = os.path.join(HERE, "fig_lobo_vs_formativo.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
