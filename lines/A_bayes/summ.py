import json, os, sys, numpy as np
for tag in sys.argv[1:]:
    d = [json.load(open(f"results/{tag}_seed{s}.json")) for s in range(3) if os.path.exists(f"results/{tag}_seed{s}.json")]
    if not d: print(tag, "none"); continue
    a = [x["brier_alc"] for x in d]
    st = ""
    if "stress" in d[0]:
        st = " stress " + str({f.split(":")[1]: round(np.mean([x["stress"]["folds"][f]["brier_alc"] for x in d]), 4) for f in d[0]["stress"]["folds"]})
    print(f"{tag:18s} n={len(d)} {np.mean(a):.4f} +- {np.std(a):.4f} strict={all(x.get('strict_ok') for x in d)} macro={np.mean([x['brier_alc_bench_macro'] for x in d]):.4f}",
          " ".join(f"B{b}={np.mean([x[f'B{b}'] for x in d]):.4f}" for b in (0, 1, 3, 7, 15, 31)), st)
