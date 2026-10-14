"""Line J simulator: real model dirs on the eval_local trajectories (pair or benchmark scope),
LOBO folds (4 binary) and stress folds (matharena, mmdocrag binarized; train = 4 binary), with
optional LEVEL SHIFT: the test benchmark's labels are redrawn from its own Rasch fit
(theta_s + d_i + delta), delta set so the benchmark mean rate is q. Same pairs, splits and
acquired items as the evaluator (random acquisition does not depend on labels).

usage: jsim.py CONFIGS.json SCOPE SEEDS SHIFTS FOLDS [MAX_PAIRS]
  CONFIGS.json: list of {"name", "dir", "cfg"}; SCOPE: pair | int (other pairs in labeled)
  SHIFTS: none,0.05,0.5,...; FOLDS: lobo | stress | all
Writes one JSON line per (config, seed, fold, shift) to results/J_sim_<tag>.jsonl."""
import hashlib, importlib.util, json, os, sys, uuid
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "lines", "D_calib"))
import bench_data, sim  # noqa
BUD = (0, 1, 3, 7, 15, 31); W = {0: .1, 1: .2, 3: .2, 7: .2, 15: .2, 31: .1}
ELIG = ["multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
STRESS = ["matharena", "mmdocrag"]


def load(d):
    spec = importlib.util.spec_from_file_location("m_" + uuid.uuid4().hex[:8], os.path.join(d, "model.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


_R = {}
def rasch(obs, b, seed):
    if (b, seed) in _R:
        return _R[(b, seed)]
    g = obs[obs.bench == b]
    sc, si = np.unique(g.subject_id.to_numpy(), return_inverse=True)
    ic, ii = np.unique(g.item_id.to_numpy(), return_inverse=True)
    y = g.label.to_numpy(float); th, d = np.zeros(len(sc)), np.zeros(len(ic))
    for _ in range(300):
        p = 1 / (1 + np.exp(-(th[si] + d[ii])))
        th += (np.bincount(si, y - p, len(th)) - 0.05 * th) / (np.bincount(si, p * (1 - p), len(th)) + 0.05)
        p = 1 / (1 + np.exp(-(th[si] + d[ii])))
        d += (np.bincount(ii, y - p, len(d)) - 0.3 * d) / (np.bincount(ii, p * (1 - p), len(d)) + 0.3)
    eta = th[si] + d[ii]
    _R[(b, seed)] = (dict(zip(sc, th)), dict(zip(ic, d)), eta)
    return _R[(b, seed)]


def delta_for(eta, q):
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(eta + mid)))) < q: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def shifted(pairs, obs, b, seed, q):
    if q is None:
        return [(p, p["y"], np.asarray(p["labs"])) for p in pairs]
    th, d, eta = rasch(obs, b, seed); dl = delta_for(eta, q); out = []
    for p in pairs:
        h = hashlib.sha256(f"{seed}|{b}|{q}|{p['subject_id']}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        t = th[p["subject_id"]] + dl
        y = (rng.random(len(p["eval_items"])) < 1 / (1 + np.exp(-(t + np.array([d[i] for i in p["eval_items"]]))))).astype(int)
        la = (rng.random(len(p["acq_items"])) < 1 / (1 + np.exp(-(t + np.array([d[i] for i in p["acq_items"]]))))).astype(int)
        out.append((p, y, la))
    return out


def run(confs, scope, seed, shifts, folds, max_pairs, tag):
    T = sim.trajectories_all(seed); obs, S, I, _ = sim.data_all(seed)
    o = obs[["subject_id", "item_id", "bench", "label"]]
    mk = lambda s, i: bench_data.make_input(S, I, s, i, "")
    benches = (ELIG if folds in ("lobo", "all") else []) + (STRESS if folds in ("stress", "all") else [])
    fout = open(os.path.join(ROOT, "results", f"J_sim_{tag}.jsonl"), "a")
    for b in benches:
        train = o[o.bench.isin(ELIG) & (o.bench != b)].reset_index(drop=True)
        P = [r for r in T if r["bench"] == b]
        if max_pairs and len(P) > max_pairs:
            P = [P[t] for t in np.random.default_rng(seed).choice(len(P), max_pairs, replace=False)]
        evs = {id(p): [mk(p["subject_id"], i) for i in p["eval_items"]] for p in P}
        for c in confs:
            M = load(os.path.join(HERE, c["dir"]))
            M.CFG.update(c.get("cfg", {}))
            if hasattr(M, "local_fit"):
                M.local_fit(train, S, I)
            elif hasattr(M, "apply_cfg"):
                M.apply_cfg()
            for q in shifts:
                data = shifted(P, o, b, seed, q)
                rng = np.random.default_rng(seed + 99); recs = []
                for j, (p, y, la) in enumerate(data):
                    others = [x for x in data if x[0] is not p]
                    if scope != "pair" and len(others) > scope:
                        others = [others[t] for t in rng.choice(len(others), scope, replace=False)]
                    r = {}
                    for k in BUD:
                        lab = [[mk(p["subject_id"], i), int(l)] for i, l in zip(p["acq_items"][:k], la[:k])]
                        if scope != "pair":
                            lab += [[mk(x[0]["subject_id"], i), int(l)] for x in others for i, l in zip(x[0]["acq_items"][:k], x[2][:k])]
                        pr = np.array(M.predict_batch(evs[id(p)], lab)) if hasattr(M, "predict_batch") else np.array([M.predict(e, lab) for e in evs[id(p)]])
                        r[k] = float(np.mean((pr - y) ** 2))
                    recs.append(r)
                res = {f"B{k}": float(np.mean([r[k] for r in recs])) for k in BUD}
                res["alc"] = float(np.mean([sum(W[k] * r[k] for k in BUD) for r in recs]))
                line = {"conf": c["name"], "seed": seed, "fold": b, "shift": q, "scope": scope, "n": len(recs), **res}
                fout.write(json.dumps(line) + "\n"); fout.flush()
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in line.items()}), flush=True)


if __name__ == "__main__":
    confs = json.load(open(sys.argv[1]))
    scope = "pair" if sys.argv[2] == "pair" else int(sys.argv[2])
    shifts = [None if s == "none" else float(s) for s in sys.argv[4].split(",")]
    mp = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    tag = os.path.splitext(os.path.basename(sys.argv[1]))[0] + f"_{sys.argv[2]}_{sys.argv[5]}"
    for s in sys.argv[3].split(","):
        run(confs, scope, int(s), shifts, sys.argv[5], mp, tag)
