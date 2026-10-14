"""Local PAIEC evaluator: Brier ALC over budgets 0/1/3/7/15/31.

Protocol, following https://aimslab.stanford.edu/competition and
paiec_baseline/tools/streaming_ingestion.py:
  * every subject-benchmark pair with >= 80 distinct items is a pair;
  * its items are split 50/50 at random into an acquisition pool and an
    evaluation pool; only evaluation items are scored;
  * acquisition items are streamed one by one; labeling.py's
    acquisition_function(input, prediction, labeled, context) -> bool decides
    whether to buy each label (without labeling.py: the official default,
    uniform selection sampling, P(query) = labels_remaining / items_remaining);
  * the evaluation pool is scored with 0, 1, 3, 7, 15 and 31 labels from the
    same nested trajectory; labeled only holds that pair's revealed labels;
  * Brier ALC = .1 B0 + .2 B1 + .2 B3 + .2 B7 + .2 B15 + .1 B31, pairs weighted
    equally.

Validation splits (the hidden test uses benchmarks that are NOT in the public
training data, so --split lobo is the primary one):
  lobo      leave one benchmark out: fit on 5 benchmarks, score pairs of the 6th
  pairs     hold out 20% of subject-benchmark pairs (benchmarks seen in training)
  subjects  hold out 20% of normalized_name values across all benchmarks

Model contract: --model DIR with model.py (predict) and optional labeling.py.
If model.py defines local_fit(train_obs, subjects, items), the evaluator calls
it once per fold on a fresh module copy, with only the fold's training rows
(DataFrame columns subject_id, item_id, bench, label, interactors, n_trials).
An optional predict_batch(inputs, labeled) -> list[float] is used when present.

    .venv/Scripts/python eval_local.py --model models/constant --seed 0
Writes results/<name>_<split>_seed<seed>.json and prints that JSON as the last line.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import types
import uuid

import numpy as np
import pandas as pd

import bench_data

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
MAX_LABELS = 31
STRICT_TOL = 1e-9
HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(f"{name}_{uuid.uuid4().hex[:8]}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Submission:
    def __init__(self, model_dir):
        self.dir = os.path.abspath(model_dir)
        if self.dir not in sys.path:
            sys.path.insert(0, self.dir)
        self.model = load_module(os.path.join(self.dir, "model.py"), "model")
        lab = os.path.join(self.dir, "labeling.py")
        self.acq = load_module(lab, "labeling").acquisition_function if os.path.exists(lab) else None

    def predict_many(self, inputs, labeled):
        # Same deduplication as the official client: identical inputs share one call.
        keys = [json.dumps(x, sort_keys=True) for x in inputs]
        unique = {}
        for k, x in zip(keys, inputs):
            unique.setdefault(k, x)
        ukeys = list(unique)
        # One copy of labeled per checkpoint (it can hold thousands of entries with
        # --labeled-scope benchmark); a model that mutates it is rejected below.
        shared = copy.deepcopy(labeled)
        if hasattr(self.model, "predict_batch"):
            values = self.model.predict_batch([copy.deepcopy(unique[k]) for k in ukeys], shared)
        else:
            values = [self.model.predict(copy.deepcopy(unique[k]), shared) for k in ukeys]
        if shared != labeled:
            raise ValueError("model mutated the labeled list; predict must treat it as read-only")
        cache = {k: check(v) for k, v in zip(ukeys, values)}
        return np.array([cache[k] for k in keys])

    # --strict: the official client calls predict(input, labeled) one item at a time,
    # in workers recreated at every checkpoint. These checks catch models whose
    # local scores depend on predict_batch or on state carried between calls.
    def _state_names(self):
        skip = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)
        return [k for k, v in vars(self.model).items()
                if not k.startswith("__") and not isinstance(v, skip)]

    def snapshot(self):
        """Module globals right after local_fit: our stand-in for a freshly started worker."""
        try:
            self._snap = {k: copy.deepcopy(getattr(self.model, k)) for k in self._state_names()}
        except Exception as exc:
            raise StrictError(f"cannot deepcopy model globals after local_fit ({exc!r}); "
                              "strict mode needs copyable module state") from exc

    def strict_check(self, inputs, labeled, p_normal, where):
        failures = []
        unique = list({json.dumps(x, sort_keys=True): x for x in inputs}.values())
        keys = [json.dumps(x, sort_keys=True) for x in inputs]
        index = {json.dumps(x, sort_keys=True): i for i, x in enumerate(unique)}
        shared = copy.deepcopy(labeled)  # one copy per checkpoint, as in predict_many
        one = lambda x: check(self.model.predict(copy.deepcopy(x), shared))
        reference = p_normal  # what the evaluator scored
        if hasattr(self.model, "predict_batch"):
            p_item = np.array([one(x) for x in unique])[[index[k] for k in keys]]
            reference = p_item  # keep the state check separate from the batch check
            gap = float(np.max(np.abs(p_item - p_normal)))
            if gap > STRICT_TOL:
                failures.append(f"{where}: predict_batch differs from item-by-item predict "
                                f"(max |diff| = {gap:.3g}); the hidden test only calls predict")
        # Fresh-worker replay: restore the post-local_fit globals, predict items one by one
        # in reverse order, compare with the normal run, then put the live state back.
        live = dict(vars(self.model))
        for k in self._state_names():
            if k not in self._snap:
                delattr(self.model, k)
        for k, v in self._snap.items():
            setattr(self.model, k, copy.deepcopy(v))
        try:
            fresh = [None] * len(unique)
            for i in reversed(range(len(unique))):
                fresh[i] = one(unique[i])
        finally:
            vars(self.model).clear()
            vars(self.model).update(live)
        gap = float(np.max(np.abs(np.array(fresh)[[index[k] for k in keys]] - reference)))
        if gap > STRICT_TOL:
            failures.append(f"{where}: predictions depend on state kept between calls "
                            f"(fresh-worker replay differs by {gap:.3g}); the hidden test "
                            "recreates workers at each checkpoint")
        return failures


class StrictError(RuntimeError):
    pass


def check(value):
    v = float(value)
    if not math.isfinite(v) or v < 0 or v > 1:
        raise ValueError(f"prediction outside [0,1]: {value}")
    return v


def eligible_pairs(obs, min_items):
    counts = obs.groupby(["subject_id", "bench"])["item_id"].nunique()
    return [k for k, n in counts.items() if n >= min_items]


def make_folds(obs, subjects, pairs, split, rng, holdout_frac):
    pair_df = pd.DataFrame(pairs, columns=["subject_id", "bench"])
    if split == "lobo":
        for b in sorted(pair_df["bench"].unique()):
            yield b, obs[obs["bench"] != b], pair_df[pair_df["bench"] == b]
    elif split == "pairs":
        mask = rng.random(len(pair_df)) < holdout_frac
        test = pair_df[mask]
        key = obs["subject_id"] + "\x00" + obs["bench"]
        tkey = set(test["subject_id"] + "\x00" + test["bench"])
        yield "pairs", obs[~key.isin(tkey)], test
    elif split == "subjects":
        names = sorted({subjects[s]["normalized_name"] for s in obs["subject_id"].unique()})
        held = {n for n in names if rng.random() < holdout_frac}
        is_held = obs["subject_id"].map(lambda s: subjects[s]["normalized_name"] in held)
        test = pair_df[pair_df["subject_id"].map(lambda s: subjects[s]["normalized_name"] in held)]
        yield "subjects", obs[~is_held], test
    else:
        raise ValueError(split)


def pair_rng(seed, subject_id, bench):
    # Pool split, stream order and random acquisition depend only on (seed, pair),
    # so every model sees the same items in the same order.
    h = hashlib.sha256(f"{seed}|{subject_id}|{bench}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


class PairSim:
    """One subject-benchmark pair: pools, acquisition stream and nested label trajectory.

    The evaluator drives every pair through the checkpoints in lockstep (stage by stage),
    so with --labeled-scope benchmark a pair can see the other pairs' labels at the same
    checkpoint. extra = labels from other pairs (empty with --labeled-scope pair).
    """

    def __init__(self, sub, rows, subjects, items, rng, max_items, seed, strict=False, failures=None):
        rows = rows.sample(frac=1.0, random_state=int(rng.integers(2**31))).reset_index(drop=True)
        if max_items and len(rows) > max_items:
            rows = rows.iloc[:max_items]
        half = len(rows) // 2
        acq_rows, eval_rows = rows.iloc[:half], rows.iloc[half:]
        self.mk = lambda r: bench_data.make_input(subjects, items, r.subject_id, r.item_id, r.interactors)
        self.eval_inputs = [self.mk(r) for r in eval_rows.itertuples()]
        self.y = eval_rows["label"].to_numpy()
        self.sid, self.bench = rows.iloc[0].subject_id, rows.iloc[0].bench
        self.bid = self.eval_inputs[0][1]["benchmark_id"]
        self.sub, self.rng, self.seed = sub, rng, seed
        self.strict, self.failures = strict, failures
        self.acq_list = list(acq_rows.itertuples())
        self.cursor = 0
        self.own = []  # this pair's labels, in acquisition order
        self.out = {}
        self.visible_31 = 0
        self.stats = {"n_eval": len(eval_rows), "n_acq_pool": len(acq_rows)}

    def visible(self, extra, tag):
        if not extra:
            return self.own
        merged = self.own + extra
        h = hashlib.sha256(f"{self.seed}|{self.sid}|{self.bench}|{tag}|shuffle".encode()).digest()
        order = np.random.default_rng(int.from_bytes(h[:8], "big")).permutation(len(merged))
        return [merged[i] for i in order]

    def advance(self, target, extra):
        """Consume the stream until this pair holds `target` labels (or the pool runs out)."""
        while len(self.own) < target and self.cursor < len(self.acq_list):
            i, r = self.cursor, self.acq_list[self.cursor]
            self.cursor += 1
            x = self.mk(r)
            context = {"subject_id": self.sid, "benchmark_id": self.bid, "labels_acquired": len(self.own),
                       "labels_remaining": MAX_LABELS - len(self.own), "max_labels": MAX_LABELS,
                       "items_remaining": len(self.acq_list) - i}
            if self.sub.acq is None:
                query = self.rng.random() < min(1.0, context["labels_remaining"] / context["items_remaining"])
            else:
                labeled = self.visible(extra, f"acq{i}")
                pred = check(self.sub.model.predict(copy.deepcopy(x), copy.deepcopy(labeled)))
                query = self.sub.acq(copy.deepcopy(x), prediction=pred, labeled=copy.deepcopy(labeled),
                                     context=copy.deepcopy(context))
                if type(query) is not bool:
                    raise ValueError("acquisition_function must return bool")
            if query:
                self.own.append([x, int(r.label)])

    def evaluate(self, budget, extra):
        # If the policy left budget unused, the checkpoint is scored with the labels it has.
        labeled = self.visible(extra, budget)
        p = self.sub.predict_many(self.eval_inputs, labeled)
        self.out[budget] = float(np.mean((p - self.y) ** 2))
        self.visible_31 = len(labeled)
        if self.strict:
            self.failures.extend(self.sub.strict_check(self.eval_inputs, labeled, p,
                                                       f"pair ({self.sid}, budget {budget})"))

    def result(self):
        return self.out, {**self.stats, "labels_used": len(self.own), "eval_base_rate": float(self.y.mean()),
                          "labels_visible_at_31": self.visible_31}


def run_group(sims, scope):
    """Run the pairs of one benchmark through the checkpoints together.

    At checkpoint k each pair first advances its own stream to k labels; acquisition calls
    see the other pairs' labels from the previous checkpoint. Scoring at k then sees the
    pair's k labels plus k labels from every other pair (scope "benchmark").
    """
    prev = {id(s): [] for s in sims}
    for budget in BUDGETS:
        for s in sims:
            s.advance(budget, prev[id(s)])
        now = {id(s): ([e for t in sims if t is not s for e in t.own[:budget]] if scope == "benchmark" else [])
               for s in sims}
        for s in sims:
            s.evaluate(budget, now[id(s)])
        prev = now


def alc(bs):
    return sum(WEIGHTS[b] * bs[b] for b in BUDGETS)


def summarize(df):
    out = {"n_pairs": int(len(df)), "brier_alc": float(df["alc"].mean()),
           **{f"B{b}": float(df[f"B{b}"].mean()) for b in BUDGETS},
           "base_rate": float(df["eval_base_rate"].mean())}
    return out


def run_fold(args, sub_factory, fold, train_obs, test_pairs, grouped, subjects, items, strict_state):
    """Score one fold. Returns per-pair records. strict_state = [failures list, pairs checked]."""
    if args.max_pairs_per_bench:
        # Caps the co-sampled pairs per benchmark; with --labeled-scope benchmark this also
        # caps whose labels appear in labeled.
        # (groupby.apply would drop the "bench" column in pandas 3)
        test_pairs = pd.concat([g.sample(min(len(g), args.max_pairs_per_bench), random_state=args.seed)
                                for _, g in test_pairs.groupby("bench")])
    if len(test_pairs) == 0:
        return []
    sub = sub_factory()  # fresh module per fold: no state leaks across folds
    if hasattr(sub.model, "local_fit"):
        sub.model.local_fit(train_obs.reset_index(drop=True), subjects, items)
    if args.strict:
        sub.snapshot()
    sims = []
    for j, p in enumerate(test_pairs.itertuples()):
        strict = args.strict and (j == 0 or pair_rng(args.seed + 7919, p.subject_id, p.bench).random()
                                  < args.strict_frac)
        strict_state[1] += int(strict)
        sims.append(PairSim(sub, grouped[(p.subject_id, p.bench)], subjects, items,
                            pair_rng(args.seed, p.subject_id, p.bench), args.max_items_per_pair,
                            args.seed, strict, strict_state[0]))
    for bench in sorted({s.bench for s in sims}):
        group = [s for s in sims if s.bench == bench]
        if args.labeled_scope == "pair":
            for s in group:
                run_group([s], "pair")
        else:
            run_group(group, "benchmark")
    records = []
    for s in sims:
        bs, st = s.result()
        records.append({"fold": fold, "bench": s.bench, "subject_id": s.sid, **st,
                        **{f"B{b}": bs[b] for b in BUDGETS}, "alc": alc(bs)})
    return records


def measure_latency(args, sub_factory, obs, pairs, grouped, subjects, items, n_calls):
    """Time n_calls real predict(input, labeled) calls, item by item, with labeled at the size
    of --labeled-scope benchmark at checkpoint 31, on the benchmark with the most pairs."""
    pair_df = pd.DataFrame(pairs, columns=["subject_id", "bench"])
    bench = pair_df["bench"].value_counts().idxmax()
    test_pairs = pair_df[pair_df["bench"] == bench]
    if args.max_pairs_per_bench:
        test_pairs = test_pairs.sample(min(len(test_pairs), args.max_pairs_per_bench), random_state=args.seed)
    sub = sub_factory()
    if hasattr(sub.model, "local_fit"):
        sub.model.local_fit(obs[obs["bench"] != bench].reset_index(drop=True), subjects, items)
    sub.acq = None  # trajectories only build labeled here; the default random policy is enough
    sims = [PairSim(sub, grouped[(p.subject_id, p.bench)], subjects, items,
                    pair_rng(args.seed, p.subject_id, p.bench), args.max_items_per_pair, args.seed)
            for p in test_pairs.itertuples()]
    for s in sims:
        s.advance(MAX_LABELS, [])
    times, sizes, k = [], [], 0
    while len(times) < n_calls:
        s = sims[k % len(sims)]
        labeled = copy.deepcopy(s.visible([e for t in sims if t is not s for e in t.own], "latency"))
        for x in s.eval_inputs[: max(1, n_calls // len(sims) + 1)]:
            if len(times) >= n_calls:
                break
            x = copy.deepcopy(x)
            t = time.perf_counter()
            check(sub.model.predict(x, labeled))
            times.append((time.perf_counter() - t) * 1000)
            sizes.append(len(labeled))
        k += 1
    ms = np.array(times)
    return {"bench": bench, "n_calls": int(len(ms)), "labeled_size_mean": float(np.mean(sizes)),
            "first_call_ms": float(ms[0]), "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)), "mean_ms": float(ms.mean()),
            "projected_hours_220k_calls": float(ms.mean() * 220_000 / 3.6e6)}


def main():
    ap = argparse.ArgumentParser(description="Local Brier-ALC evaluator for PAIEC")
    ap.add_argument("--model", required=True, help="directory with model.py [labeling.py]")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "measurement-db"))
    ap.add_argument("--split", default="lobo", choices=["lobo", "pairs", "subjects"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--min-items", type=int, default=80)
    ap.add_argument("--max-items-per-pair", type=int, default=0, help="0 = all items")
    ap.add_argument("--max-pairs-per-bench", type=int, default=0, help="0 = all pairs")
    ap.add_argument("--labeled-scope", default="pair", choices=["pair", "benchmark"],
                    help="pair: labeled has only the pair's labels; benchmark: plus the same number "
                         "of labels from every other evaluated pair of the benchmark")
    ap.add_argument("--strict", action="store_true",
                    help="check predict_batch == predict and no state between calls on a sample of pairs")
    ap.add_argument("--strict-frac", type=float, default=0.05, help="share of pairs checked (min 1 per fold)")
    ap.add_argument("--stress", action="store_true",
                    help="extra held-out folds on matharena and mmdocrag, binarized at response >= 0.5")
    ap.add_argument("--latency", type=int, default=0, metavar="N",
                    help="time N item-by-item predict calls with benchmark-scope labeled")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    obs_all, subjects, items, meta = bench_data.load(args.data, seed=args.seed,
                                                     eligible_only=not args.stress)
    eligible = {d for d, m in meta.items() if m["eligible"]}
    obs = obs_all[obs_all["bench"].isin(eligible)]
    pairs = eligible_pairs(obs, args.min_items)
    name = os.path.basename(os.path.normpath(args.model))
    strict_state = [[], 0]
    grouped = {k: g for k, g in obs_all.groupby(["subject_id", "bench"])}
    factory = lambda: Submission(args.model)

    per_pair = []
    for fold, train_obs, test_pairs in make_folds(obs, subjects, pairs, args.split, rng, args.holdout_frac):
        per_pair += run_fold(args, factory, fold, train_obs, test_pairs, grouped, subjects, items, strict_state)
    stress_pairs = []
    if args.stress:
        # Stress folds: train on all binary benchmarks, score a non-binary one binarized.
        for b in sorted(set(meta) - eligible):
            test = pd.DataFrame([p for p in eligible_pairs(obs_all[obs_all["bench"] == b], args.min_items)],
                                columns=["subject_id", "bench"])
            stress_pairs += run_fold(args, factory, f"stress:{b}", obs, test, grouped, subjects, items,
                                     strict_state)
    strict_failures, strict_pairs = strict_state

    df = pd.DataFrame(per_pair)
    result = {"model": name, "split": args.split, "seed": args.seed, "labeled_scope": args.labeled_scope,
              "n_pairs": int(len(df)),
              "brier_alc": float(df["alc"].mean())}
    for b in BUDGETS:
        result[f"B{b}"] = float(df[f"B{b}"].mean())
    per_bench = {bench: summarize(g) for bench, g in df.groupby("bench")}
    # Mean over benchmarks (each benchmark weighted equally), a second view on the same runs.
    result["brier_alc_bench_macro"] = float(np.mean([v["brier_alc"] for v in per_bench.values()]))
    result["per_benchmark"] = per_bench
    result["per_fold"] = {fold: summarize(g) for fold, g in df.groupby("fold")}
    worst = max(result["per_fold"], key=lambda f: result["per_fold"][f]["brier_alc"])
    result["worst_fold"], result["worst_fold_brier_alc"] = worst, result["per_fold"][worst]["brier_alc"]
    if args.stress:
        sdf = pd.DataFrame(stress_pairs)
        folds = {**result["per_fold"], **{f: summarize(g) for f, g in sdf.groupby("fold")}}
        sworst = max(folds, key=lambda f: folds[f]["brier_alc"])
        result["stress"] = {"binarization": "response >= 0.5 on one sampled trial",
                            "folds": {f: summarize(g) for f, g in sdf.groupby("fold")},
                            "worst_fold_incl_stress": sworst,
                            "worst_brier_alc_incl_stress": folds[sworst]["brier_alc"]}
    result["n_eval_predictions"] = int(df["n_eval"].sum() * len(BUDGETS))
    result["short_trajectories"] = int((df["labels_used"] < MAX_LABELS).sum())
    result["mean_labels_visible_at_31"] = float(df["labels_visible_at_31"].mean())
    result["strict_ok"] = (not strict_failures) if args.strict else None
    if args.strict:
        result["strict_pairs_checked"] = strict_pairs
        result["strict_failures"] = strict_failures[:20]
    if args.latency:
        result["latency"] = measure_latency(args, factory, obs, pairs, grouped, subjects, items, args.latency)
    result["seconds"] = round(time.time() - t0, 1)
    result["config"] = {k: v for k, v in vars(args).items() if k not in ("out",)}

    out = args.out or os.path.join(HERE, "results", f"{name}_{args.split}_seed{args.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    pd.concat([df, pd.DataFrame(stress_pairs)], ignore_index=True).to_csv(out.replace(".json", "_pairs.csv"),
                                                                         index=False)
    print(json.dumps(result))
    if strict_failures:
        print(f"STRICT FAILED ({len(strict_failures)} checks): " + strict_failures[0], file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
