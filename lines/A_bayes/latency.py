"""ms per item-by-item predict (as in the test), labeled of the pair only and of 10 / 82 pairs."""
import sys, time, importlib.util, numpy as np
def load(path):
    spec = importlib.util.spec_from_file_location("m_" + path.replace("/", "_"), path + "/model.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
rng = np.random.default_rng(0)
def subj(j): return {"normalized_name": f"model-{j}", "provider": "OpenAI", "release_date": "2025-03-01", "access_date": "", "harness": "", "harness_version": "", "reasoning_effort": "", "subject_features_extra": ""}
def item(i): return {"item_content": f"task number {i} " + "x" * 3000, "item_features": f"lang={['py','c++','java'][i % 3]}", "benchmark_id": "bench_abc", "interactors": ""}
own = [[[subj(0), item(int(i))], int(rng.random() < 0.3)] for i in rng.choice(2126, 31, replace=False)]
for path in sys.argv[1:]:
    for n_pairs in (0, 9, 81):
        lab = own + [[[subj(j), item(int(i))], int(rng.random() < 0.3)] for j in range(1, n_pairs + 1) for i in rng.choice(2126, 31, replace=False)]
        for memo in (False, True):
            m = load(path)
            t0 = time.perf_counter(); n = 30
            for i in range(n):
                if not memo and hasattr(m, "_MEMO"): m._MEMO.clear()
                m.predict([subj(0), item(5000 + i)], lab)
            dt = (time.perf_counter() - t0) / n
            print(f"{path} pairs={n_pairs + 1:3d} labeled={len(lab):5d} memo={memo}: {dt * 1000:7.1f} ms/call -> 220k calls {dt * 220000 / 3600:.2f} h")
