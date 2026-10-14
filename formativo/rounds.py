"""Interleaved formative rounds: submit each arm once per round, collect results as they finish.

    python formativo/rounds.py --rounds 4 --arms J1,J2,v6,l0,l3,l6,mean --cap 40

The arm order rotates every round. A round waits up to --round-wait minutes for its own
submissions; a submission stuck in the Codabench queue stays open and is collected later
(also by the next run, from subs.csv). Stops before exceeding --cap submissions in the
current UTC day. RESULTADOS.md is rewritten after every collected result.
"""
import argparse
import csv
import datetime as dt
import importlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sub  # noqa: E402
import summary  # noqa: E402


def today_count():
    if not os.path.exists(sub.SUBS):
        return 0
    day = dt.datetime.now(dt.timezone.utc).date().isoformat()
    return sum(1 for r in csv.DictReader(open(sub.SUBS)) if r["submitted_utc"].startswith(day))


def arm_counts_today():
    day = dt.datetime.now(dt.timezone.utc).date().isoformat()
    c = {}
    for r in csv.DictReader(open(sub.SUBS)):
        if r["submitted_utc"].startswith(day):
            c[r["arm"]] = c.get(r["arm"], 0) + 1
    return c


def open_subs():
    return [(r["arm"], int(r["id"])) for r in csv.DictReader(open(sub.SUBS))
            if not os.path.exists(os.path.join(sub.HERE, "metrics", f"{r['id']}.json"))]


def collect(s, tag):
    """Fetch every finished open submission; returns the still-open IDs."""
    left = []
    for arm, i in open_subs():
        try:
            st = sub.sa.api(s, "GET", f"submissions/{i}/")
        except Exception as e:
            print(f"STATUS FAILED {arm} {i}: {e}", flush=True)
            left.append(i)
            continue
        if st["status"] not in sub.DONE and not st.get("scores"):
            left.append(i)
            continue
        try:
            sub.details(i)
            m = json.load(open(os.path.join(sub.HERE, "metrics", f"{i}.json")))
            print(f"DONE {tag} {arm} {i} {m['status']} brier_alc={m.get('brier_alc')}", flush=True)
        except Exception as e:
            print(f"DETAILS FAILED {arm} {i}: {e}", flush=True)
        try:
            importlib.reload(summary).main()
        except Exception as e:
            print(f"SUMMARY FAILED: {e}", flush=True)
    return left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--arms", default="v6,l0,l3,l6,mean")
    ap.add_argument("--cap", type=int, default=40)
    ap.add_argument("--start", type=int, default=0, help="rotation offset")
    ap.add_argument("--round-wait", type=float, default=40, help="minutes to wait for a round")
    ap.add_argument("--final-wait", type=float, default=120, help="minutes to wait for stragglers at the end")
    ap.add_argument("--extra", default="", help="arms submitted twice in the first --extra-rounds rounds")
    ap.add_argument("--extra-rounds", type=int, default=0)
    ap.add_argument("--plan", default="", help="JSON re-read every round: {order: [...], quota: {arm: max per UTC day}}")
    ap.add_argument("--start-at", default="", help="UTC time (ISO) to start submitting; collects until then")
    a = ap.parse_args()
    arms = a.arms.split(",")
    s = sub.session()
    collect(s, "earlier")
    if a.start_at:
        t = dt.datetime.fromisoformat(a.start_at).replace(tzinfo=dt.timezone.utc)
        print(f"waiting until {t.isoformat()}", flush=True)
        while dt.datetime.now(dt.timezone.utc) < t:
            time.sleep(120)
            collect(s, "earlier")
    day0 = dt.datetime.now(dt.timezone.utc).date()
    r = 0
    while r < a.rounds:
        if a.plan:
            plan = json.load(open(a.plan))
            quota = plan.get("quota", {})
            counts = arm_counts_today()
            order, used = [], {}
            for arm in plan["order"]:
                if not os.path.exists(os.path.join(sub.HERE, "zips", f"{arm}.zip")):
                    continue
                if counts.get(arm, 0) + used.get(arm, 0) < quota.get(arm, 0):
                    order.append(arm)
                    used[arm] = used.get(arm, 0) + 1
            if not order:
                # nothing left in the plan: keep collecting and re-reading it until the UTC day ends
                if dt.datetime.now(dt.timezone.utc).date() != day0 or today_count() >= a.cap:
                    break
                collect(s, "idle")
                time.sleep(120)
                continue
        else:
            k = (a.start + r) % len(arms)
            order = arms[k:] + arms[:k]
            if r < a.extra_rounds and a.extra:
                order += a.extra.split(",")
        mine = []
        for arm in order:
            if today_count() >= a.cap:
                print(f"cap {a.cap} reached for today", flush=True)
                break
            try:
                sub.submit(arm)
            except SystemExit as e:
                print(f"SUBMIT FAILED {arm}: {e}", flush=True)
                continue
            mine.append(int(list(csv.DictReader(open(sub.SUBS)))[-1]["id"]))
        r += 1
        if not mine:
            break
        t0 = time.time()
        while True:
            left = collect(s, f"round {r - 1}")
            if not set(mine) & set(left) or time.time() - t0 > a.round_wait * 60:
                break
            time.sleep(30)
        if set(mine) & set(left):
            print(f"round {r - 1}: still open after {a.round_wait} min: {sorted(set(mine) & set(left))}", flush=True)
        if today_count() >= a.cap:
            print(f"cap {a.cap} reached for today", flush=True)
            break
    t0 = time.time()
    while collect(s, "final") and time.time() - t0 < a.final_wait * 60:
        time.sleep(60)
    print("ROUNDS FINISHED; open:", [i for _, i in open_subs()], flush=True)


if __name__ == "__main__":
    main()
