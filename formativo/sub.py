"""Formative-phase helper around paiec_baseline/submit_api.py.

    python formativo/sub.py submit ARM        # uploads zips/ARM.zip, appends to subs.csv
    python formativo/sub.py wait ID [ID ...]  # polls until every ID is Finished/Failed
    python formativo/sub.py details ID [--raw]  # fetches logs; --raw prints the scoring output

Only aggregate numbers from the scoring output are kept (metrics/ID.json). Raw logs are
never written to disk.
"""
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
PAIEC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PAIEC, "paiec_baseline"))
import submit_api as sa  # noqa: E402

SUBS = os.path.join(HERE, "subs.csv")
DONE = {"Finished", "Failed", "Cancelled"}


def session():
    s = requests.Session()
    sa.authenticate(s)
    return s


def submit(arm):
    z = os.path.join(HERE, "zips", f"{arm}.zip")
    p = subprocess.run([sys.executable, os.path.join(PAIEC, "paiec_baseline", "submit_api.py"), z],
                       capture_output=True, text=True)
    print(p.stdout[-800:], p.stderr[-1500:])
    m = re.search(r"Submission created: (\d+)", p.stdout)
    if not m:
        sys.exit("submission failed")
    new = not os.path.exists(SUBS)
    with open(SUBS, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["arm", "id", "submitted_utc"])
        w.writerow([arm, m.group(1), dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")])
    print("ID", m.group(1))


def wait(ids, every=30):
    s = session()
    pending = set(ids)
    while pending:
        for i in sorted(pending):
            st = sa.api(s, "GET", f"submissions/{i}/")
            if st["status"] in DONE:
                print(i, st["status"], st.get("status_details") or "", flush=True)
                pending.discard(i)
        if pending:
            time.sleep(every)


def fetch_logs(s, i):
    d = sa.api(s, "GET", f"submissions/{i}/get_details/")
    out = {}
    for log in d.get("logs", []):
        url = log.get("data_file")
        if url:
            r = requests.get(url, timeout=60)
            out[log.get("name", "?")] = r.text if r.ok else f"<HTTP {r.status_code}>"
    return d, out


NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def parse(text):
    """Aggregate metrics only: unweighted mean over the listed pairs, per budget.

    The scoring output lists one row per anonymous (subject, benchmark) pair; the
    leaderboard score is the plain mean of the pair rows in the ALC summary. Rows and
    IDs are discarded here, only the means and the row count are kept."""
    m, section = {}, None
    rows = {}
    for line in text.splitlines():
        if line.startswith("ALC summary"):
            section = "alc"
        b = re.match(r"Label budget: (\d+)", line)
        if b:
            section = f"B{b.group(1)}"
        r = re.match(r"\S+\s+\S+\s+(\d+)\s+" + NUM + r"\s+" + NUM + r"\s*$", line)
        if r and section:
            rows.setdefault(section, []).append((int(r.group(1)), float(r.group(2)), float(r.group(3))))
    for sec, rr in rows.items():
        key = "brier_alc" if sec == "alc" else f"brier_{sec}"
        m[key] = sum(x[1] for x in rr) / len(rr)
        m[key.replace("brier", "ece")] = sum(x[2] for x in rr) / len(rr)
    if "alc" in rows:
        m["n_pairs"] = len(rows["alc"])
        m["n_items_total"] = sum(x[0] for x in rows["alc"])
    return m


def details(i, raw=False):
    s = session()
    st = sa.api(s, "GET", f"submissions/{i}/")
    d, logs = fetch_logs(s, i)
    print("status", st["status"], "| scores", st.get("scores"))
    print("detail keys", sorted(d.keys()), "| logs", list(logs))
    if raw:
        for name, text in logs.items():
            print(f"===== {name} ({len(text)} chars)\n{text[-6000:]}")
    text = "\n".join(t for n, t in logs.items() if "scor" in n.lower()) or "\n".join(logs.values())
    m = parse(text)
    for sc in st.get("scores", []) or []:
        if isinstance(sc, dict) and "score" in sc:
            m.setdefault("score_" + str(sc.get("column_key") or sc.get("index")), float(sc["score"]))
    os.makedirs(os.path.join(HERE, "metrics"), exist_ok=True)
    json.dump({"id": int(i), "status": st["status"], **m}, open(os.path.join(HERE, "metrics", f"{i}.json"), "w"), indent=1)
    print(json.dumps(m))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "submit":
        submit(args[0])
    elif cmd == "wait":
        wait([int(a) for a in args])
    elif cmd == "details":
        details(int(args[0]), raw="--raw" in args)
