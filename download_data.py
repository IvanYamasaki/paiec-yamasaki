"""Download the PAIEC training tables (measurement-db) and the public item embeddings.

measurement-db is gated on Hugging Face. Request access with your account, create a
read token and save it in paiec/.hf_token (one line) or export HF_TOKEN.

    .venv/Scripts/python download_data.py              # tables only (~70 MB)
    .venv/Scripts/python download_data.py --traces     # plus traces (~1 GB)
    .venv/Scripts/python download_data.py --embed-only # public embeddings only

Files land in data/measurement-db/<benchmark>/ and data/measurement-db-embed/.
The resolved revision is written to data/revision.json for provenance.
"""

import argparse
import json
import os

from huggingface_hub import HfApi, snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DB_REPO = "aims-foundations/measurement-db"
EMBED_REPO = "aims-foundations/measurement-db-embed"


def _token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    path = os.path.join(HERE, ".hf_token")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read().strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", action="store_true")
    ap.add_argument("--embed-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    revisions = {}

    info = HfApi().repo_info(EMBED_REPO, repo_type="dataset")
    snapshot_download(EMBED_REPO, repo_type="dataset", revision=info.sha,
                      local_dir=os.path.join(DATA, "measurement-db-embed"))
    revisions[EMBED_REPO] = info.sha

    if not args.embed_only:
        token = _token()
        if not token:
            raise SystemExit("measurement-db is gated: put a read token in paiec/.hf_token or HF_TOKEN")
        info = HfApi(token=token).repo_info(DB_REPO, repo_type="dataset")
        tables = ["benchmarks", "subjects", "items", "response", "responses"]
        if args.traces:
            tables.append("traces")
        # Top-level benchmark dirs only; skip raw/ sources, assets and reproduction runs.
        patterns = [f"*/{t}.parquet" for t in tables] + ["*/metadata.yaml", "*/characterization.yaml", "README.md"]
        snapshot_download(DB_REPO, repo_type="dataset", revision=info.sha, token=token,
                          local_dir=os.path.join(DATA, "measurement-db"), allow_patterns=patterns,
                          ignore_patterns=["*/raw/*", "*/reproduction_checks/*"])
        revisions[DB_REPO] = info.sha

    with open(os.path.join(DATA, "revision.json"), "w") as f:
        json.dump(revisions, f, indent=2)
    print(json.dumps(revisions, indent=2))


if __name__ == "__main__":
    main()
