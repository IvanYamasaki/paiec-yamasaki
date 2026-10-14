# paiec-yamasaki

Team Yamasaki's solution for the Predictive AI Evaluation Competition (PAIEC), a NeurIPS 2026 workshop competition. The technical report is in `report/draft.md`.

The submitted predictor family is a Rasch-style model per test benchmark. The minimal variant (J2) is 12 lines of standard-library Python in `lines/J_robusto/submit_J2_min/model.py`. Variants J1, J3 and J4 sit next to it, and the development models (A v6, v7) are in `lines/A_bayes/`.

## Install

Python 3.12 on Windows was used for all runs.

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
git clone https://github.com/aims-foundations/paiec_baseline   # official baseline, commit 82d330d
```

## Data

The training data is `aims-foundations/measurement-db` on Hugging Face. It is gated: request access, then save a read token in `.hf_token` (one line) or set `HF_TOKEN`.

```bash
.venv/Scripts/python download_data.py      # about 70 MB into data/
.venv/Scripts/python describe_data.py      # Table 1
```

Without access, `make_synthetic.py` writes six synthetic IRT benchmarks with the same layout; pass `--data data/synthetic` to the scripts.

## Reproduce the tables

```bash
# Table 2: pair and bench/10 scopes, seeds 0-2, strict, stress folds
bash lines/J_robusto/runeval.sh lines/J_robusto/submit_J2_min J2

# package check: imports, purity, latency
.venv/Scripts/python lines/D_calib/pack/check_pack.py --model lines/J_robusto/submit_J2_min

# Table 3: needs the per-round score files in formativo/metrics/
.venv/Scripts/python formativo/summary.py
.venv/Scripts/python report/make_table3.py
```

`python eval_local.py --help` lists the evaluation options. See `README_BANCADA.md` for the local evaluation bench (written in Portuguese).

## What is not here

The raw per-round Codabench score files (`formativo/metrics/`), the data, run outputs, and the LLM-generated item notes and scores used in development are left out. `formativo/RESULTADOS.md` has the formative-phase results. The A v6 model reads an optional notes file and runs without it. The draft cites a few internal analysis scripts (for example `advisor/check4_prior_shift.py`) that are not part of this repository.

## License

MIT, see `LICENSE`.
