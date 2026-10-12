# Complaint Intelligence Platform (CFPB)

Triage, risk prediction and regulation-grounded response drafting for consumer financial complaints.

## Data
- `data/raw/narratives_archive/` - CFPB FOIA Reading Room archive exports (May 2021 - Feb 2026), **with** complaint narratives. The CFPB stopped publishing narratives in its live database on 14 Aug 2026; the archive preserves them.
- `data/raw/ccdb_current/complaints.csv` - current Consumer Complaint Database export (metadata only, no narratives). Used for tabular risk modelling and drift monitoring.

## Setup (uv)
```bash
uv sync          # creates .venv, installs dependencies + this project (editable)
```

## Run
```bash
uv run python scripts/00_combine_archive.py     # once: 13 CSVs -> cleaned Parquet parts
uv run python scripts/m1_01_train_baseline.py   # Module 1: TF-IDF + LogReg baseline
```

## Results - Module 1 (product classification, time-based test split)
| Model | Macro-F1 | Accuracy | ms / 1k | Cost / 1k |
|---|---|---|---|---|
| TF-IDF + LogReg | | | | $0 |
| DistilBERT (fine-tuned) | | | | |
| LLM (zero/few-shot) | | | | |
