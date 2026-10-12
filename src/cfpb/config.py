"""All paths and shared settings live here, so no script hard-codes a path."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../CFPB

DATA = ROOT / "data"
RAW_ARCHIVE = DATA / "raw" / "narratives_archive"   # the 13 FOIA archive CSVs (with narratives)
RAW_CURRENT = DATA / "raw" / "ccdb_current"         # your 5 GB complaints.csv (no narratives)
INTERIM = DATA / "interim"                          # combined + cleaned, all rows
PROCESSED = DATA / "processed"                      # task-ready train/val/test splits
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# The combined dataset: one Parquet file per archive CSV, read together as one table.
# (One giant file would need all ~3M narratives in RAM at once.)
COMBINED_DIR = INTERIM / "narratives_parts"

SEED = 42

# Time-based split boundaries, shared by every module so results are comparable
VAL_START = "2025-01-01"
TEST_START = "2025-07-01"

for p in (COMBINED_DIR, PROCESSED, MODELS, FIGURES):
    p.mkdir(parents=True, exist_ok=True)
