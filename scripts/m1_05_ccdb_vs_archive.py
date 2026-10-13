"""
Data check: is the Jan-Feb 2026 collapse in the narrative archive real, or a publication artifact?

    uv run python scripts/m1_05_ccdb_vs_archive.py

Compares, month by month, ALL complaints in the current database export
(data/raw/ccdb_current/complaints.csv, no narratives) with the complaints that
have a published narrative in the archive. If the database shows normal volume and a
normal credit-reporting share in early 2026, the archive's drop is about which
complaints got narratives published - not about what consumers complained about.

Needs reports/m1_monthly_mix.csv (from scripts/m1_03_label_shift.py).
Writes reports/m1_ccdb_vs_archive.csv
"""
import pandas as pd

from cfpb import config
from cfpb.text import PRODUCT_MAP

CR = "Credit reporting"


def main():
    files = sorted(list(config.RAW_CURRENT.glob("*.csv")) + list(config.RAW_CURRENT.glob("*.zip")))
    if not files:
        raise SystemExit(f"Put the current database export in {config.RAW_CURRENT}")

    counts = []
    for chunk in pd.read_csv(files[0], usecols=["Date received", "Product"], dtype=str,
                             chunksize=500_000):
        month = pd.to_datetime(chunk["Date received"], errors="coerce").dt.to_period("M")
        product = chunk["Product"].map(PRODUCT_MAP)
        counts.append(pd.DataFrame({"month": month, "is_cr": product == CR}))
    db = pd.concat(counts).dropna(subset=["month"])
    db = db[db["month"] >= pd.Period("2024-07", "M")]
    g = db.groupby("month")["is_cr"].agg(db_volume="size", db_cr_share="mean")
    g["db_cr_share"] *= 100

    arch = pd.read_csv(config.REPORTS / "m1_monthly_mix.csv")
    arch["month"] = pd.PeriodIndex(arch["date"], freq="M")
    arch = arch.set_index("month")[["volume", CR]].rename(
        columns={"volume": "archive_volume", CR: "archive_cr_share"})

    out = g.join(arch, how="left")
    out["narrative_rate_%"] = out["archive_volume"] / out["db_volume"] * 100
    out = out[["db_volume", "archive_volume", "narrative_rate_%", "db_cr_share", "archive_cr_share"]]
    print("Per month: all complaints (database) vs complaints with published narratives (archive)")
    print(out.round(1).to_string())
    out.round(2).to_csv(config.REPORTS / "m1_ccdb_vs_archive.csv")


if __name__ == "__main__":
    main()
