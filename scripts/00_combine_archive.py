"""
Step 0 (run once): combine the 13 FOIA archive CSVs into one cleaned dataset.

    python scripts/00_combine_archive.py

Reads   data/raw/narratives_archive/*.csv  (or .zip - pandas reads zipped CSVs directly)
Writes  data/interim/narratives_parts/<name>.parquet  (one part per CSV)

Every later script loads these parts (via cfpb.data) instead of the raw CSVs:
Parquet is ~5x smaller and loads ~20x faster, and you can load only the columns you need.
"""
import pandas as pd

from cfpb import config
from cfpb.text import PRODUCT_MAP, clean_text

# Raw CSV header -> short snake_case name used everywhere in the project
RENAME = {
    "Date received": "date",
    "Product": "product_raw",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative_raw",
    "Company public response": "company_public_response",
    "Company": "company",
    "State": "state",
    "ZIP code": "zip",
    "Tags": "tags",
    "Submitted via": "submitted_via",
    "Date sent to company": "date_sent",
    "Company response to consumer": "company_response",
    "Timely response?": "timely",
    "Complaint ID": "complaint_id",
}


def process_file(path) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(path, dtype=str, chunksize=200_000, low_memory=False):
        chunk = chunk.rename(columns=RENAME)
        chunk = chunk[[c for c in RENAME.values() if c in chunk.columns]]
        chunk = chunk.dropna(subset=["narrative_raw", "product_raw"])
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date_sent"] = pd.to_datetime(df["date_sent"], errors="coerce")
    df["complaint_id"] = pd.to_numeric(df["complaint_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["date", "complaint_id"]).drop_duplicates("complaint_id")

    df["product"] = df["product_raw"].map(PRODUCT_MAP)       # NaN = unmapped, kept for review
    df["text"] = df["narrative_raw"].map(clean_text)
    df["n_chars"] = df["text"].str.len().astype("int32")
    return df.drop(columns=["narrative_raw"])


def main():
    csvs = {f.stem: f for f in config.RAW_ARCHIVE.glob("*.csv")}
    zips = {f.stem: f for f in config.RAW_ARCHIVE.glob("*.zip")}
    both = csvs.keys() & zips.keys()
    if both:
        print(f"Both .csv and .zip found for {sorted(both)} - using the .csv, ignoring the .zip")
    files = sorted({**zips, **csvs}.values())
    if not files:
        raise SystemExit(f"Put the archive CSVs in {config.RAW_ARCHIVE}")

    summary = []
    for f in files:
        out = config.COMBINED_DIR / f"{f.stem}.parquet"
        if out.exists():
            print(f"skip {f.name} (already done)")
            df = pd.read_parquet(out, columns=["date", "product_raw", "product"])
        else:
            print(f"processing {f.name} ...")
            df = process_file(f)
            df.to_parquet(out, index=False)
        summary.append(df)
        print(f"  {len(df):>9,} rows | {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")

    allp = pd.concat(summary, ignore_index=True)
    print(f"\nTOTAL {len(allp):,} complaints with narratives\n")
    print("Rows per year:\n", allp["date"].dt.year.value_counts().sort_index(), "\n")
    print("Normalised products:\n", allp["product"].value_counts(), "\n")
    unmapped = allp.loc[allp["product"].isna(), "product_raw"].value_counts()
    if len(unmapped):
        print("UNMAPPED product names - add large ones to PRODUCT_MAP in src/cfpb/text.py,\n"
              "then delete data/interim/narratives_parts/ and re-run:\n", unmapped)


if __name__ == "__main__":
    main()
