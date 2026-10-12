"""
Module 1, Step 1: build the train/val/test splits that every Module 1 model uses.

    uv run python scripts/m1_00_make_splits.py                       # capped train (default)
    uv run python scripts/m1_00_make_splits.py --train-mode natural  # experiment: natural mix

- val  = Jan-Jun 2025, test = Jul 2025 onward, both random samples with the REAL
  product mix (that is what the model will face in production).
- train = before 2025, at most --train-cap rows per product, so rare products
  (payday, student loan...) get thousands of examples instead of a few hundred.
- val/test are FROZEN after the first run: re-running only rebuilds train, so all
  results stay comparable. Use --force to rebuild val/test too (old results then
  become incomparable - delete reports/m1_results.csv).

Writes data/processed/m1_{train,val,test}.parquet and m1_splits_meta.json
"""
import argparse
import json

import pandas as pd

from cfpb import config
from cfpb.data import part_files

META_COLS = ["complaint_id", "date", "product", "n_chars"]
KEEP_COLS = ["complaint_id", "date", "product", "sub_product", "issue", "company", "text", "n_chars"]
OUT = {s: config.PROCESSED / f"m1_{s}.parquet" for s in ("train", "val", "test")}
META_FILE = config.PROCESSED / "m1_splits_meta.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-mode", choices=["capped", "natural"], default="capped")
    ap.add_argument("--train-cap", type=int, default=20_000, help="max train rows per product (capped)")
    ap.add_argument("--train-n", type=int, default=170_000, help="train rows (natural)")
    ap.add_argument("--val-n", type=int, default=30_000)
    ap.add_argument("--test-n", type=int, default=50_000)
    ap.add_argument("--force", action="store_true", help="also rebuild val/test")
    args = ap.parse_args()
    seed = config.SEED

    # Pass 1: metadata only (no text) -> cheap on RAM
    meta = pd.concat((pd.read_parquet(f, columns=META_COLS) for f in part_files()), ignore_index=True)
    meta = meta.dropna(subset=["product"]).drop_duplicates("complaint_id")
    meta = meta[meta["n_chars"] >= 50]

    tr = meta[meta["date"] < config.VAL_START]
    va = meta[(meta["date"] >= config.VAL_START) & (meta["date"] < config.TEST_START)]
    te = meta[meta["date"] >= config.TEST_START]

    if args.train_mode == "capped":
        tr_ids = tr.sample(frac=1, random_state=seed).groupby("product").head(args.train_cap)
    else:
        tr_ids = tr.sample(min(args.train_n, len(tr)), random_state=seed)
    ids = {"train": set(tr_ids["complaint_id"])}

    if OUT["val"].exists() and OUT["test"].exists() and not args.force:
        print("Keeping existing (frozen) val/test sets.")
        for s in ("val", "test"):
            ids[s] = set(pd.read_parquet(OUT[s], columns=["complaint_id"])["complaint_id"])
        rebuild = ["train"]
    else:
        ids["val"] = set(va.sample(min(args.val_n, len(va)), random_state=seed)["complaint_id"])
        ids["test"] = set(te.sample(min(args.test_n, len(te)), random_state=seed)["complaint_id"])
        rebuild = ["train", "val", "test"]

    # Pass 2: read text only for the chosen rows, one part at a time
    wanted = {cid: s for s in rebuild for cid in ids[s]}
    parts = []
    for f in part_files():
        d = pd.read_parquet(f, columns=KEEP_COLS)
        d = d[d["complaint_id"].isin(wanted.keys())].drop_duplicates("complaint_id")
        parts.append(d.assign(split=d["complaint_id"].map(wanted)))
    df = pd.concat(parts, ignore_index=True)

    for s in rebuild:
        df[df["split"] == s].drop(columns="split").reset_index(drop=True).to_parquet(OUT[s], index=False)

    META_FILE.write_text(json.dumps({"train_mode": args.train_mode, "train_cap": args.train_cap,
                                     "train_n": args.train_n, "seed": seed}, indent=2))

    counts = pd.DataFrame({s: pd.read_parquet(OUT[s], columns=["product"])["product"].value_counts()
                           for s in OUT}).fillna(0).astype(int)
    shares = (counts / counts.sum() * 100).round(1).add_suffix(" %")
    print(pd.concat([counts, shares], axis=1).sort_values("test", ascending=False))
    print("\nrows:", counts.sum().to_dict(), "| train mode:", args.train_mode)


if __name__ == "__main__":
    main()
