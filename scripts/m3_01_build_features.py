"""
Module 3, Step 1: build the risk dataset.

Question: when a complaint ARRIVES, how likely is it to end with the company paying the consumer
("Closed with monetary relief")? Those are the complaints where the company was most likely at fault.

    uv run python scripts/m3_01_build_features.py

Time windows - each step only uses the past:
  history  May 2021 - Dec 2022 : learns relief rates per company / issue / sub-product, and a text model
  train    Jan 2023 - Jun 2024 : trains the risk models
  val      Jul 2024 - Dec 2024 : early stopping / tuning
  test     Jan 2025 - Feb 2026 : final scores

Only information available when the complaint arrives is used. Left out ON PURPOSE because they are
filled in later (= leakage): company_response (the answer itself), company_public_response, timely, date_sent.

Writes data/processed/m3_{train,val,test}.parquet, models/m3_text_model.joblib, m3_meta.json
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from cfpb import config
from cfpb.data import part_files

WINDOWS = {"history": ("2021-01-01", "2023-01-01"), "train": ("2023-01-01", "2024-07-01"),
           "val": ("2024-07-01", "2025-01-01"), "test": ("2025-01-01", "2027-01-01")}
META = ["complaint_id", "date", "product", "sub_product", "issue", "company", "state", "tags",
        "submitted_via", "company_response", "n_chars"]
TARGET_VALUE = "Closed with monetary relief"
SMOOTH = 100          # rates for small groups are pulled toward the overall rate
KEYWORDS = {          # simple, explainable text flags
    "kw_refund": r"\brefund",
    "kw_unauthorized": r"\bunauthori[sz]ed\b",
    "kw_fraud": r"\bfraud|\bscam",
    "kw_fee": r"\bfees?\b|\bcharges?\b",
    "kw_overdraft": r"\boverdraft",
    "kw_double_charge": r"charged twice|double charge|duplicate (?:charge|payment)",
    "kw_dispute": r"\bdisput",
    "kw_identity_theft": r"identity theft|stolen identity",
}


def smoothed_rate(hist, col, global_rate):
    g = hist.groupby(col)["y"].agg(["sum", "count"])
    return ((g["sum"] + SMOOTH * global_rate) / (g["count"] + SMOOTH)), g["count"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-text-n", type=int, default=200_000, help="complaints to train the text model")
    ap.add_argument("--train-n", type=int, default=300_000)
    ap.add_argument("--val-n", type=int, default=100_000)
    ap.add_argument("--test-n", type=int, default=200_000)
    args = ap.parse_args()
    seed = config.SEED

    # ---- pass 1: metadata only (cheap)
    meta = pd.concat((pd.read_parquet(f, columns=META) for f in part_files()), ignore_index=True)
    meta = meta.dropna(subset=["product", "company_response"]).drop_duplicates("complaint_id")
    meta = meta[meta["company_response"] != "In progress"]            # outcome not known yet
    meta["y"] = (meta["company_response"] == TARGET_VALUE).astype("int8")
    meta["window"] = None
    for w, (a, b) in WINDOWS.items():
        meta.loc[(meta["date"] >= a) & (meta["date"] < b), "window"] = w
    print("Complaints and money-relief rate per window:")
    print(meta.groupby("window")["y"].agg(complaints="size", relief_rate="mean").round(4), "\n")

    hist = meta[meta["window"] == "history"].assign(
        sub_product=lambda x: x["sub_product"].fillna("None"), issue=lambda x: x["issue"].fillna("None"))
    global_rate = hist["y"].mean()

    # ---- rates learned from history (applied to later windows only)
    lookups = {}
    for col in ["company", "issue", "sub_product", "product"]:
        rate, count = smoothed_rate(hist, col, global_rate)
        lookups[col] = (rate, count)
    top_companies = hist["company"].value_counts().head(100).index
    top_states = hist["state"].value_counts().head(25).index

    # ---- choose rows: random samples of each window (real mix kept)
    n_for = {"train": args.train_n, "val": args.val_n, "test": args.test_n}
    picked = {w: meta[meta["window"] == w].sample(min(n_for[w], (meta["window"] == w).sum()), random_state=seed)
              for w in n_for}
    hist_text = hist.sample(min(args.history_text_n, len(hist)), random_state=seed)
    wanted = set(hist_text["complaint_id"]).union(*[set(d["complaint_id"]) for d in picked.values()])

    # ---- pass 2: read text only for the chosen rows
    texts = []
    for f in part_files():
        d = pd.read_parquet(f, columns=["complaint_id", "text"])
        texts.append(d[d["complaint_id"].isin(wanted)])
    text = pd.concat(texts).drop_duplicates("complaint_id").set_index("complaint_id")["text"]

    # ---- text model, trained on HISTORY only
    print(f"Training the text model on {len(hist_text):,} history complaints ...")
    text_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200_000,
                                  sublinear_tf=True, stop_words="english")),
        ("clf", LogisticRegression(C=2.0, max_iter=2000)),
    ])
    text_model.fit(hist_text["complaint_id"].map(text).fillna(""), hist_text["y"])
    joblib.dump(text_model, config.MODELS / "m3_text_model.joblib")

    # ---- features for train / val / test
    for w, d in picked.items():
        d = d.copy()
        t = d["complaint_id"].map(text).fillna("")
        tl = t.str.lower()
        out = pd.DataFrame({
            "complaint_id": d["complaint_id"].to_numpy(), "date": d["date"].to_numpy(), "y": d["y"].to_numpy(),
            "product": d["product"].to_numpy(), "sub_product": d["sub_product"].fillna("None").to_numpy(),
            "issue": d["issue"].fillna("None").to_numpy(),
            "company_group": np.where(d["company"].isin(top_companies), d["company"], "Other"),
            "company": d["company"].to_numpy(),
            "state_group": np.where(d["state"].isin(top_states), d["state"], "Other"),
            "submitted_via": d["submitted_via"].fillna("None").to_numpy(),
            "older_american": d["tags"].fillna("").str.contains("Older American").astype("int8").to_numpy(),
            "servicemember": d["tags"].fillna("").str.contains("Servicemember").astype("int8").to_numpy(),
            "month_of_year": d["date"].dt.month.to_numpy(),
            "log_length": np.log1p(d["n_chars"]).to_numpy(),
            "text_score": text_model.predict_proba(t)[:, 1],
        })
        for col in ["company", "issue", "sub_product", "product"]:
            rate, count = lookups[col]
            key = d[col].fillna("None") if col in ("issue", "sub_product") else d[col]
            out[f"hist_rate_{col}"] = key.map(rate).fillna(global_rate).to_numpy()
        out["hist_company_volume"] = np.log1p(d["company"].map(lookups["company"][1]).fillna(0)).to_numpy()
        for k, pat in KEYWORDS.items():
            out[k] = tl.str.contains(pat, regex=True).astype("int8").to_numpy()
        out.to_parquet(config.PROCESSED / f"m3_{w}.parquet", index=False)
        print(f"  {w}: {len(out):,} complaints, relief rate {out['y'].mean():.2%}")

    (config.PROCESSED / "m3_meta.json").write_text(json.dumps(
        {"windows": WINDOWS, "global_history_rate": float(global_rate), "target": TARGET_VALUE,
         "excluded_for_leakage": ["company_response", "company_public_response", "timely", "date_sent"]},
        indent=2))
    print("\nDone. Next: uv run python scripts/m3_02_train_explain.py")


if __name__ == "__main__":
    main()
