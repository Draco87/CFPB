"""
Module 7, Step 1: save everything the API needs to score ONE new complaint, and prove it matches training.
(laptop, ~5 minutes)

    uv run python scripts/m7_01_export_artifacts.py

Why: in Module 3 the risk features were built in bulk (e.g. "this company's payout rate in 2021-22").
A live service gets one complaint at a time, so it needs those history numbers saved in a file.

Saves models/m7_risk_artifacts.json:
  - payout rates per company / issue / sub-product / product (from the same history window as Module 3)
  - complaint volume per company, the top-100 companies and top-25 states
  - the category lists the XGBoost model was trained with
  - recalibration (Platt) numbers fitted on the validation period
  - the score that marks the riskiest 10% of the test period ("high risk" band)

Then the SKEW TEST: rebuilds the features of 2,000 test complaints from their raw text and details with the
API's code (src/cfpb/risk.py) and compares them with the features the model was trained/tested on.
Any difference = "training/serving skew", a classic silent bug in deployed models.
"""
import importlib.util
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from cfpb import config
from cfpb.data import part_files
from cfpb.risk import ARTIFACTS, CAT, FEATS, as_categories, build_features

_spec = importlib.util.spec_from_file_location("m3_01", config.ROOT / "scripts" / "m3_01_build_features.py")
m3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m3)          # reuse Module 3's windows, smoothing and keyword rules - one source of truth


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    # ---- history lookups, exactly as in m3_01
    meta = pd.concat((pd.read_parquet(f, columns=m3.META) for f in part_files()), ignore_index=True)
    meta = meta.dropna(subset=["product", "company_response"]).drop_duplicates("complaint_id")
    meta = meta[meta["company_response"] != "In progress"]
    meta["y"] = (meta["company_response"] == m3.TARGET_VALUE).astype("int8")
    a, b = m3.WINDOWS["history"]
    hist = meta[(meta["date"] >= a) & (meta["date"] < b)].assign(
        sub_product=lambda x: x["sub_product"].fillna("None"), issue=lambda x: x["issue"].fillna("None"))
    g = float(hist["y"].mean())
    rates, volume = {}, None
    for col in ["company", "issue", "sub_product", "product"]:
        r, cnt = m3.smoothed_rate(hist, col, g)
        rates[col] = {str(k): float(v) for k, v in r.items()}
        if col == "company":
            volume = {str(k): int(v) for k, v in cnt.items()}
    tr = pd.read_parquet(config.PROCESSED / "m3_train.parquet")
    va = pd.read_parquet(config.PROCESSED / "m3_val.parquet")
    te = pd.read_parquet(config.PROCESSED / "m3_test.parquet")
    art = {"global_rate": g, "rates": rates, "company_volume": volume,
           "top_companies": [str(c) for c in hist["company"].value_counts().head(100).index],
           "top_states": [str(s) for s in hist["state"].value_counts().head(25).index],
           "keywords": m3.KEYWORDS,
           "cats": {c: sorted(tr[c].astype(str).unique()) for c in CAT}}

    # ---- recalibration + high-risk threshold, from the saved model
    model = xgb.XGBClassifier()
    model.load_model(str(config.MODELS / "m3_xgb_full.json"))
    p_va = model.predict_proba(as_categories(va[FEATS], art["cats"]))[:, 1]
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_va).reshape(-1, 1), va["y"])
    art["platt"] = {"a": float(platt.coef_[0][0]), "b": float(platt.intercept_[0])}
    p_te = model.predict_proba(as_categories(te[FEATS], art["cats"]))[:, 1]
    art["top10_threshold"] = float(np.quantile(p_te, 0.90))
    ARTIFACTS.write_text(json.dumps(art))
    print(f"Saved {ARTIFACTS} ({ARTIFACTS.stat().st_size / 1e6:.1f} MB): {len(rates['company']):,} companies, "
          f"history payout rate {g:.2%}, high-risk threshold {art['top10_threshold']:.4f}")

    # ---- skew test: rebuild 2,000 test complaints from raw data with the API code
    sample = te.sample(min(2000, len(te)), random_state=config.SEED)
    ids = set(sample["complaint_id"])
    cols = ["complaint_id", "date", "product", "sub_product", "issue", "company", "state", "tags", "submitted_via", "text"]
    raw = pd.concat((pd.read_parquet(f, columns=cols).loc[lambda x: x["complaint_id"].isin(ids)] for f in part_files()))
    raw = raw.drop_duplicates("complaint_id").set_index("complaint_id").loc[sample["complaint_id"]].reset_index()
    raw["date"] = raw["date"].astype(str).str[:10]
    rows = raw.astype(object).where(raw.notna(), None).to_dict("records")
    text_model = joblib.load(config.MODELS / "m3_text_model.joblib")
    api = build_features(rows, art, text_model).reset_index(drop=True)
    ref = sample[FEATS].reset_index(drop=True)
    print("\nSkew test - API features vs training features on 2,000 test complaints:")
    worst = 0.0
    for c in FEATS:
        if c in CAT:
            same = (api[c].astype(str) == ref[c].astype(str)).mean()
            bad = 1 - same
            print(f"  {c:24s} identical in {same:.1%}")
        else:
            diff = np.abs(api[c].astype(float).to_numpy() - ref[c].astype(float).to_numpy())
            bad = float((diff > 1e-6).mean())
            print(f"  {c:24s} max difference {diff.max():.2e}")
        worst = max(worst, bad)
    pa = model.predict_proba(as_categories(api, art["cats"]))[:, 1]
    pr = model.predict_proba(as_categories(ref, art["cats"]))[:, 1]
    print(f"\nPrediction difference: max {np.abs(pa - pr).max():.2e}")
    print("PASS - the API builds the same features as training." if worst == 0 and np.abs(pa - pr).max() < 1e-6
          else "CHECK - some features differ; paste this output.")


if __name__ == "__main__":
    main()
