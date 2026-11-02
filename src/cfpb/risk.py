"""
Payout-risk model for ONE new complaint at a time (Module 3 model, served in Module 7).

Training (m3_01) built features in bulk from history tables. To score a single new complaint the API needs the
same history numbers - each company's past payout rate, the top-100 company list, etc. - so
scripts/m7_01_export_artifacts.py saves them to models/m7_risk_artifacts.json. That script also checks that
features built here match the training features exactly (no training/serving skew).
"""
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd

from cfpb import config
from cfpb.text import clean_text

ARTIFACTS = config.MODELS / "m7_risk_artifacts.json"
CAT = ["product", "sub_product", "issue", "company_group", "state_group", "submitted_via"]
NUM = ["older_american", "servicemember", "month_of_year", "log_length", "hist_rate_company",
       "hist_rate_issue", "hist_rate_sub_product", "hist_rate_product", "hist_company_volume"]
TEXT = ["text_score", "kw_refund", "kw_unauthorized", "kw_fraud", "kw_fee", "kw_overdraft",
        "kw_double_charge", "kw_dispute", "kw_identity_theft"]
FEATS = CAT + NUM + TEXT
LABELS = {"product": "Product", "sub_product": "Sub-product", "issue": "Issue", "company_group": "Company",
          "state_group": "State", "submitted_via": "Submitted via", "older_american": "Older American tag",
          "servicemember": "Servicemember tag", "month_of_year": "Month of year", "log_length": "Complaint length",
          "hist_rate_company": "Company's past payout rate", "hist_rate_issue": "Issue's past payout rate",
          "hist_rate_sub_product": "Sub-product's past payout rate", "hist_rate_product": "Product's past payout rate",
          "hist_company_volume": "Company's past complaint volume",
          "text_score": "What the complaint says (text model)", "kw_refund": "Mentions a refund",
          "kw_unauthorized": "Mentions 'unauthorized'", "kw_fraud": "Mentions fraud/scam", "kw_fee": "Mentions fees",
          "kw_overdraft": "Mentions overdraft", "kw_double_charge": "Mentions a double charge",
          "kw_dispute": "Mentions a dispute", "kw_identity_theft": "Mentions identity theft"}


def build_features(rows, art, text_model):
    """rows: list of dicts with text, product (9-group name), and optional sub_product, issue, company, state,
    submitted_via, tags, date (YYYY-MM-DD). Returns a DataFrame with exactly the Module 3 feature columns."""
    d = pd.DataFrame(rows)
    for c in ["sub_product", "issue", "company", "state", "submitted_via", "tags", "date"]:
        if c not in d:
            d[c] = None
    text = d["text"].fillna("").map(clean_text)
    low = text.str.lower()
    dates = pd.to_datetime(d["date"].fillna(date.today().isoformat()), errors="coerce").fillna(pd.Timestamp.today())
    tags = d["tags"].fillna("")
    g = art["global_rate"]
    rate = art["rates"]
    out = pd.DataFrame({
        "product": d["product"],
        "sub_product": d["sub_product"].fillna("None"),
        "issue": d["issue"].fillna("None"),
        "company_group": np.where(d["company"].isin(art["top_companies"]), d["company"], "Other"),
        "state_group": np.where(d["state"].isin(art["top_states"]), d["state"], "Other"),
        "submitted_via": d["submitted_via"].fillna("None"),
        "older_american": tags.str.contains("Older American").astype("int8"),
        "servicemember": tags.str.contains("Servicemember").astype("int8"),
        "month_of_year": dates.dt.month,
        "log_length": np.log1p(text.str.len()),
        "hist_rate_company": d["company"].map(rate["company"]).fillna(g),
        "hist_rate_issue": d["issue"].fillna("None").map(rate["issue"]).fillna(g),
        "hist_rate_sub_product": d["sub_product"].fillna("None").map(rate["sub_product"]).fillna(g),
        "hist_rate_product": d["product"].map(rate["product"]).fillna(g),
        "hist_company_volume": np.log1p(d["company"].map(art["company_volume"]).fillna(0)),
        "text_score": text_model.predict_proba(text)[:, 1],
    })
    for k, pat in art["keywords"].items():
        out[k] = low.str.contains(pat, regex=True).astype("int8")
    return out[FEATS]


def as_categories(X, cats):
    X = X.copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c].where(X[c].isin(cats[c])), categories=cats[c])   # unseen -> missing
    return X


class RiskModel:
    def __init__(self):
        import xgboost as xgb
        self.xgb = xgb
        self.art = json.loads(ARTIFACTS.read_text())
        self.text_model = joblib.load(config.MODELS / "m3_text_model.joblib")
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(config.MODELS / "m3_xgb_full.json"))

    def recalibrate(self, p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        z = self.art["platt"]["a"] * np.log(p / (1 - p)) + self.art["platt"]["b"]
        return 1 / (1 + np.exp(-z))

    def predict(self, rows, top_k=3):
        X = as_categories(build_features(rows, self.art, self.text_model), self.art["cats"])
        raw = self.model.predict_proba(X)[:, 1]
        contrib = self.model.get_booster().predict(self.xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
        out = []
        for i in range(len(X)):
            c = contrib[i, :-1]
            order = np.argsort(-np.abs(c))[:top_k]
            out.append({
                "payout_probability": round(float(self.recalibrate(raw[i])), 4),
                "risk_band": "high" if raw[i] >= self.art["top10_threshold"] else "normal",
                "top_drivers": [{"feature": LABELS.get(FEATS[j], FEATS[j]), "value": _show(FEATS[j], X.iloc[i, j]),
                                 "pushes": "up" if c[j] > 0 else "down"} for j in order],
            })
        return out


def _show(col, v):
    if pd.isna(v):
        return "unknown"
    if col.startswith("hist_rate") or col == "text_score":
        return f"{float(v) * 100:.1f}%"
    if col.startswith("kw_") or col in ("older_american", "servicemember"):
        return "yes" if int(v) == 1 else "no"
    if col == "log_length":
        return f"{int(np.expm1(float(v))):,} characters"
    if col == "hist_company_volume":
        return f"{int(np.expm1(float(v))):,} past complaints"
    return str(v)
