"""
Module 7, Step 4: monitoring - replay 2025-26 month by month, as if the models were live.   (laptop, ~5 minutes)

    uv run python scripts/m7_03_monitor.py

A deployed model is trained once but used for months. The world changes underneath it. Each month we ask:

  1. Has the INPUT changed?      product mix, complaint length          -> PSI vs the training data
  2. Has the OUTPUT changed?     distribution of risk scores            -> PSI vs the validation period
  3. Is it still ACCURATE?       triage macro-F1; risk ranking (ROC-AUC) -> needs the true answers
  4. Are the percentages right?  predicted payout rate vs actual        -> calibration ratio

PSI rule of thumb (industry standard): < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.

Fix tested: ROLLING RECALIBRATION. Module 3's percentages run too high because payouts kept falling. Instead of
recalibrating once (on H2 2024), refit the Platt scaling every month on the most recent 3 months whose outcomes
are known. Outcomes arrive late in real life (a company has 15 days to respond, more to close), so month m uses
months m-4 .. m-2 - a 1-month gap for late labels. Ranking is unchanged; only the percentages move.
Months with a MAJOR product-mix shift (PSI > 0.25) are left out of the recalibration window by default: the first
run showed one abnormal month (Jan 2025) pulled the next three months' percentages far too low.
Compare with:  uv run python scripts/m7_03_monitor.py --keep-drift-months

Writes reports/m7_monitoring.html and reports/m7_monitoring.csv
"""
import importlib.util

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from cfpb import config
from cfpb.report import BLUE, INK, MUTED, ORANGE, build_html, dates, fig_html, pct, table_html
from cfpb.risk import FEATS, as_categories
from cfpb.text import clean_text

PSI_WARN, PSI_ALERT = 0.10, 0.25
CAL_OK = (0.8, 1.25)            # predicted / actual payout rate within this band = acceptable
MIN_POS = 30                    # months with fewer payout cases are too noisy to judge


def psi(ref, cur, bins=None):
    """Population stability index between two samples (categorical if bins is None, else numeric bin edges)."""
    if bins is None:
        cats = sorted(set(ref) | set(cur))
        r = pd.Series(ref).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        c = pd.Series(cur).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        r = np.histogram(ref, bins)[0] / len(ref)
        c = np.histogram(cur, bins)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def level(v):
    return "major" if v > PSI_ALERT else "moderate" if v > PSI_WARN else "stable"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def platt(p, y):
    return LogisticRegression(C=1e6, max_iter=1000).fit(logit(p), y)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-drift-months", dest="skip_drift_months", action="store_false",
                    help="also recalibrate on months with a major product-mix shift (PSI > 0.25)")
    args = ap.parse_args()
    # ------------------------------------------------ risk model replay (Module 3 data)
    tr, va, te = (pd.read_parquet(config.PROCESSED / f"m3_{s}.parquet") for s in ("train", "val", "test"))
    cats = {c: sorted(tr[c].astype(str).unique()) for c in ["product", "sub_product", "issue", "company_group",
                                                          "state_group", "submitted_via"]}
    model = xgb.XGBClassifier()
    model.load_model(str(config.MODELS / "m3_xgb_full.json"))
    score = lambda d: model.predict_proba(as_categories(d[FEATS], cats))[:, 1]
    s_va = score(va)
    static = platt(s_va, va["y"])                                  # Module 3: recalibrated once
    lab = pd.concat([va.assign(score=s_va), te.assign(score=score(te))], ignore_index=True)
    lab["month"] = pd.to_datetime(lab["date"]).dt.to_period("M")
    # reference for the SCORE distribution = validation, not training: the model has seen the training rows,
    # so its training scores are over-confident and would make every later month look like drift
    score_bins = np.unique(np.quantile(s_va, np.linspace(0, 1, 11)))
    score_bins[0], score_bins[-1] = -np.inf, np.inf
    len_bins = np.unique(np.quantile(tr["log_length"], np.linspace(0, 1, 11)))
    len_bins[0], len_bins[-1] = -np.inf, np.inf

    # product-mix drift of EVERY month (validation too), so recalibration can skip abnormal months
    month_psi = {mm: psi(tr["product"], g["product"]) for mm, g in lab.groupby("month")}
    rows = []
    months = sorted(lab.loc[lab["month"] >= pd.Period(pd.Timestamp(lab["date"].min()), "M") + 4, "month"].unique())
    test_months = sorted(pd.to_datetime(te["date"]).dt.to_period("M").unique())
    for m in [x for x in months if x in test_months]:
        cur = lab[lab["month"] == m]
        window = [w for w in (m - 4, m - 3, m - 2)]                         # labels known by now (1-month gap)
        if args.skip_drift_months:                                          # don't learn from abnormal months
            window = [w for w in window if month_psi.get(w, 0) <= PSI_ALERT]
        hist = lab[lab["month"].isin(window)]
        rolling = platt(hist["score"].to_numpy(), hist["y"]) if hist["y"].sum() >= MIN_POS else static
        y, s = cur["y"].to_numpy(), cur["score"].to_numpy()
        actual = y.mean()
        rows.append({
            "month": str(m), "complaints": len(cur), "payout_rate": actual, "payouts": int(y.sum()),
            "psi_product": psi(tr["product"], cur["product"]),
            "psi_length": psi(tr["log_length"], cur["log_length"], len_bins),
            "psi_score": psi(s_va, s, score_bins),
            "risk_roc_auc": roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan,
            "pred_static": static.predict_proba(logit(s))[:, 1].mean(),
            "pred_rolling": rolling.predict_proba(logit(s))[:, 1].mean(),
        })
    res = pd.DataFrame(rows)
    res["cal_static"] = res["pred_static"] / res["payout_rate"]
    res["cal_rolling"] = res["pred_rolling"] / res["payout_rate"]
    res.loc[res["payouts"] < MIN_POS, ["risk_roc_auc", "cal_static", "cal_rolling"]] = np.nan

    # ------------------------------------------------ triage replay (Module 1 data), if available
    tri = None
    tfidf_path = config.MODELS / "m1_tfidf_logreg.joblib"
    if tfidf_path.exists() and (config.PROCESSED / "m1_test.parquet").exists():
        print("Scoring triage on the Module 1 validation + test months ...")
        m1 = pd.concat([pd.read_parquet(config.PROCESSED / f"m1_{s}.parquet", columns=["date", "product", "text"])
                        for s in ("val", "test")], ignore_index=True)
        m1["pred"] = joblib.load(tfidf_path).predict(m1["text"].map(clean_text))
        m1["month"] = pd.to_datetime(m1["date"]).dt.to_period("M").astype(str)
        tri = (m1.groupby("month").apply(lambda g: pd.Series({
            "triage_n": len(g), "triage_macro_f1": f1_score(g["product"], g["pred"], average="macro")}),
            include_groups=False).reset_index())
        res = res.merge(tri, on="month", how="left")
    res.round(4).to_csv(config.REPORTS / "m7_monitoring.csv", index=False)

    # ------------------------------------------------ console summary
    show = res[["month", "complaints", "payout_rate", "psi_product", "psi_score", "risk_roc_auc",
                "cal_static", "cal_rolling"] + (["triage_macro_f1"] if tri is not None else [])]
    print("\n", show.round(3).to_string(index=False))
    ok = lambda v: CAL_OK[0] <= v <= CAL_OK[1]
    judged = res.dropna(subset=["cal_static"])
    n_static, n_roll = int(judged["cal_static"].map(ok).sum()), int(judged["cal_rolling"].map(ok).sum())
    print(f"\nMonths with acceptable calibration (predicted/actual between {CAL_OK[0]} and {CAL_OK[1]}): "
          f"static {n_static}/{len(judged)}, rolling {n_roll}/{len(judged)}")
    print(f"Average |predicted - actual| payout rate: static {np.mean(np.abs(judged['pred_static'] - judged['payout_rate'])):.3%}, "
          f"rolling {np.mean(np.abs(judged['pred_rolling'] - judged['payout_rate'])):.3%}")
    for col, name in [("psi_product", "product mix"), ("psi_score", "risk scores"), ("psi_length", "complaint length")]:
        worst = res.loc[res[col].idxmax()]
        print(f"Largest shift in {name}: PSI {worst[col]:.2f} ({level(worst[col])}) in {worst['month']}")

    # ------------------------------------------------ report
    x = pd.to_datetime(res["month"])
    fig1, ax = plt.subplots(figsize=(9, 3.4))
    for col, name, colr in [("psi_product", "Product mix", BLUE), ("psi_score", "Risk scores", ORANGE),
                            ("psi_length", "Complaint length", MUTED)]:
        ax.plot(x, res[col], marker="o", color=colr, label=name)
    ax.axhline(PSI_WARN, color=MUTED, ls=":", lw=1)
    ax.axhline(PSI_ALERT, color=ORANGE, ls="--", lw=1)
    ax.text(x.iloc[0], PSI_ALERT, " major shift", va="bottom", color=ORANGE, fontsize=8)
    ax.set_ylabel("PSI vs reference")
    ax.legend(frameon=False, fontsize=8)
    dates(ax)
    ax.set_title("How different is each month from the training data?", loc="left", color=INK, fontsize=11)

    fig2, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(x, res["payout_rate"] * 100, color=INK, marker="o", label="Actual payout rate")
    ax.plot(x, res["pred_static"] * 100, color=ORANGE, marker="o", label="Predicted - recalibrated once (Module 3)")
    ax.plot(x, res["pred_rolling"] * 100, color=BLUE, marker="o", label="Predicted - recalibrated monthly")
    ax.set_ylabel("%")
    ax.legend(frameon=False, fontsize=8)
    dates(ax)
    ax.set_title("Are the payout percentages right?", loc="left", color=INK, fontsize=11)

    figs = fig_html(fig1) + fig_html(fig2)
    if tri is not None:
        fig3, ax = plt.subplots(figsize=(9, 3))
        tx = pd.to_datetime(tri["month"])
        ax.plot(tx, tri["triage_macro_f1"], marker="o", color=BLUE, label="Triage macro-F1")
        ax.plot(x, res["risk_roc_auc"], marker="o", color=ORANGE, label="Risk ROC-AUC")
        ax.legend(frameon=False, fontsize=8)
        dates(ax)
        ax.set_title("Is it still accurate? (needs the true answers, which arrive late)", loc="left", color=INK, fontsize=11)
        figs += fig_html(fig3)

    tbl = pd.DataFrame({
        "Month": res["month"], "Complaints": res["complaints"].map("{:,}".format),
        "Payout rate": res["payout_rate"].map(lambda v: pct(v, 2)),
        "Product-mix PSI": res["psi_product"].map(lambda v: f"{v:.2f} ({level(v)})"),
        "Risk-score PSI": res["psi_score"].map(lambda v: f"{v:.2f} ({level(v)})"),
        "Risk ROC-AUC": res["risk_roc_auc"].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}"),
        "Predicted/actual (once)": res["cal_static"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}"),
        "Predicted/actual (monthly)": res["cal_rolling"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")})
    if tri is not None:
        tbl["Triage macro-F1"] = res["triage_macro_f1"].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}")
    sections = [
        ("What monitoring checks", [
            "Each test month is treated as if the models were live. Inputs and outputs can be checked every day "
            "(no answers needed); accuracy and calibration need the real outcomes, which arrive weeks later.",
            "<b>PSI</b> (population stability index) measures how different a month is from the training data: "
            "below 0.1 stable, 0.1-0.25 moderate shift, above 0.25 major shift.",
            "<b>Predicted/actual</b>: 1.0 = the payout percentages are right on average; 1.5 = 50% too high. "
            f"Months with fewer than {MIN_POS} payouts are too noisy to judge and shown as '-'."], figs, []),
        ("Month by month", [
            f"Recalibrating monthly (on the latest 3 months with known outcomes"
            f"{', skipping months with a major product-mix shift' if args.skip_drift_months else ''}) kept calibration acceptable in "
            f"<b>{n_roll}/{len(judged)}</b> months, versus <b>{n_static}/{len(judged)}</b> when recalibrated once."],
         table_html(tbl, num_cols=tuple(tbl.columns[1:])), [
             "Rule for production: alert when product-mix or risk-score PSI exceeds 0.25, or predicted/actual leaves "
             f"{CAL_OK[0]}-{CAL_OK[1]}; recalibrate monthly; retrain when ranking (ROC-AUC) or triage macro-F1 drops."]),
    ]
    out = config.REPORTS / "m7_monitoring.html"
    out.write_text(build_html("Monitoring replay, 2025-26", "Drift, accuracy and calibration month by month, as if live.",
                              sections), encoding="utf-8")
    print(f"\nReport -> {out}")


if __name__ == "__main__":
    main()
