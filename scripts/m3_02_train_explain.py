"""
Module 3, Step 2: train, compare and EXPLAIN the risk models.

    uv add xgboost          (once)
    uv run python scripts/m3_02_train_explain.py

Models (from simplest to richest):
  1. Company track record only - no training, just each company's past payout rate
  2. Logistic regression       - complaint details (product, issue, company, state, tags ...)
  3. XGBoost, details only     - same inputs, non-linear model
  4. XGBoost, details + text   - adds what the complaint SAYS (text score + keyword flags)

Scored on 2025-26 complaints the models never saw. Explanations use SHAP values
(XGBoost's built-in TreeSHAP): how much each input pushed each prediction up or down.

Writes reports/m3_report.html (open in a browser), reports/m3_results.csv, models/m3_xgb_full.json
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cfpb import config
from cfpb.report import (BLUE, INK, MUTED, ORANGE, AQUA, build_html, dates, fig_html, n, pct,
                         share_text, table_html)

CAT = ["product", "sub_product", "issue", "company_group", "state_group", "submitted_via"]
NUM = ["older_american", "servicemember", "month_of_year", "log_length", "hist_rate_company",
       "hist_rate_issue", "hist_rate_sub_product", "hist_rate_product", "hist_company_volume"]
TEXT = ["text_score", "kw_refund", "kw_unauthorized", "kw_fraud", "kw_fee", "kw_overdraft",
        "kw_double_charge", "kw_dispute", "kw_identity_theft"]
LABELS = {  # plain-English names for the report
    "product": "Product", "sub_product": "Sub-product", "issue": "Issue", "company_group": "Company",
    "state_group": "State", "submitted_via": "Submitted via", "older_american": "Older American tag",
    "servicemember": "Servicemember tag", "month_of_year": "Month of year", "log_length": "Complaint length",
    "hist_rate_company": "Company's past payout rate", "hist_rate_issue": "Issue's past payout rate",
    "hist_rate_sub_product": "Sub-product's past payout rate", "hist_rate_product": "Product's past payout rate",
    "hist_company_volume": "Company's past complaint volume", "text_score": "What the complaint says (text model)",
    "kw_refund": "Mentions a refund", "kw_unauthorized": "Mentions 'unauthorized'", "kw_fraud": "Mentions fraud/scam",
    "kw_fee": "Mentions fees/charges", "kw_overdraft": "Mentions overdraft", "kw_double_charge": "Mentions a double charge",
    "kw_dispute": "Mentions a dispute", "kw_identity_theft": "Mentions identity theft"}
N_BOOT = 300
RNG = np.random.default_rng(config.SEED)


def capture_at(y, s, frac):
    """Share of all payout cases found if you review the top `frac` riskiest complaints."""
    k = max(1, int(len(s) * frac))
    top = np.argsort(-s)[:k]
    return y[top].sum() / max(1, y.sum())


def scores(y, s):
    return {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
            "capture_top10": capture_at(y, s, 0.10), "brier": brier_score_loss(y, np.clip(s, 0, 1))}


def show(c, val):
    """Feature value in plain words."""
    if c in CAT:
        return str(val)
    if c.startswith("hist_rate") or c == "text_score":
        return pct(val)
    if c.startswith("kw_") or c in ("older_american", "servicemember"):
        return "yes" if val == 1 else "no"
    if c == "log_length":
        return f"{int(np.expm1(val)):,} characters"
    if c == "hist_company_volume":
        return f"{int(np.expm1(val)):,} past complaints"
    return f"{val:g}"


def as_cat(df, cats_from):
    out = df[CAT + NUM + TEXT].copy()
    for c in CAT:
        out[c] = pd.Categorical(out[c], categories=cats_from[c])
    return out


def main():
    tr, va, te = (pd.read_parquet(config.PROCESSED / f"m3_{s}.parquet") for s in ("train", "val", "test"))
    meta = json.loads((config.PROCESSED / "m3_meta.json").read_text())
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,} | test payout rate {te['y'].mean():.2%}")
    y_tr, y_va, y_te = tr["y"].to_numpy(), va["y"].to_numpy(), te["y"].to_numpy()
    cats = {c: sorted(tr[c].astype(str).unique()) for c in CAT}
    preds = {}

    # 1. company track record (no training)
    preds["1. Company track record only"] = te["hist_rate_company"].to_numpy()

    # 2. logistic regression on details
    lr = Pipeline([("prep", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CAT),
        ("num", StandardScaler(), NUM)])), ("clf", LogisticRegression(max_iter=3000, C=1.0))])
    lr.fit(tr[CAT + NUM], y_tr)
    preds["2. Logistic regression (details)"] = lr.predict_proba(te[CAT + NUM])[:, 1]

    # 3 + 4. XGBoost
    def fit_xgb(cols):
        m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=6, subsample=0.8,
                              colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                              enable_categorical=True, max_cat_to_onehot=1, eval_metric="aucpr",
                              early_stopping_rounds=100, n_jobs=-1, random_state=config.SEED)
        Xtr, Xva = as_cat(tr, cats)[cols], as_cat(va, cats)[cols]
        m.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=250)
        return m

    xgb_details = fit_xgb(CAT + NUM)
    preds["3. XGBoost (details)"] = xgb_details.predict_proba(as_cat(te, cats)[CAT + NUM])[:, 1]
    xgb_full = fit_xgb(CAT + NUM + TEXT)
    X_te = as_cat(te, cats)[CAT + NUM + TEXT]
    preds["4. XGBoost (details + text)"] = xgb_full.predict_proba(X_te)[:, 1]
    xgb_full.save_model(str(config.MODELS / "m3_xgb_full.json"))

    # ---- scores + bootstrap CIs
    boot_idx = [RNG.integers(0, len(y_te), len(y_te)) for _ in range(N_BOOT)]
    rows, boot_pr = [], {}
    for name, s in preds.items():
        sc = scores(y_te, s)
        boot_pr[name] = np.array([average_precision_score(y_te[i], s[i]) for i in boot_idx])
        lo, hi = np.percentile(boot_pr[name], [2.5, 97.5])
        rows.append({"model": name, **sc, "pr_auc_ci_low": lo, "pr_auc_ci_high": hi})
    res = pd.DataFrame(rows)
    res.round(4).to_csv(config.REPORTS / "m3_results.csv", index=False)
    print("\n", res.round(4).to_string(index=False))
    full, det = "4. XGBoost (details + text)", "3. XGBoost (details)"
    diff = boot_pr[full] - boot_pr[det]
    text_gain = (res.set_index("model").loc[full, "pr_auc"] - res.set_index("model").loc[det, "pr_auc"],
                 *np.percentile(diff, [2.5, 97.5]))

    # ---- SHAP values (exact TreeSHAP, built into XGBoost)
    sample = te.sample(min(5000, len(te)), random_state=config.SEED)
    Xs = as_cat(sample, cats)[CAT + NUM + TEXT]
    contrib = xgb_full.get_booster().predict(xgb.DMatrix(Xs, enable_categorical=True), pred_contribs=True)
    shap = pd.DataFrame(contrib[:, :-1], columns=Xs.columns, index=sample.index)
    importance = shap.abs().mean().sort_values(ascending=False)

    # ================= REPORT =================
    base = y_te.mean()
    R = res.set_index("model")
    best = R.loc[full]
    sections = []

    sections.append(("The question", [
        "When a complaint arrives, can we predict whether it will end with the company <b>paying the consumer "
        "money</b>? Those are the complaints where the company was most likely at fault, so a bank would want "
        "to look at them first.",
        f"In the 2025-26 test period, {share_text(base)} complaints ended with a payout. That is rare, "
        "which is why we judge the models by how well they <i>rank</i> complaints, not by accuracy "
        f"(a model that always says 'no payout' would be {pct(1 - base)} accurate and useless).",
        "Each model only uses what is known when the complaint arrives: product, issue, company, state, tags, "
        "the text, and payout rates learned from 2021-22. The company's eventual answer is never used as an input."],
        "", []))

    # payout rate by product (from train)
    by_prod = tr.groupby("product")["y"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(by_prod.index, by_prod.values * 100, color=BLUE, height=0.6)
    for yv, v in enumerate(by_prod.values):
        ax.text(v * 100, yv, f" {pct(v)}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, by_prod.max() * 118)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_title("Share of complaints that ended with a payout, by product (2023-24)", loc="left", color=INK, fontsize=11)
    sections.append(("Which complaints end with a payout?", [
        f"Payouts are far more common for <b>{by_prod.index[-1]}</b> ({pct(by_prod.iloc[-1])}) "
        f"and <b>{by_prod.index[-2]}</b> ({pct(by_prod.iloc[-2])}) than for <b>{by_prod.index[0]}</b> "
        f"({pct(by_prod.iloc[0])})."], fig_html(fig), []))

    # model comparison
    tbl = pd.DataFrame({
        "Model": R.index,
        "PR-AUC (95% CI)": [f"{r.pr_auc:.3f} ({r.pr_auc_ci_low:.3f}-{r.pr_auc_ci_high:.3f})" for r in R.itertuples()],
        "ROC-AUC": [f"{v:.3f}" for v in R["roc_auc"]],
        "Payouts found in riskiest 10%": [pct(v) for v in R["capture_top10"]]})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fr = np.linspace(0, 1, 101)
    colors = {list(preds)[0]: MUTED, list(preds)[1]: AQUA, det: ORANGE, full: BLUE}
    for name, s in preds.items():
        srt = y_te[np.argsort(-s)]
        cum = np.concatenate([[0], np.cumsum(srt) / srt.sum()])
        xs = np.linspace(0, 1, len(cum))
        ax.plot(xs * 100, cum * 100, color=colors[name], lw=2 if name == full else 1.4, label=name)
    ax.plot([0, 100], [0, 100], color=MUTED, ls="--", lw=1, label="Random order")
    ax.set_xlim(0, 50)
    ax.set_xlabel("% of complaints reviewed, riskiest first")
    ax.set_ylabel("% of all payout cases found")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("How quickly each model finds the payout cases", loc="left", color=INK, fontsize=11)
    sections.append(("How good are the models?", [
        f"If the bank reviews the riskiest <b>10%</b> of complaints first, the best model finds "
        f"<b>{pct(best.capture_top10)}</b> of all payout cases - <b>{best.capture_top10 / 0.10:.1f}x</b> better "
        f"than reviewing in random order (which would find 10%).",
        f"Adding the complaint text changed PR-AUC by <b>{text_gain[0]:+.3f}</b> "
        f"(95% CI {text_gain[1]:+.3f} to {text_gain[2]:+.3f})"
        + (" - a real improvement." if text_gain[1] > 0 else " - not a clear improvement."),
        "PR-AUC is the main score here: higher means payout cases are ranked nearer the top. "
        f"Random guessing would score about {base:.3f}."],
        table_html(tbl, num_cols=("ROC-AUC", "Payouts found in riskiest 10%")) + fig_html(fig), []))

    # SHAP importance
    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh([LABELS.get(c, c) for c in top.index], top.values, color=BLUE, height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average impact on the prediction (mean |SHAP|, log-odds)")
    ax.set_title("What drives the risk score", loc="left", color=INK, fontsize=11)
    top5 = [LABELS.get(c, c) for c in importance.index[:5]]
    sections.append(("What drives the predictions? (explainability)", [
        "SHAP values show how much each input pushed each prediction up or down. Averaged over 5,000 test "
        "complaints, the biggest drivers are: <b>" + "</b>, <b>".join(top5) + "</b>."],
        fig_html(fig), []))

    # three example explanations
    sample = sample.assign(score=xgb_full.predict_proba(Xs)[:, 1])
    picks = [sample["score"].idxmax(), (sample["score"] - sample["score"].median()).abs().idxmin(),
             sample["score"].idxmin()]
    ex_bullets = []
    for label, i in zip(["Highest risk", "Typical", "Lowest risk"], picks):
        row, contrib_row = sample.loc[i], shap.loc[i].sort_values(key=np.abs, ascending=False).head(3)
        reasons = []
        for c, v in contrib_row.items():
            shown = show(c, row[c])
            reasons.append(f"{LABELS.get(c, c)} = <i>{shown}</i> ({'raises' if v > 0 else 'lowers'} risk)")
        ex_bullets.append(f"<b>{label}</b> - {row['product']} / {row['issue']} at {row['company']}: predicted "
                          f"{pct(row['score'])} chance of payout (actual: {'payout' if row['y'] else 'no payout'}). "
                          "Main reasons: " + "; ".join(reasons) + ".")
    sections.append(("Three example explanations", [
        "For any single complaint, the model can say <i>why</i> it gave that score - the basis for showing "
        "an agent the reasons, not just a number."], "", ex_bullets))

    # calibration
    s = preds[full]
    bins = pd.qcut(s, 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"pred": s, "y": y_te, "bin": bins}).groupby("bin").mean()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    lim = max(cal["pred"].max(), cal["y"].max()) * 105
    ax.plot([0, lim], [0, lim], color=MUTED, ls="--", lw=1)
    ax.plot(cal["pred"] * 100, cal["y"] * 100, "o-", color=BLUE, lw=2, ms=6)
    ax.set_xlabel("Predicted payout chance (%)")
    ax.set_ylabel("Actual payout rate (%)")
    ax.set_title("Do the predicted chances match reality?", loc="left", color=INK, fontsize=11)
    worst = (cal["pred"] - cal["y"]).abs().idxmax()
    sections.append(("Can the percentages be trusted? (calibration)", [
        "Complaints were grouped into 10 bins by predicted chance. If the dots sit on the dashed line, "
        "'20% chance' really means about 20 in 100 get a payout.",
        f"In the riskiest bin the model predicted {pct(cal['pred'].iloc[-1])} and the real rate was "
        f"{pct(cal['y'].iloc[-1])}. The biggest gap is in bin {int(worst) + 1}: predicted "
        f"{pct(cal.loc[worst, 'pred'])} vs actual {pct(cal.loc[worst, 'y'])}."], fig_html(fig), []))

    # drift over time
    allm = pd.concat([tr.assign(w="train"), va.assign(w="val"), te.assign(w="test")])
    monthly = allm.groupby(allm["date"].dt.to_period("M"))["y"].mean()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(monthly.index.to_timestamp(), monthly.values * 100, color=BLUE, lw=2)
    for d_, lbl in [("2024-07-01", "val"), ("2025-01-01", "test")]:
        ax.axvline(pd.Timestamp(d_), color=MUTED, ls="--", lw=1)
        ax.text(pd.Timestamp(d_), monthly.max() * 100, f" {lbl}", va="top", color=MUTED, fontsize=9)
    dates(ax)
    ax.set_title("Payout rate by month", loc="left", color=INK, fontsize=11)
    sections.append(("Is the payout rate stable over time?", [
        f"Payout rate: {pct(y_tr.mean())} in training (2023 - mid 2024), {pct(y_va.mean())} in validation, "
        f"{pct(base)} in test (2025-26).",
        "If the rate shifts, a model's percentages drift out of line even if its ranking stays good - "
        "a reason to monitor it (Module 7)."], fig_html(fig), []))

    # groups (Responsible AI preview)
    grp_rows = []
    for name, mask in [("All complaints", np.ones(len(te), bool)),
                       ("Older American", te["older_american"].to_numpy() == 1),
                       ("Servicemember", te["servicemember"].to_numpy() == 1)]:
        if mask.sum() < 50:
            continue
        grp_rows.append({"Group": name, "Complaints": n(mask.sum()), "Actual payout rate": pct(y_te[mask].mean()),
                         "Average predicted": pct(s[mask].mean()),
                         "PR-AUC": f"{average_precision_score(y_te[mask], s[mask]):.3f}" if y_te[mask].sum() else "-"})
    sections.append(("Does it work equally well for different groups? (first look)", [
        "A fair model should rank complaints about as well for tagged groups as for everyone, and its average "
        "prediction should match each group's real rate. Module 6 looks at this in depth."],
        table_html(pd.DataFrame(grp_rows), num_cols=("Complaints", "Actual payout rate", "Average predicted", "PR-AUC")),
        []))

    out = config.REPORTS / "m3_report.html"
    out.write_text(build_html("Complaint risk model - will the company pay?",
                              f"Trained on 2023 - mid 2024, tested on {n(len(te))} complaints from 2025-26. "
                              f"Leakage guard: {', '.join(meta['excluded_for_leakage'])} are never used as inputs.",
                              sections), encoding="utf-8")
    print(f"\nTop drivers: {top5}")
    print(f"Text gain in PR-AUC: {text_gain[0]:+.4f} (95% CI {text_gain[1]:+.4f} to {text_gain[2]:+.4f})")
    print(f"Report -> {out}")


if __name__ == "__main__":
    main()
