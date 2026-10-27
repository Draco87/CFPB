"""
Module 6, Step 1: does the risk model (Module 3) treat groups fairly?   (laptop, ~5 minutes)

    uv run python scripts/m6_01_fairness.py

Groups available in the data: complaints tagged "Older American", tagged "Servicemember", and untagged.
(The CFPB data has no gender, race or income - so this is a check on the groups we CAN see, stated as such.)

For each group, four plain questions:
  1. Ranking   - does the model put this group's payout cases near the top as well as for others?
                 ROC-AUC (not PR-AUC: PR-AUC depends on how common payouts are, so it is unfair to compare
                 groups with different payout rates - the Module 3 "Servicemember gap" may be partly that).
  2. Finding   - if agents review the riskiest 10% overall, what share of this group's payout cases get reviewed?
                 (equal opportunity: a fair model should find each group's cases about equally often)
  3. Flagging  - what share of the group lands in that top 10%? (compared with its real payout rate)
  4. Calibration - predicted payout rate vs actual, after the Module 3 recalibration.
Every group-vs-rest difference gets a 95% bootstrap interval: if it includes 0, the gap could be chance.
Also: is a group's different payout rate explained by the products it complains about?

Needs data/processed/m3_{train,val,test}.parquet and models/m3_xgb_full.json (Module 3).
Writes reports/m6_fairness.html and reports/m6_fairness.csv
"""
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from cfpb import config
from cfpb.report import BLUE, INK, MUTED, ORANGE, build_html, fig_html, n, pct, table_html

# reuse Module 3's exact feature lists and category handling (one source of truth)
_spec = importlib.util.spec_from_file_location("m3", config.ROOT / "scripts" / "m3_02_train_explain.py")
m3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m3)
FEATS = m3.CAT + m3.NUM + m3.TEXT

N_BOOT = 500
TOP = 0.10
RNG = np.random.default_rng(config.SEED)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def group_stats(y, s, recal, flagged):
    pos = y == 1
    return {"n": len(y), "payout_rate": y.mean(), "pred_rate": recal.mean(),
            "roc_auc": roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan,
            "pr_auc": average_precision_score(y, s) if y.sum() else np.nan,
            "found": flagged[pos].mean() if pos.any() else np.nan,       # share of payout cases in the top 10%
            "flag_rate": flagged.mean()}


def boot_diff(y, s, flagged, mask, metric):
    """95% interval for (group - rest), resampling each side separately."""
    gi, ri = np.flatnonzero(mask), np.flatnonzero(~mask)

    def val(idx):
        yy = y[idx]
        if metric == "roc_auc":
            return roc_auc_score(yy, s[idx]) if 0 < yy.sum() < len(yy) else np.nan
        return flagged[idx][yy == 1].mean() if yy.any() else np.nan

    d = [val(RNG.choice(gi, len(gi))) - val(RNG.choice(ri, len(ri))) for _ in range(N_BOOT)]
    return np.nanpercentile(d, [2.5, 97.5])


def verdict(lo, hi):
    if lo > 0:
        return "better for this group"
    if hi < 0:
        return "WORSE for this group"
    return "no clear difference"


def main():
    tr, va, te = (pd.read_parquet(config.PROCESSED / f"m3_{s}.parquet") for s in ("train", "val", "test"))
    cats = {c: sorted(tr[c].astype(str).unique()) for c in m3.CAT}
    model = xgb.XGBClassifier()
    model.load_model(str(config.MODELS / "m3_xgb_full.json"))
    s = model.predict_proba(m3.as_cat(te, cats)[FEATS])[:, 1]
    p_va = model.predict_proba(m3.as_cat(va, cats)[FEATS])[:, 1]
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_va), va["y"].to_numpy())
    recal = platt.predict_proba(logit(s))[:, 1]
    y = te["y"].to_numpy()
    flagged = s >= np.quantile(s, 1 - TOP)

    older, sm = te["older_american"].to_numpy() == 1, te["servicemember"].to_numpy() == 1
    groups = {"All complaints": np.ones(len(te), bool), "Older American": older,
              "Servicemember": sm, "No tag": ~older & ~sm}

    # product-mix check: the payout rate a group WOULD have if each of its products paid out at the overall rate
    prod_rate = te.groupby("product")["y"].mean()
    rows, base = [], group_stats(y, s, recal, flagged)
    print(f"Test set: {len(te):,} complaints, payout rate {y.mean():.2%}. Reviewing the riskiest {TOP:.0%} "
          f"finds {base['found']:.0%} of all payout cases.\n")
    for name, m in groups.items():
        if m.sum() < 200:
            print(f"  {name}: only {m.sum()} complaints - skipped")
            continue
        st = group_stats(y[m], s[m], recal[m], flagged[m])
        st["group"] = name
        st["expected_from_products"] = te.loc[m, "product"].map(prod_rate).mean()
        if name != "All complaints":
            for metric in ("roc_auc", "found"):
                lo, hi = boot_diff(y, s, flagged, m, metric)
                st[f"{metric}_diff_lo"], st[f"{metric}_diff_hi"] = lo, hi
        rows.append(st)
    res = pd.DataFrame(rows)
    res.round(4).to_csv(config.REPORTS / "m6_fairness.csv", index=False)

    # ---------- plain-English console summary
    for r in res.itertuples():
        print(f"{r.group}: {r.n:,} complaints | payout {r.payout_rate:.2%} (products alone would predict "
              f"{r.expected_from_products:.2%}) | predicted {r.pred_rate:.2%}")
        print(f"    ranking ROC-AUC {r.roc_auc:.3f} | finds {r.found:.0%} of its payout cases in the top 10% | "
              f"{r.flag_rate:.0%} of the group flagged")
        if r.group != "All complaints":
            print(f"    vs everyone else: ranking {verdict(r.roc_auc_diff_lo, r.roc_auc_diff_hi)} "
                  f"[{r.roc_auc_diff_lo:+.3f} to {r.roc_auc_diff_hi:+.3f}], finding "
                  f"{verdict(r.found_diff_lo, r.found_diff_hi)} "
                  f"[{r.found_diff_lo * 100:+.0f} to {r.found_diff_hi * 100:+.0f} points]")

    # ---------- report
    tbl = pd.DataFrame({
        "Group": res["group"], "Complaints": [n(v) for v in res["n"]],
        "Actual payout rate": [pct(v, 2) for v in res["payout_rate"]],
        "Expected from its products": [pct(v, 2) for v in res["expected_from_products"]],
        "Predicted (recalibrated)": [pct(v, 2) for v in res["pred_rate"]],
        "Ranking (ROC-AUC)": [f"{v:.3f}" for v in res["roc_auc"]],
        "PR-AUC": [f"{v:.3f}" for v in res["pr_auc"]],
        "Payout cases found in top 10%": [pct(v) for v in res["found"]],
        "Share of group flagged": [pct(v) for v in res["flag_rate"]]})
    sub = res[res["group"] != "All complaints"]
    sig = pd.DataFrame({
        "Group vs everyone else": sub["group"],
        "Ranking difference (ROC-AUC)": [f"{lo:+.3f} to {hi:+.3f}" for lo, hi in zip(sub["roc_auc_diff_lo"], sub["roc_auc_diff_hi"])],
        "Ranking verdict": [verdict(lo, hi) for lo, hi in zip(sub["roc_auc_diff_lo"], sub["roc_auc_diff_hi"])],
        "Finding difference (points)": [f"{lo * 100:+.0f} to {hi * 100:+.0f}" for lo, hi in zip(sub["found_diff_lo"], sub["found_diff_hi"])],
        "Finding verdict": [verdict(lo, hi) for lo, hi in zip(sub["found_diff_lo"], sub["found_diff_hi"])]})

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(res) + 1))
    yy = np.arange(len(res))[::-1]
    ax.barh(yy + 0.18, res["found"] * 100, height=0.34, color=BLUE, label="Payout cases found in top 10%")
    ax.barh(yy - 0.18, res["flag_rate"] * 100, height=0.34, color=ORANGE, label="Share of group flagged")
    ax.set_yticks(yy, res["group"])
    ax.set_xlabel("%")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Who gets reviewed if agents check the riskiest 10%?", loc="left", color=INK, fontsize=11)

    worse = [g for g, lo, hi in zip(sub["group"], sub["found_diff_lo"], sub["found_diff_hi"]) if hi < 0]
    worse += [g for g, lo, hi in zip(sub["group"], sub["roc_auc_diff_lo"], sub["roc_auc_diff_hi"]) if hi < 0 and g not in worse]
    sections = [
        ("What this checks", [
            "Module 3's model flags complaints likely to end with the company paying. If agents use it to decide "
            "which complaints to look at first, it should not quietly overlook some groups.",
            "The data only tells us two groups: complaints tagged <b>Older American</b> and <b>Servicemember</b>. "
            "There is no gender, race or income, so this cannot rule out unfairness to groups we cannot see."], "", []),
        ("1. The numbers per group", [
            "<b>Ranking (ROC-AUC)</b>: 0.5 = random, 1 = perfect. Unlike PR-AUC it does not depend on how common "
            "payouts are, so it is the fair way to compare groups with different payout rates.",
            "<b>Payout cases found in top 10%</b>: if agents review the riskiest 10% of all complaints, how many of "
            "this group's real payout cases they would see. Similar values = equal opportunity.",
            "<b>Expected from its products</b>: the payout rate the group would have if it paid out like everyone "
            "else for the same products. Close to the actual rate = the difference is about <i>what</i> people "
            "complain about, not <i>who</i> complains."],
         table_html(tbl, num_cols=tuple(tbl.columns[1:])) + fig_html(fig), []),
        ("2. Are the gaps real or chance?", [
            "For each group we resampled the data 500 times and recomputed the gap to everyone else. The range "
            "shown holds the true gap 95% of the time; if it includes 0, the gap may be chance."],
         table_html(sig), [
             ("<b>Clear gaps against:</b> " + ", ".join(worse) + ". Worth monitoring (Module 7) and checking whether "
              "the inputs for that group are less informative.") if worse else
             "<b>No group is clearly worse off</b> on ranking or on finding its payout cases - the differences are "
             "within what chance alone produces.",
             "PR-AUC is shown for completeness but should not be compared across groups: a group with a lower payout "
             "rate gets a lower PR-AUC even from a perfect-ranking model."]),
    ]
    out = config.REPORTS / "m6_fairness.html"
    out.write_text(build_html("Fairness check - risk model",
                              f"{n(len(te))} test complaints (2025-26); groups from the CFPB 'tags' field.", sections),
                   encoding="utf-8")
    print(f"\nReport -> {out}")


if __name__ == "__main__":
    main()
