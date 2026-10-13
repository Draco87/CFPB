"""
Module 1 diagnostic: does the product mix shift over time, and how much does it cost the model?

    uv run python scripts/m1_03_label_shift.py
    uv run python scripts/m1_03_label_shift.py --preds models/m1_distilbert/test_preds.parquet --train-prior empirical

1. Monthly complaint volume and product shares (all 3M complaints) -> reports/m1_monthly_mix.csv + figure
2. Product mix: train vs val vs test, and the mix the EM method estimates for test WITHOUT labels
3. Test macro-F1 under four prior settings:
     none            - raw model output
     val prior       - what the baseline reports (mix learned from Jan-Jun 2025)
     EM prior        - mix estimated from unlabelled test data (deployable; needs well-calibrated
                       probabilities, so it may or may not beat 'val prior' - that is the experiment)
     oracle prior    - true test mix (uses test labels: diagnosis only, NEVER report as a result.
                       It maximises accuracy, not necessarily macro-F1)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from cfpb import config
from cfpb.data import load_combined
from cfpb.metrics import class_prior, em_prior, prior_adjust


def monthly_mix():
    meta = load_combined(["date", "product"]).dropna(subset=["product"])
    month = meta["date"].dt.to_period("M")
    shares = pd.crosstab(month, meta["product"], normalize="index") * 100
    shares.insert(0, "volume", month.value_counts().sort_index())
    shares.round(2).to_csv(config.REPORTS / "m1_monthly_mix.csv")

    recent = shares[shares.index >= pd.Period("2024-07", "M")]
    print("Monthly volume and product share (%), from Jul 2024:")
    print(recent.round(1).to_string(), "\n")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, height_ratios=[1, 2])
    x = shares.index.to_timestamp()
    a1.bar(x, shares["volume"], width=20, color="#888")
    a1.set_ylabel("complaints / month")
    for c in shares.columns[1:]:
        a2.plot(x, shares[c], label=c, lw=1.6)
    a2.set_ylabel("share of month (%)")
    for d, lbl in [(config.VAL_START, "val"), (config.TEST_START, "test")]:
        for ax in (a1, a2):
            ax.axvline(pd.Timestamp(d), color="k", ls="--", lw=0.8)
        a2.text(pd.Timestamp(d), a2.get_ylim()[1] * 0.95, f" {lbl}", va="top")
    a2.legend(fontsize=8, ncol=2, loc="center left")
    a1.set_title("CFPB complaints with narratives: volume and product mix over time")
    plt.tight_layout()
    plt.savefig(config.FIGURES / "m1_monthly_mix.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(config.REPORTS / "m1_tfidf_logreg_test_preds.parquet"))
    ap.add_argument("--train-prior", choices=["uniform", "empirical"], default="uniform",
                    help="uniform for the TF-IDF model (class_weight=balanced), empirical for DistilBERT")
    ap.add_argument("--alpha", type=float, default=0.75, help="the alpha the model chose on val")
    args = ap.parse_args()

    monthly_mix()

    preds = pd.read_parquet(args.preds)
    pcols = [c for c in preds.columns if c.startswith("p_")]
    classes = np.array([c[2:] for c in pcols])
    proba = preds[pcols].to_numpy()
    y = preds["y_true"].to_numpy()

    train = pd.read_parquet(config.PROCESSED / "m1_train.parquet", columns=["product"])["product"]
    val = pd.read_parquet(config.PROCESSED / "m1_val.parquet", columns=["product"])["product"]
    train_prior = (np.full(len(classes), 1 / len(classes)) if args.train_prior == "uniform"
                   else class_prior(train, classes))
    priors = {
        "val prior": class_prior(val, classes),
        "EM prior (no labels)": em_prior(proba, train_prior),
        "oracle (true test mix)": class_prior(y, classes),
    }

    mix = pd.DataFrame({"train": class_prior(train, classes), **priors}, index=classes) * 100
    print("Product mix (%):")
    print(mix.round(1).to_string(), "\n")

    rows = []
    for name, target, a in [("none", train_prior, 0.0)] + [(k, v, args.alpha) for k, v in priors.items()]:
        pred = classes[prior_adjust(proba, train_prior, target, a).argmax(1)]
        rows.append({"prior used": name,
                     "macro_f1": f1_score(y, pred, average="macro"),
                     "accuracy": accuracy_score(y, pred),
                     "DebtColl recall": recall_score(y, pred, labels=["Debt collection"], average="macro", zero_division=0),
                     "MoneyTr precision": precision_score(y, pred, labels=["Money transfer / virtual currency"],
                                                          average="macro", zero_division=0)})
    out = pd.DataFrame(rows).set_index("prior used")
    print(f"TEST scores ({args.preds}):")
    print(out.round(4).to_string())
    print("\n'oracle' uses test labels: diagnosis only, not a reportable result.")
    out.round(4).to_csv(config.REPORTS / "m1_label_shift.csv")


if __name__ == "__main__":
    main()
