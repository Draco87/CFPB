"""
Module 1, Step 4: compare models on the same frozen test set, with uncertainty.

    uv run python scripts/m1_04_compare_models.py

Reads whichever prediction files exist:
    reports/m1_tfidf_logreg_test_preds.parquet
    models/m1_distilbert/test_preds.parquet        (downloaded from Kaggle)

For each model, two ways to set the expected product mix:
    val prior  - mix of the validation period (Jan-Jun 2025; distorted by the Jan 2025 spike)
    EM prior   - mix estimated from the unlabelled test complaints (no labels used)
Reports macro-F1 with 95% bootstrap CIs, a paired comparison between models
(bootstrap CI of the difference + McNemar test), and macro-F1 per month.

Writes reports/m1_comparison.csv, m1_comparison_pairwise.csv, m1_comparison_monthly.csv
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2

from cfpb import config
from cfpb.metrics import class_prior, em_prior, prior_adjust

MODELS = {
    "tfidf_logreg": {"preds": config.REPORTS / "m1_tfidf_logreg_test_preds.parquet", "train_prior": "uniform"},
    "distilbert": {"preds": config.MODELS / "m1_distilbert" / "test_preds.parquet", "train_prior": "empirical"},
}
N_BOOT = 1000
RNG = np.random.default_rng(config.SEED)


def macro_f1(y, p, k):
    """Fast macro-F1 on integer labels (same definition as sklearn's average='macro')."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    present = (cm.sum(0) + cm.sum(1)) > 0
    return (2 * tp / np.maximum(denom, 1))[present].mean()


def get_alpha(name):
    results = config.REPORTS / "m1_results.csv"
    if results.exists():
        r = pd.read_csv(results)
        r = r[r["model"].str.startswith(name)]
        if len(r) and "prior_alpha" in r:
            return float(r["prior_alpha"].iloc[-1])
    rj = config.MODELS / f"m1_{name}" / "results.json"
    if rj.exists():
        return float(json.loads(rj.read_text())["prior_alpha"])
    print(f"  ({name}: alpha not found, using 0.75)")
    return 0.75


def main():
    test = pd.read_parquet(config.PROCESSED / "m1_test.parquet", columns=["complaint_id", "date"])
    train = pd.read_parquet(config.PROCESSED / "m1_train.parquet", columns=["product"])["product"]
    val = pd.read_parquet(config.PROCESSED / "m1_val.parquet", columns=["product"])["product"]

    loaded = {}
    for name, spec in MODELS.items():
        if spec["preds"].exists():
            loaded[name] = pd.read_parquet(spec["preds"])
        else:
            print(f"skip {name}: {spec['preds']} not found")
    if not loaded:
        raise SystemExit("No prediction files found.")

    # Align every model on the same complaints, in the same order
    ids = set(test["complaint_id"])
    for d in loaded.values():
        ids &= set(d["complaint_id"])
    test = test[test["complaint_id"].isin(ids)].sort_values("complaint_id").reset_index(drop=True)
    first = next(iter(loaded.values()))
    classes = np.array(sorted(c[2:] for c in first.columns if c.startswith("p_")))
    k = len(classes)
    enc = {c: i for i, c in enumerate(classes)}
    y = (first.set_index("complaint_id").loc[test["complaint_id"], "y_true"].map(enc).to_numpy())
    n = len(y)
    print(f"{n:,} test complaints, {k} products, models: {list(loaded)}\n")

    # Integer predictions per (model, prior mode)
    preds = {}
    for name, d in loaded.items():
        d = d.set_index("complaint_id").loc[test["complaint_id"]]
        proba = d[[f"p_{c}" for c in classes]].to_numpy()
        tp = (np.full(k, 1 / k) if MODELS[name]["train_prior"] == "uniform" else class_prior(train, classes))
        alpha = get_alpha(name)
        for mode, target in [("val prior", class_prior(val, classes)), ("EM prior", em_prior(proba, tp))]:
            preds[(name, mode)] = prior_adjust(proba, tp, target, alpha).argmax(1)

    # 1. Scores with bootstrap CIs (same resamples for every model -> paired)
    boot_idx = [RNG.integers(0, n, n) for _ in range(N_BOOT)]
    boot = {key: np.array([macro_f1(y[i], p[i], k) for i in boot_idx]) for key, p in preds.items()}
    rows = []
    for (name, mode), p in preds.items():
        lo, hi = np.percentile(boot[(name, mode)], [2.5, 97.5])
        rows.append({"model": name, "prior": mode, "macro_f1": macro_f1(y, p, k),
                     "ci95_low": lo, "ci95_high": hi, "accuracy": (p == y).mean()})
    table = pd.DataFrame(rows)
    print("Macro-F1 with 95% bootstrap CI:")
    print(table.round(4).to_string(index=False), "\n")
    table.round(4).to_csv(config.REPORTS / "m1_comparison.csv", index=False)

    # 2. Paired comparison between models (same prior mode)
    names = list(loaded)
    pair_rows = []
    for mode in ("val prior", "EM prior"):
        for a_i in range(len(names)):
            for b_i in range(a_i + 1, len(names)):
                a, b = names[a_i], names[b_i]
                diff = boot[(b, mode)] - boot[(a, mode)]
                ca, cb = preds[(a, mode)] == y, preds[(b, mode)] == y
                n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
                stat = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
                pair_rows.append({"prior": mode, "comparison": f"{b} - {a}",
                                  "macro_f1_diff": macro_f1(y, preds[(b, mode)], k) - macro_f1(y, preds[(a, mode)], k),
                                  "diff_ci95_low": np.percentile(diff, 2.5),
                                  "diff_ci95_high": np.percentile(diff, 97.5),
                                  f"only {a} right": n01, f"only {b} right": n10,
                                  "mcnemar_p": chi2.sf(stat, 1)})
    if pair_rows:
        pairs = pd.DataFrame(pair_rows)
        print("Paired comparison (CI excluding 0 / McNemar p < 0.05 => real difference):")
        print(pairs.round(4).to_string(index=False), "\n")
        pairs.round(6).to_csv(config.REPORTS / "m1_comparison_pairwise.csv", index=False)

    # 3. Macro-F1 per month (EM prior) -> does performance hold up over time?
    month = test["date"].dt.to_period("M").astype(str).to_numpy()
    monthly = pd.DataFrame({"month": sorted(set(month))})
    monthly["n"] = [int((month == m).sum()) for m in monthly["month"]]
    for name in names:
        p = preds[(name, "EM prior")]
        monthly[name] = [macro_f1(y[month == m], p[month == m], k) for m in monthly["month"]]
    print("Macro-F1 per month (EM prior):")
    print(monthly.round(4).to_string(index=False))
    monthly.round(4).to_csv(config.REPORTS / "m1_comparison_monthly.csv", index=False)


if __name__ == "__main__":
    main()
