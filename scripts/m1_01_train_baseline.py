"""
Module 1, Step 2: TF-IDF + Logistic Regression baseline for product classification.

    uv run python scripts/m1_00_make_splits.py      # first (once)
    uv run python scripts/m1_01_train_baseline.py

Writes  models/m1_tfidf_logreg.joblib
        reports/m1_tfidf_logreg_{val,test}_report.txt
        reports/figures/m1_tfidf_logreg_confusion.png
        reports/m1_tfidf_logreg_test_preds.parquet   (for error analysis + significance tests)
        reports/m1_results.csv                        (comparison table, one row per model)
"""
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from cfpb import config
from cfpb.metrics import append_result, class_prior, prior_adjust, tune_alpha

NAME = "tfidf_logreg"


def load_splits():
    paths = {s: config.PROCESSED / f"m1_{s}.parquet" for s in ("train", "val", "test")}
    if not all(p.exists() for p in paths.values()):
        raise SystemExit("Splits missing - run: uv run python scripts/m1_00_make_splits.py")
    meta = json.loads((config.PROCESSED / "m1_splits_meta.json").read_text())
    return [pd.read_parquet(paths[s]) for s in ("train", "val", "test")] + [meta]


def build_model():
    return Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                                     sublinear_tf=True, stop_words="english")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                     max_features=200_000, sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")),
    ])


def main():
    train, val, test, meta = load_splits()
    name = f"{NAME}[{meta['train_mode']}]"
    print(f"{name}: train {len(train):,} | val {len(val):,} | test {len(test):,}")

    model = build_model()
    t0 = time.time()
    model.fit(train["text"], train["product"])
    print(f"trained in {time.time() - t0:.0f}s")
    joblib.dump(model, config.MODELS / "m1_tfidf_logreg.joblib")
    classes = model.classes_

    # class_weight="balanced" makes the model behave as if all products were equally common.
    # Tune how far to shift back toward the real mix, using VAL only.
    train_prior = np.full(len(classes), 1 / len(classes))
    target_prior = class_prior(val["product"], classes)
    val_proba = model.predict_proba(val["text"])
    alpha, scores = tune_alpha(val_proba, val["product"], classes, train_prior, target_prior)
    print("VAL macro-F1 by prior-shift alpha:", {a: round(s, 4) for a, s in scores.items()},
          f"-> using alpha={alpha}")
    val_pred = classes[prior_adjust(val_proba, train_prior, target_prior, alpha).argmax(1)]
    val_report = classification_report(val["product"], val_pred, digits=3, zero_division=0)
    (config.REPORTS / f"m1_{NAME}_val_report.txt").write_text(val_report)
    print("VAL report\n", val_report)

    # TEST: touched once, with everything already decided on val
    t0 = time.time()
    test_proba = model.predict_proba(test["text"])
    ms_per_1k = (time.time() - t0) / len(test) * 1e6
    pred = classes[prior_adjust(test_proba, train_prior, target_prior, alpha).argmax(1)]

    row = {"model": name, "train_rows": len(train),
           "accuracy": accuracy_score(test["product"], pred),
           "macro_f1": f1_score(test["product"], pred, average="macro"),
           "weighted_f1": f1_score(test["product"], pred, average="weighted"),
           "prior_alpha": alpha, "ms_per_1k": ms_per_1k, "hardware": "cpu", "cost_per_1k_usd": 0.0}
    print("TEST", {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

    test_report = classification_report(test["product"], pred, digits=3, zero_division=0)
    (config.REPORTS / f"m1_{NAME}_test_report.txt").write_text(test_report)
    print("TEST report\n", test_report)

    preds = pd.DataFrame({"complaint_id": test["complaint_id"].to_numpy(),
                          "y_true": test["product"].to_numpy(), "y_pred": pred})
    preds = pd.concat([preds, pd.DataFrame(test_proba, columns=[f"p_{c}" for c in classes])], axis=1)
    preds.to_parquet(config.REPORTS / f"m1_{NAME}_test_preds.parquet", index=False)

    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay.from_predictions(test["product"], pred, normalize="true",
                                            xticks_rotation=45, values_format=".2f",
                                            ax=ax, colorbar=False)
    ax.set_title(f"{name}, test set (rows = true product, normalised)")
    plt.tight_layout()
    plt.savefig(config.FIGURES / f"m1_{NAME}_confusion.png", dpi=150)

    append_result(row, config.REPORTS / "m1_results.csv")
    print("results ->", config.REPORTS / "m1_results.csv")


if __name__ == "__main__":
    main()