"""
Module 1, Step 2: TF-IDF + Logistic Regression baseline for product classification.

    python scripts/m1_01_train_baseline.py

Needs   data/interim/narratives_parts/  (run scripts/00_combine_archive.py first)
Writes  data/processed/m1_{train,val,test}.parquet   same splits reused by DistilBERT & LLM steps
        models/m1_tfidf_logreg.joblib
        reports/m1_baseline_report.txt
        reports/figures/m1_baseline_confusion.png
        reports/m1_results.csv                        comparison table, one row per model
"""
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from cfpb import config
from cfpb.data import load_sample, time_split

N_PER_PART = 15_000   # 13 parts x 15k ~ 195k rows. Lower if you run out of RAM.
COLS = ["complaint_id", "date", "product", "issue", "text", "n_chars"]


def build_splits():
    df = load_sample(N_PER_PART, columns=COLS)
    df = df.dropna(subset=["product"])
    df = df[df["n_chars"] >= 50]                      # drop near-empty narratives
    train, val, test = time_split(df)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        d.to_parquet(config.PROCESSED / f"m1_{name}.parquet", index=False)
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")
    if min(len(train), len(val), len(test)) == 0:
        raise SystemExit("A split is empty - check the date ranges of your archive files.")
    return train, val, test


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
    train, val, test = build_splits()
    print(train["product"].value_counts(), "\n")

    model = build_model()
    t0 = time.time()
    model.fit(train["text"], train["product"])
    print(f"trained in {time.time() - t0:.0f}s")
    joblib.dump(model, config.MODELS / "m1_tfidf_logreg.joblib")

    # Tune hyperparameters against VAL only; look at TEST once, at the end.
    val_pred = model.predict(val["text"])
    print(f"VAL  macro-F1 {f1_score(val['product'], val_pred, average='macro'):.4f}")

    t0 = time.time()
    pred = model.predict(test["text"])
    ms_per_1k = (time.time() - t0) / len(test) * 1e6
    row = {
        "model": "tfidf_logreg",
        "accuracy": accuracy_score(test["product"], pred),
        "macro_f1": f1_score(test["product"], pred, average="macro"),
        "weighted_f1": f1_score(test["product"], pred, average="weighted"),
        "ms_per_1k": ms_per_1k,
        "cost_per_1k_usd": 0.0,
    }
    print("TEST", {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

    report = classification_report(test["product"], pred, digits=3, zero_division=0)
    (config.REPORTS / "m1_baseline_report.txt").write_text(report)
    print(report)

    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay.from_predictions(test["product"], pred, normalize="true",
                                            xticks_rotation=45, values_format=".2f",
                                            ax=ax, colorbar=False)
    ax.set_title("TF-IDF + LogReg, test set (rows = true product)")
    plt.tight_layout()
    plt.savefig(config.FIGURES / "m1_baseline_confusion.png", dpi=150)

    results_file = config.REPORTS / "m1_results.csv"
    results = pd.read_csv(results_file) if results_file.exists() else pd.DataFrame()
    results = results[results.get("model", pd.Series(dtype=str)) != row["model"]] if len(results) else results
    pd.concat([results, pd.DataFrame([row])], ignore_index=True).to_csv(results_file, index=False)
    print(f"results -> {results_file}")


if __name__ == "__main__":
    main()
