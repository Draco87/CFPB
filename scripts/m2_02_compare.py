"""
Module 2: compare every model on the same 900 test complaints (100 per product).

    uv run python scripts/m2_02_compare.py

Needs (downloaded from Kaggle):  models/m2/base_preds.parquet, ft_preds.parquet, m2_results.json
Also uses Module 1's files when present (LLM 7B, DistilBERT, TF-IDF).
Writes reports/m2_product_comparison.csv and reports/m2_fields.csv
"""
import json

import numpy as np
import pandas as pd

from cfpb import config

M2 = config.MODELS / "m2"
N_BOOT = 1000


def macro_f1(y, p, classes):
    s = []
    for c in classes:
        tp = np.sum((y == c) & (p == c))
        d = 2 * tp + np.sum((y != c) & (p == c)) + np.sum((y == c) & (p != c))
        s.append(2 * tp / d if d else 0.0)
    return float(np.mean(s))


def main():
    ft_file = M2 / "ft_preds.parquet"
    if not ft_file.exists():
        raise SystemExit(f"{ft_file} not found - download Module 2 outputs from Kaggle first.")
    ft = pd.read_parquet(ft_file)
    ids, y = ft["complaint_id"].to_numpy(), ft["y_true"].to_numpy()
    classes = sorted(set(y))
    res = json.loads((M2 / "m2_results.json").read_text())

    # (name, predictions, ms per complaint, labelled examples used)
    rows = []

    def add(name, preds, ms, labels):
        rows.append((name, np.asarray(preds), ms, labels))

    add("Qwen 1.5B fine-tuned (LoRA)", ft["y_pred"], res["finetuned"]["sec_per_complaint"] * 1000,
        f"{res['n_train']:,}")
    if (M2 / "base_preds.parquet").exists():
        b = pd.read_parquet(M2 / "base_preds.parquet").set_index("complaint_id").loc[ids]
        add("Qwen 1.5B untrained", b["y_pred"], res["base"]["sec_per_complaint"] * 1000, "0")
    llm_dir = config.MODELS / "m1_llm"
    llm_res = json.loads((llm_dir / "llm_results.json").read_text()) if (llm_dir / "llm_results.json").exists() else {}
    for mode, lab in [("zero", "0"), ("few", "9")]:
        f = llm_dir / f"llm_preds_{mode}.parquet"
        if f.exists():
            d = pd.read_parquet(f).set_index("complaint_id").loc[ids]
            ms = llm_res.get("modes", {}).get(mode, {}).get("sec_per_complaint", np.nan) * 1000
            add(f"Qwen 7B {mode}-shot", d["y_pred"], ms, lab)
    for name, path, rj in [("DistilBERT", config.MODELS / "m1_distilbert" / "test_preds.parquet",
                            config.MODELS / "m1_distilbert" / "results.json"),
                           ("TF-IDF + LogReg", config.REPORTS / "m1_tfidf_logreg_test_preds.parquet", None)]:
        if path.exists():
            d = pd.read_parquet(path).set_index("complaint_id").loc[ids]
            p = np.array(classes)[d[[f"p_{c}" for c in classes]].to_numpy().argmax(1)]
            ms = json.loads(rj.read_text()).get("ms_per_1k", np.nan) / 1000 if rj and rj.exists() else np.nan
            add(name, p, ms, "174,088")

    rng = np.random.default_rng(config.SEED)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    table = []
    for name, p, ms, lab in rows:
        b = [macro_f1(y[i], p[i], classes) for i in boots]
        table.append({"model": name, "macro_f1": macro_f1(y, p, classes),
                      "ci95_low": np.percentile(b, 2.5), "ci95_high": np.percentile(b, 97.5),
                      "accuracy": float(np.mean(p == y)), "ms_per_complaint": ms, "labels_used": lab})
    table = pd.DataFrame(table).sort_values("macro_f1", ascending=False)
    print(f"PRODUCT - same {len(y)} test complaints ({len(y) // len(classes)} per product):")
    print(table.round(4).to_string(index=False))
    table.round(4).to_csv(config.REPORTS / "m2_product_comparison.csv", index=False)

    # Structured fields: fine-tuned model vs a naive baseline
    # (naive = "if you knew the product, guess its most common sub-product / issue in training")
    train = pd.read_parquet(config.PROCESSED / "m1_train.parquet", columns=["date", "product", "sub_product", "issue"])
    train = train[train["date"] >= "2023-01-01"].fillna("None")
    f = res["finetuned"]
    fields = []
    ok = (ft["y_pred"] == ft["y_true"]).to_numpy()          # rows where the model got the product right
    for field in ["sub_product", "issue"]:
        top = train.groupby("product")[field].agg(lambda s: s.value_counts().index[0])
        truth = ft[f"true_{field}"].to_numpy()
        model_hit = (ft[f"pred_{field}"].to_numpy() == truth)
        naive_true = (ft["y_true"].map(top).fillna("None").to_numpy() == truth)
        naive_pred = (ft["y_pred"].map(top).fillna("None").to_numpy() == truth)
        fields.append({"field": field,
                       "fine-tuned": model_hit.mean(),
                       "naive, uses model's product (fair)": naive_pred.mean(),
                       "naive, given TRUE product": naive_true.mean(),
                       "fine-tuned | product right": model_hit[ok].mean(),
                       "naive | product right": naive_true[ok].mean()})
    fields = pd.DataFrame(fields)
    print("\n'naive' = always guess the most common sub-product/issue for the product.")
    print(f"\nSTRUCTURED FIELDS (fine-tuned 1.5B) | valid JSON: {f['json_valid_rate']:.1%} | "
          f"all three fields right: {f['all_three_correct']:.1%}")
    print(fields.round(4).to_string(index=False))
    fields.round(4).to_csv(config.REPORTS / "m2_fields.csv", index=False)

    wrong = ft[ft["pred_issue"] != ft["true_issue"]].head(5)
    print("\nA few issue mistakes (true -> predicted):")
    for _, r in wrong.iterrows():
        print(f"  [{r['y_true']}] {r['true_issue']}  ->  {r['pred_issue']}")


if __name__ == "__main__":
    main()
