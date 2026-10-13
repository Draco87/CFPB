"""
Module 1, Step 6: compare the LLM with the two trained models on the SAME 900 complaints.

    uv run python scripts/m1_07_compare_llm.py

Needs models/m1_llm/llm_preds_*.parquet and llm_results.json (downloaded from Kaggle),
plus the two models' test predictions.

The 900 complaints are 100 per product (balanced), unlike the real mix, so for the trained
models two versions are shown:
    as deployed  - with the prior adjustment tuned for the real mix
    no prior adj - raw model output (fairer on a balanced sample; the LLM has no prior either)

Writes reports/m1_llm_comparison.csv
"""
import json

import numpy as np
import pandas as pd

from cfpb import config

LLM_DIR = config.MODELS / "m1_llm"
TRAINED = {
    "tfidf_logreg": config.REPORTS / "m1_tfidf_logreg_test_preds.parquet",
    "distilbert": config.MODELS / "m1_distilbert" / "test_preds.parquet",
}
N_BOOT = 1000


def macro_f1(y, p, classes):
    scores = []
    for c in classes:
        tp = np.sum((y == c) & (p == c))
        denom = 2 * tp + np.sum((y != c) & (p == c)) + np.sum((y == c) & (p != c))
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def main():
    llm_files = sorted(LLM_DIR.glob("llm_preds_*.parquet"))
    if not llm_files:
        raise SystemExit(f"No LLM predictions in {LLM_DIR} - download them from Kaggle first.")
    first = pd.read_parquet(llm_files[0])
    ids = first["complaint_id"].to_numpy()
    y = first["y_true"].to_numpy()
    classes = sorted(set(y))

    preds = {}
    for f in llm_files:
        d = pd.read_parquet(f).set_index("complaint_id").loc[ids]
        preds[f"LLM {f.stem.split('_')[-1]}-shot"] = d["y_pred"].to_numpy()
    for name, path in TRAINED.items():
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        d = pd.read_parquet(path).set_index("complaint_id").loc[ids]
        pcols = [f"p_{c}" for c in classes]
        preds[f"{name} (as deployed)"] = d["y_pred"].to_numpy()
        preds[f"{name} (no prior adj)"] = np.array(classes)[d[pcols].to_numpy().argmax(1)]

    speed = {}
    rj = LLM_DIR / "llm_results.json"
    if rj.exists():
        r = json.loads(rj.read_text())
        for mode, v in r["modes"].items():
            speed[f"LLM {mode}-shot"] = v["sec_per_complaint"] * 1000
        print(f"LLM: {r['model']} on {r['hardware']}\n")
    res_csv = config.REPORTS / "m1_results.csv"
    if res_csv.exists():
        for _, row in pd.read_csv(res_csv).iterrows():
            base = str(row["model"]).split("[")[0]
            for suffix in (" (as deployed)", " (no prior adj)"):
                speed[base + suffix] = row["ms_per_1k"] / 1000
    rj_bert = config.MODELS / "m1_distilbert" / "results.json"
    if rj_bert.exists():
        ms = json.loads(rj_bert.read_text()).get("ms_per_1k")
        if ms is not None:
            speed["distilbert (as deployed)"] = speed["distilbert (no prior adj)"] = ms / 1000

    rng = np.random.default_rng(config.SEED)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    rows = []
    for name, p in preds.items():
        b = [macro_f1(y[i], p[i], classes) for i in boots]
        rows.append({"model": name, "macro_f1": macro_f1(y, p, classes),
                     "ci95_low": np.percentile(b, 2.5), "ci95_high": np.percentile(b, 97.5),
                     "accuracy": float(np.mean(p == y)),
                     "ms_per_complaint": speed.get(name)})
    out = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print(f"Same {len(y)} test complaints ({len(y) // len(classes)} per product):")
    print(out.round(4).to_string(index=False))
    out.round(4).to_csv(config.REPORTS / "m1_llm_comparison.csv", index=False)

    best_llm = max((k for k in preds if k.startswith("LLM")), key=lambda k: macro_f1(y, preds[k], classes))
    print(f"\nPer-product F1 ({best_llm} vs trained models, no prior adj):")
    names = [best_llm] + [k for k in preds if k.endswith("(no prior adj)")]
    per = {name: [macro_f1(y, preds[name], [c]) for c in classes] for name in names}
    print(pd.DataFrame(per, index=classes).round(3).to_string())


if __name__ == "__main__":
    main()
