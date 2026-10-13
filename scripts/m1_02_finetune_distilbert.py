"""
Module 1, Step 3: fine-tune DistilBERT on the same frozen splits as the baseline.

Needs an NVIDIA GPU (CPU only for a smoke test with --limit 1000).

Local (inside the project):
    uv run python scripts/m1_02_finetune_distilbert.py

Kaggle (free GPU) - this file is self-contained, it does not import `cfpb`:
    !python m1_02_finetune_distilbert.py --data-dir /kaggle/input/<your-dataset> \
            --out-dir /kaggle/working/m1_distilbert

Writes to --out-dir:  best/ (model + tokenizer), test_preds.parquet, val/test reports, results.json
and, when run inside the project, adds a row to reports/m1_results.csv.
"""
import argparse
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEED = 42


# --- same prior-shift helpers as src/cfpb/metrics.py (duplicated so Kaggle needs only this file)
def class_prior(labels, classes):
    return pd.Series(labels).value_counts(normalize=True).reindex(classes, fill_value=1e-6).to_numpy()


def prior_adjust(proba, train_prior, target_prior, alpha):
    return proba * (np.asarray(target_prior) / np.asarray(train_prior)) ** alpha


def tune_alpha(proba, y_true, classes, train_prior, target_prior):
    scores = {a: f1_score(y_true, classes[prior_adjust(proba, train_prior, target_prior, a).argmax(1)],
                          average="macro") for a in ALPHAS}
    return max(scores, key=scores.get), scores
# ---


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "models" / "m1_distilbert")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only N rows per split")
    args = ap.parse_args()
    set_seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = (pd.read_parquet(args.data_dir / f"m1_{s}.parquet") for s in ("train", "val", "test"))
    if args.limit:
        train, val, test = (d.sample(min(args.limit, len(d)), random_state=SEED) for d in (train, val, test))
    meta_file = args.data_dir / "m1_splits_meta.json"
    mode = json.loads(meta_file.read_text())["train_mode"] if meta_file.exists() else "unknown"
    name = f"distilbert[{mode}]"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"{name} on {gpu}: train {len(train):,} | val {len(val):,} | test {len(test):,}")

    classes = np.array(sorted(train["product"].unique()))
    label2id = {c: i for i, c in enumerate(classes)}
    tok = AutoTokenizer.from_pretrained(args.model)

    def to_ds(d):
        ds = Dataset.from_pandas(pd.DataFrame({"text": d["text"].to_numpy(),
                                               "label": d["product"].map(label2id).to_numpy()}))
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                      batched=True, remove_columns=["text"])

    train_ds, val_ds, test_ds = to_ds(train), to_ds(val), to_ds(test)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(classes),
        id2label={i: c for c, i in label2id.items()}, label2id=label2id)

    def compute_metrics(p):
        pred = p.predictions.argmax(1)
        return {"macro_f1": f1_score(p.label_ids, pred, average="macro"),
                "accuracy": accuracy_score(p.label_ids, pred)}

    n_dev = max(1, torch.cuda.device_count())
    total_steps = math.ceil(len(train_ds) / (args.bs * n_dev)) * args.epochs
    base = dict(
        output_dir=str(args.out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=int(0.06 * total_steps),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        report_to="none",
        seed=SEED,
    )
    # Batch complaints of similar length together (much less padding -> faster).
    # transformers v4: group_by_length=True | v5: train_sampling_strategy="group_by_length"
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "train_sampling_strategy" in params:
        grouping = {"train_sampling_strategy": "group_by_length"}
    elif "group_by_length" in params:
        grouping = {"group_by_length": True}
    else:
        grouping = {}
    try:
        targs = TrainingArguments(**base, **grouping)
    except (TypeError, ValueError) as e:
        print(f"Length grouping not available ({e}) - training without it (slower, same result).")
        targs = TrainingArguments(**base)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)

    t0 = time.time()
    trainer.train()
    train_min = (time.time() - t0) / 60
    trainer.save_model(str(args.out_dir / "best"))
    tok.save_pretrained(str(args.out_dir / "best"))

    # Prior-shift alpha tuned on VAL (train is capped/balanced, val/test have the real mix)
    train_prior = class_prior(train["product"], classes)
    target_prior = class_prior(val["product"], classes)
    val_proba = softmax(trainer.predict(val_ds).predictions)
    alpha, scores = tune_alpha(val_proba, val["product"].to_numpy(), classes, train_prior, target_prior)
    print("VAL macro-F1 by alpha:", {a: round(s, 4) for a, s in scores.items()}, f"-> alpha={alpha}")
    val_pred = classes[prior_adjust(val_proba, train_prior, target_prior, alpha).argmax(1)]
    (args.out_dir / "val_report.txt").write_text(
        classification_report(val["product"], val_pred, digits=3, zero_division=0))

    t0 = time.time()
    test_proba = softmax(trainer.predict(test_ds).predictions)
    ms_per_1k = (time.time() - t0) / len(test) * 1e6
    pred = classes[prior_adjust(test_proba, train_prior, target_prior, alpha).argmax(1)]

    row = {"model": name, "train_rows": len(train),
           "accuracy": accuracy_score(test["product"], pred),
           "macro_f1": f1_score(test["product"], pred, average="macro"),
           "weighted_f1": f1_score(test["product"], pred, average="weighted"),
           "prior_alpha": alpha, "ms_per_1k": ms_per_1k, "hardware": gpu,
           "train_minutes": round(train_min, 1), "cost_per_1k_usd": 0.0}
    print("TEST", {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    report = classification_report(test["product"], pred, digits=3, zero_division=0)
    (args.out_dir / "test_report.txt").write_text(report)
    print(report)

    preds = pd.DataFrame({"complaint_id": test["complaint_id"].to_numpy(),
                          "y_true": test["product"].to_numpy(), "y_pred": pred})
    preds = pd.concat([preds, pd.DataFrame(test_proba, columns=[f"p_{c}" for c in classes])], axis=1)
    preds.to_parquet(args.out_dir / "test_preds.parquet", index=False)
    (args.out_dir / "results.json").write_text(json.dumps(row, indent=2))

    results_csv = ROOT / "reports" / "m1_results.csv"
    if results_csv.parent.is_dir() and not args.limit:
        old = pd.read_csv(results_csv) if results_csv.exists() else pd.DataFrame(columns=["model"])
        old = old[old["model"] != name]
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(results_csv, index=False)
        print("results ->", results_csv)


if __name__ == "__main__":
    main()