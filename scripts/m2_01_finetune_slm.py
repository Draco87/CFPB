"""
Module 2: teach a SMALL LLM (Qwen2.5-1.5B-Instruct) to read a complaint and return
structured JSON:  {"product": "...", "sub_product": "...", "issue": "..."}

Runs on Kaggle (GPU T4, Internet ON). Self-contained - does not import `cfpb`.
    Cell 1:  !pip install -q -U peft
    Cell 2:  !python m2_01_finetune_slm.py --data-dir /kaggle/input/<processed> --out-dir /kaggle/working/m2

What it does (all logged, ~2 hours in total):
  1. BEFORE - the base 1.5B model, untrained, picks the product from a menu
              (same style as Module 1's 7B test).
  2. TRAIN  - LoRA fine-tuning on ~18k complaints from 2023-2024
              (recent years = current CFPB category names). Only small add-on
              matrices are trained (~1% of the weights), so it fits a free T4.
  3. AFTER  - the fine-tuned model answers in JSON on the SAME 900 test complaints
              (100 per product) used in Module 1.

Writes to --out-dir: base_preds.parquet, ft_preds.parquet, adapter/ (the trained add-on), m2_results.json
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # one GPU: simpler and more reliable for LoRA training

import argparse
import inspect
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

SEED = 42
CATEGORIES = {   # identical to Module 1's LLM test, for the BEFORE step
    "Bank account": "checking or savings accounts, deposits, overdrafts, account fees, opening or closing an account",
    "Credit / prepaid card": "credit cards, store cards, prepaid cards, card billing disputes, card fees, rewards",
    "Credit reporting": "credit reports and scores, wrong information on a credit report, credit bureau "
                        "investigations, identity theft on a report, credit repair",
    "Debt collection": "a collector or company trying to collect a debt, collection calls or letters, "
                       "debts not owed, collection threats",
    "Money transfer / virtual currency": "money transfers, payment apps, wire transfers, money orders, "
                                         "cryptocurrency, check cashing",
    "Mortgage": "home loans, mortgage servicing, escrow, foreclosure, home equity loans, refinancing",
    "Payday / personal loan": "payday loans, title loans, installment or personal loans, cash advances",
    "Student loan": "federal or private student loans, loan servicers, repayment plans, forgiveness",
    "Vehicle loan": "auto loans or leases, vehicle financing, repossession",
}
CLASSES = list(CATEGORIES)
MENU = "Categories:\n" + "\n".join(f"{i}. {c} - {d}" for i, (c, d) in enumerate(CATEGORIES.items(), 1))
BASE_SYSTEM = ("You route consumer complaints about financial products to the right team. "
               "Read the complaint and choose exactly one product category. Reply with the category number only.")
FT_SYSTEM = "Extract the routing fields from this consumer financial complaint. Reply with JSON only."
FIELDS = ["product", "sub_product", "issue"]


# ---------- helpers
def clip(text, n_words):
    w = str(text).split()
    return " ".join(w[:n_words]) + (" ..." if len(w) > n_words else "")


def norm(v):
    return "None" if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "" else str(v)


def base_prompt(tok, text, n_words):
    msgs = [{"role": "system", "content": BASE_SYSTEM},
            {"role": "user", "content": f'{MENU}\n\nComplaint:\n"""{clip(text, n_words)}"""\n\nAnswer with the number only.'}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def ft_prompt(tok, text, n_words):
    msgs = [{"role": "system", "content": FT_SYSTEM}, {"role": "user", "content": clip(text, n_words)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def target_json(row):
    return json.dumps({"product": row["product"], "sub_product": norm(row["sub_product"]),
                       "issue": norm(row["issue"])}, ensure_ascii=False)


def parse_number(answer):
    m = re.search(r"\d+", answer)
    if m and 1 <= int(m.group()) <= len(CLASSES):
        return CLASSES[int(m.group()) - 1]
    return "INVALID"


def parse_json(answer):
    a, b = answer.find("{"), answer.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        d = json.loads(answer[a:b + 1])
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


@torch.no_grad()
def generate_all(model, tok, prompts, bs, max_new_tokens):
    """Order-safe batched generation: answer i belongs to prompt i."""
    tok.padding_side = "left"
    order = np.argsort([len(p) for p in prompts], kind="stable")
    answers = [None] * len(prompts)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for i, t in zip(idx, tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            answers[i] = t.strip()
        if s // bs % 20 == 0:
            print(f"  {min(s + bs, len(order))}/{len(order)}", flush=True)
    return answers


def build_features(tok, df, n_words, max_len):
    """Prompt tokens are masked (-100) so the model only learns to write the JSON answer."""
    end = "<|im_end|>"
    feats, dropped = [], 0
    for _, row in df.iterrows():
        p_ids = tok(ft_prompt(tok, row["text"], n_words), add_special_tokens=False)["input_ids"]
        a_ids = tok(target_json(row) + end, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(a_ids) > max_len:
            dropped += 1
            continue
        feats.append({"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids})
    return feats, dropped


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id):
    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            lab[i, :k] = torch.tensor(b["labels"])
            att[i, :k] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return collate


def load_model(name):
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)


# ---------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--per-class", type=int, default=100, help="test complaints per product")
    ap.add_argument("--train-per-product", type=int, default=2000)
    ap.add_argument("--train-from", default="2023-01-01")
    ap.add_argument("--words", type=int, default=250)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--skip-base", action="store_true")
    args = ap.parse_args()
    set_seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from peft import LoraConfig, get_peft_model    # imported here so a missing install fails with a clear message

    # Same 900 test complaints as Module 1 (same seed, same sampling)
    cols = ["complaint_id", "product", "sub_product", "issue", "text"]
    test = pd.read_parquet(args.data_dir / "m1_test.parquet", columns=cols)
    sample = (test.sample(frac=1, random_state=SEED).groupby("product").head(args.per_class)
                  .sort_values("complaint_id").reset_index(drop=True))
    train = pd.read_parquet(args.data_dir / "m1_train.parquet", columns=cols + ["date"])
    train = train[train["date"] >= args.train_from]
    train = train.sample(frac=1, random_state=SEED).groupby("product").head(args.train_per_product)
    print(f"Test: {len(sample)} complaints | Train: {len(train):,} complaints from {args.train_from}")
    print("  train per product:", train["product"].value_counts().to_dict())

    coverage = {}
    for f in ["sub_product", "issue"]:
        seen = set(train[f].map(norm))
        coverage[f] = float(sample[f].map(norm).isin(seen).mean())
        print(f"  {f}: {train[f].map(norm).nunique()} distinct values in train; "
              f"{coverage[f]:.1%} of test answers appear in train (= best possible accuracy)")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model).to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded {args.model} on {gpu}")
    results = {"model": args.model, "hardware": gpu, "n_test": len(sample),
               "n_train": len(train), "coverage": coverage}
    y = sample["product"].to_numpy()

    # ---- 1. BEFORE: untrained 1.5B, product only
    if not args.skip_base:
        print("\n[1/3] BEFORE fine-tuning (zero-shot, product only)")
        model.eval()
        t0 = time.time()
        ans = generate_all(model, tok, [base_prompt(tok, t, 300) for t in sample["text"]], 16, 4)
        secs = time.time() - t0
        pred = [parse_number(a) for a in ans]
        results["base"] = {"macro_f1": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
                           "accuracy": accuracy_score(y, pred),
                           "invalid_rate": float(np.mean([p == "INVALID" for p in pred])),
                           "sec_per_complaint": secs / len(sample)}
        print("  ", {k: round(v, 4) for k, v in results["base"].items()}, "| sample answers:", ans[:6])
        pd.DataFrame({"complaint_id": sample["complaint_id"], "y_true": y, "y_pred": pred, "raw_answer": ans}
                     ).to_parquet(args.out_dir / "base_preds.parquet", index=False)

    # ---- 2. TRAIN with LoRA
    print("\n[2/3] Fine-tuning with LoRA")
    feats, dropped = build_features(tok, train, args.words, args.max_len)
    print(f"  {len(feats):,} training examples ({dropped} dropped as too long)")
    print("  example target:", target_json(train.iloc[0]))

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():                       # trainable LoRA weights in fp32 (needed for fp16 training)
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    steps = math.ceil(len(feats) / (args.bs * args.grad_accum)) * args.epochs
    base = dict(output_dir=str(args.out_dir / "checkpoints"), num_train_epochs=args.epochs,
                per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.grad_accum,
                learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=int(0.03 * steps),
                weight_decay=0.0, fp16=torch.cuda.is_available(), logging_steps=25, save_strategy="no",
                report_to="none", remove_unused_columns=False, seed=SEED)
    params = inspect.signature(TrainingArguments.__init__).parameters
    grouping = ({"train_sampling_strategy": "group_by_length"} if "train_sampling_strategy" in params
                else {"group_by_length": True} if "group_by_length" in params else {})
    try:
        targs = TrainingArguments(**base, **grouping)
    except (TypeError, ValueError):
        targs = TrainingArguments(**base)
    trainer = Trainer(model=model, args=targs, train_dataset=ListDataset(feats),
                      data_collator=make_collate(tok.pad_token_id))
    t0 = time.time()
    trainer.train()
    results["train_minutes"] = round((time.time() - t0) / 60, 1)
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    results["train_loss_first_last"] = [losses[0], losses[-1]] if losses else None
    print(f"  trained in {results['train_minutes']} min; loss {losses[0]:.3f} -> {losses[-1]:.3f}" if losses else "")
    if losses and not np.isfinite(losses[-1]):
        print("  WARNING: loss is NaN/inf - training failed. Paste this log to Claude.")
    model.save_pretrained(str(args.out_dir / "adapter"))
    tok.save_pretrained(str(args.out_dir / "adapter"))

    # ---- 3. AFTER: fine-tuned model, JSON output
    print("\n[3/3] AFTER fine-tuning (JSON: product, sub_product, issue)")
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    t0 = time.time()
    ans = generate_all(model, tok, [ft_prompt(tok, t, args.words) for t in sample["text"]], 16, 96)
    secs = time.time() - t0
    parsed = [parse_json(a) or {} for a in ans]
    pred = {f: [str(d.get(f, "INVALID")) for d in parsed] for f in FIELDS}
    true = {f: sample[f].map(norm).to_numpy() if f != "product" else y for f in FIELDS}
    exact = np.mean([all(pred[f][i] == true[f][i] for f in FIELDS) for i in range(len(sample))])
    results["finetuned"] = {
        "json_valid_rate": float(np.mean([bool(d) for d in parsed])),
        "product_macro_f1": f1_score(y, pred["product"], labels=CLASSES, average="macro", zero_division=0),
        "product_accuracy": accuracy_score(y, pred["product"]),
        "sub_product_accuracy": accuracy_score(true["sub_product"], pred["sub_product"]),
        "issue_accuracy": accuracy_score(true["issue"], pred["issue"]),
        "all_three_correct": float(exact),
        "sec_per_complaint": secs / len(sample)}
    print("  ", {k: round(v, 4) for k, v in results["finetuned"].items()})
    print("  sample answers:")
    for a in ans[:3]:
        print("    ", a)
    pd.DataFrame({"complaint_id": sample["complaint_id"], "y_true": y, "y_pred": pred["product"],
                  "true_sub_product": true["sub_product"], "pred_sub_product": pred["sub_product"],
                  "true_issue": true["issue"], "pred_issue": pred["issue"], "raw_answer": ans}
                 ).to_parquet(args.out_dir / "ft_preds.parquet", index=False)

    (args.out_dir / "m2_results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to", args.out_dir)


if __name__ == "__main__":
    main()
