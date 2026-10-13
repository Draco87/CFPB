"""
Module 1, Step 5: can an off-the-shelf LLM sort complaints without any training?

Runs on Kaggle (GPU T4 x2, Internet ON). Self-contained - does not import `cfpb`.
    !python m1_06_llm_classify.py --data-dir /kaggle/input/<processed> --out-dir /kaggle/working/m1_llm

What it does
- Takes 100 test complaints per product (900 total), the same ones every time (fixed seed).
- Asks an open instruction-tuned LLM (default Qwen2.5-7B-Instruct) to pick the product:
    zero-shot: category list with short descriptions only
    few-shot : same, plus one short real example per category (from the TRAIN split)
- Saves predictions per mode + a results summary.

Writes to --out-dir: llm_preds_zero.parquet, llm_preds_few.parquet, llm_results.json
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42

# Same 9 labels as the other models, in the same (alphabetical) order.
CATEGORIES = {
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

SYSTEM = ("You route consumer complaints about financial products to the right team. "
          "Read the complaint and choose exactly one product category. "
          "Reply with the category number only.")
MENU = "Categories:\n" + "\n".join(f"{i}. {c} - {d}" for i, (c, d) in enumerate(CATEGORIES.items(), 1))


def clip(text, n_words):
    w = str(text).split()
    return " ".join(w[:n_words]) + (" ..." if len(w) > n_words else "")


def user_msg(text, n_words):
    return f'{MENU}\n\nComplaint:\n"""{clip(text, n_words)}"""\n\nAnswer with the number only.'


def build_messages(text, shots, n_words):
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex_text, ex_label in shots:
        msgs.append({"role": "user", "content": user_msg(ex_text, 80)})
        msgs.append({"role": "assistant", "content": str(CLASSES.index(ex_label) + 1)})
    msgs.append({"role": "user", "content": user_msg(text, n_words)})
    return msgs


def parse(answer):
    """'3' / '3.' / 'Category 3' -> label; a category name -> label; otherwise INVALID."""
    m = re.search(r"\d+", answer)
    if m and 1 <= int(m.group()) <= len(CLASSES):
        return CLASSES[int(m.group()) - 1]
    low = answer.lower()
    for c in CLASSES:
        if c.lower() in low:
            return c
    return "INVALID"


@torch.no_grad()
def generate_all(model, tok, prompts, bs, max_new_tokens=4):
    """Order-safe batched generation: answer i belongs to prompt i.
    Prompts are run shortest-first (less padding) and put back in the original order."""
    order = np.argsort([len(p) for p in prompts], kind="stable")
    answers = [None] * len(prompts)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            answers[i] = t.strip()
        if s // bs % 25 == 0:
            print(f"  {min(s + bs, len(order))}/{len(order)}", flush=True)
    return answers


def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(device_map="auto")
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, **kw)
    except TypeError:                                   # older transformers
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, **kw)
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--modes", default="zero,few")
    ap.add_argument("--words", type=int, default=300, help="complaint words kept in the prompt")
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    test = pd.read_parquet(args.data_dir / "m1_test.parquet", columns=["complaint_id", "product", "text"])
    sample = (test.sample(frac=1, random_state=SEED).groupby("product").head(args.per_class)
                  .sort_values("complaint_id").reset_index(drop=True))
    print(f"Evaluating on {len(sample)} test complaints:", sample["product"].value_counts().to_dict())

    # Few-shot examples: one short-ish complaint per category from TRAIN (never from test)
    train = pd.read_parquet(args.data_dir / "m1_train.parquet", columns=["product", "text", "n_chars"])
    pool = train[(train["n_chars"] > 200) & (train["n_chars"] < 900)]
    shots = [(pool[pool["product"] == c].sample(1, random_state=SEED)["text"].iloc[0], c) for c in CLASSES]
    shots = [shots[i] for i in rng.permutation(len(shots))]    # avoid always ending on the same label

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {gpu} x{torch.cuda.device_count()} ...")
    t0 = time.time()
    model, tok = load_model(args.model)
    print(f"  loaded in {time.time() - t0:.0f}s")

    summary = {"model": args.model, "hardware": f"{gpu} x{torch.cuda.device_count()}",
               "n": len(sample), "per_class": args.per_class, "modes": {}}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        use_shots = shots if mode == "few" else []
        prompts = [tok.apply_chat_template(build_messages(t, use_shots, args.words),
                                           tokenize=False, add_generation_prompt=True)
                   for t in sample["text"]]
        n_tok = int(np.mean([len(tok(p)["input_ids"]) for p in prompts[:50]]))
        print(f"\n[{mode}-shot] ~{n_tok} prompt tokens per complaint")
        t0 = time.time()
        answers = generate_all(model, tok, prompts, args.bs)
        secs = time.time() - t0
        pred = [parse(a) for a in answers]

        y = sample["product"].to_numpy()
        res = {"macro_f1": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
               "accuracy": accuracy_score(y, pred),
               "invalid_rate": float(np.mean([p == "INVALID" for p in pred])),
               "sec_per_complaint": secs / len(sample),
               "prompt_tokens": n_tok}
        summary["modes"][mode] = res
        print(f"[{mode}-shot]", {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
        print(classification_report(y, pred, labels=CLASSES, digits=3, zero_division=0))
        print("  sample answers:", answers[:8])
        if res["invalid_rate"] > 0.1:
            print("  WARNING: many answers could not be parsed - check the sample answers above.")

        pd.DataFrame({"complaint_id": sample["complaint_id"], "y_true": y, "y_pred": pred,
                      "raw_answer": answers}).to_parquet(args.out_dir / f"llm_preds_{mode}.parquet", index=False)

    (args.out_dir / "llm_results.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved to", args.out_dir)


if __name__ == "__main__":
    main()
