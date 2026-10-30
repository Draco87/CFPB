"""
Module 6, Step 3: a better judge, and testing it with SEEDED ERRORS.   (Kaggle, GPU T4 x2, Internet ON)
Self-contained - does not import `cfpb`.

    !python m6_03_judge_v2.py --inputs /kaggle/input/<dataset> --out-dir /kaggle/working/m6

<dataset> must contain judged_m4_drafts.jsonl (Module 5 judge output) and m5_judge_check.csv (your 20 verdicts).

Module 5 showed two weaknesses of the judge: it counted promises as rule statements, and it rejected correct
paraphrases. Judge v2 checks ONE statement at a time and first decides whether it is a rule at all.

Test 1 - against your 20 human verdicts: does v2 spot the 6 promises, and agree with you on the 14 rule statements?
Test 2 - seeded errors: take statements the judge accepted, break them on purpose, and see if v2 notices:
    number   - change a number or deadline   ("5 business days" -> "15 business days")
    flip     - reverse the meaning           ("must" -> "is not required to")
    invented - add a requirement that no excerpt contains ("... and you are entitled to $500 in compensation")
  The unbroken originals are judged too: a good judge accepts them AND rejects the broken copies.

Writes judge_v2_details.jsonl and judge_v2_summary.json
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

SYSTEM = "You are a strict, careful fact-checker of statements about US consumer-finance regulations."
TASK = """Tasks:
1. TYPE. Answer "rule" if the statement says what a law or regulation requires, allows or forbids - including a
   "we will ..." sentence that restates such a requirement (for example with a deadline or condition taken from a
   rule). Answer "other" if it is only a promise to look into something, a request for information, a greeting
   or general customer-service wording.
2. SUPPORTED (only for type "rule"). true if an excerpt says the same thing. Plain-English rewording and
   paraphrases DO count as supported. It is NOT supported if any number or deadline differs, if the meaning is
   reversed (must / must not, required / not required), or if it adds a requirement that no excerpt contains.
Reply with JSON only: {"type": "rule", "supported": true, "excerpt": 2}   (for type "other": "supported": null)"""

NUM_SWAP = {"2": "7", "3": "7", "5": "15", "7": "3", "10": "30", "15": "5", "20": "40", "21": "7", "30": "10",
            "45": "20", "50": "500", "60": "90", "90": "60", "120": "30", "500": "50"}
WORD_NUM = {"two": "seven", "three": "seven", "five": "fifteen", "ten": "thirty", "sixty": "ninety",
            "thirty": "ten", "fifteen": "five"}
FLIPS = [(r"\bis not required to\b", "is required to"), (r"\bare not required to\b", "are required to"),
         (r"\bmust not\b", "must"), (r"\bshall not\b", "shall"), (r"\bmay not\b", "may"),
         (r"\bmust\b", "is not required to"), (r"\bshall\b", "shall not"), (r"\bis required to\b", "is not required to"),
         (r"\bare required to\b", "are not required to"), (r"\bprohibited\b", "allowed"), (r"\bmay\b", "may not"),
         (r"\bwe will\b", "we will not")]
INVENTED = ", and you are also entitled to $500 in compensation"


def clip(text, n):
    w = str(text).split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def seed_errors(claim):
    out = {}
    m = re.search(r"\b(\d+)\b", claim)
    if m and m.group(1) in NUM_SWAP:
        out["number"] = claim[:m.start()] + NUM_SWAP[m.group(1)] + claim[m.end():]
    else:
        for w, r in WORD_NUM.items():
            if re.search(rf"\b{w}\b", claim, re.I):
                out["number"] = re.sub(rf"\b{w}\b", r, claim, count=1, flags=re.I)
                break
    for pat, rep in FLIPS:
        if re.search(pat, claim, re.I):
            keep_case = lambda m, rep=rep: rep[0].upper() + rep[1:] if m.group(0)[0].isupper() else rep
            out["flip"] = re.sub(pat, keep_case, claim, count=1, flags=re.I)
            break
    out["invented"] = claim.rstrip(" .") + INVENTED + "."
    return out


def parse(answer):
    a, b = answer.find("{"), answer.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        d = json.loads(answer[a:b + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or d.get("type") not in ("rule", "other"):
        return None
    return d


def split_passages(all_passages):
    """'[cite] text\\n\\n[cite] text' (from m5_judge_check.csv) -> list of {citation, text}."""
    out = []
    for block in re.split(r"\n\n(?=\[)", str(all_passages).strip()):
        m = re.match(r"\[([^\]]+)\]\s*(.*)", block, re.S)
        if m:
            out.append({"citation": m.group(1), "text": m.group(2)})
    return out


def find(inputs, patterns, what):
    """First file under `inputs` matching any pattern; a clear message listing what IS there if none does."""
    for pat in patterns:
        hits = sorted(inputs.rglob(pat))
        if hits:
            print(f"Using {hits[0]}")
            return hits[0]
    files = [str(f.relative_to(inputs)) for f in inputs.rglob("*") if f.is_file()]
    raise SystemExit(f"Could not find {what} under {inputs}.\nFiles that ARE there: {files or 'none'}\n"
                     "Add the missing file to the dataset (Kaggle: dataset page -> New Version -> upload).")


def build_items(inputs, max_originals):
    items = []
    # Test 1: human-labelled statements
    with open(find(inputs, ["m5_judge_check*.csv", "*judge_check*.csv"], "your verdicts (m5_judge_check.csv)"),
              encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f)):
            items.append({"test": "human", "row": i + 1, "statement": r["statement"],
                          "passages": split_passages(r["all_passages"]),
                          "human": r["your_verdict"].strip().lower(),
                          "v1_supported": r["judge_says_supported"].strip().lower() in ("true", "1")})
    # Test 2: seeded errors, from statements the Module 5 judge accepted
    originals = []
    judged = find(inputs, ["judged_m4_drafts.jsonl", "judged_m4_drafts*.json*", "*judged*m4_drafts.*"],
                  "the Module 5 judge output (judged_m4_drafts.jsonl)")
    for line in judged.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        for c in (j.get("judge") or {}).get("claims", []):
            if c.get("supported") and len(str(c.get("claim", "")).split()) >= 6:
                originals.append((str(c["claim"]), j["retrieved"]))
    rng = np.random.default_rng(42)
    if len(originals) > max_originals:
        originals = [originals[i] for i in sorted(rng.choice(len(originals), max_originals, replace=False))]
    for k, (claim, passages) in enumerate(originals):
        items.append({"test": "seeded", "group": k, "kind": "original", "statement": claim, "passages": passages})
        for kind, broken in seed_errors(claim).items():
            items.append({"test": "seeded", "group": k, "kind": kind, "statement": broken, "passages": passages})
    return items


def prompt_text(item):
    ex = "\n\n".join(f"[{i + 1}] ({p['citation']})\n{clip(p['text'], 260)}" for i, p in enumerate(item["passages"]))
    return f"REGULATION EXCERPTS:\n{ex}\n\nSTATEMENT:\n{item['statement']}\n\n{TASK}"


def summarise(items):
    s = {}
    hum = [x for x in items if x["test"] == "human" and x["judge"]]
    promises = [x for x in hum if x["human"] == "promise"]
    rules = [x for x in hum if x["human"] in ("yes", "no")]
    s["human"] = {
        "promises_spotted": f"{sum(x['judge']['type'] == 'other' for x in promises)}/{len(promises)}",
        "rules_called_rule": f"{sum(x['judge']['type'] == 'rule' for x in rules)}/{len(rules)}",
        "v2_agrees_on_rules": f"{sum((x['judge'].get('supported') is True) == (x['human'] == 'yes') for x in rules)}/{len(rules)}",
        "v1_agreed_on_rules": f"{sum(x['v1_supported'] == (x['human'] == 'yes') for x in rules)}/{len(rules)}",
        "v1_promises_skipped": f"0/{len(promises)}",
    }
    seeded = [x for x in items if x["test"] == "seeded" and x["judge"]]
    by = {}
    for kind in ("original", "number", "flip", "invented"):
        g = [x for x in seeded if x["kind"] == kind]
        if not g:
            continue
        if kind == "original":
            ok = sum(x["judge"]["type"] == "rule" and x["judge"].get("supported") is True for x in g)
            by[kind] = {"n": len(g), "accepted": ok, "rate": round(ok / len(g), 3)}
        else:
            caught = sum(x["judge"]["type"] == "rule" and x["judge"].get("supported") is False for x in g)
            by[kind] = {"n": len(g), "caught": caught, "rate": round(caught / len(g), 3)}
    s["seeded"] = by
    s["unparsed"] = sum(x["judge"] is None for x in items)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-originals", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true", help="no model: fake answers, to test the plumbing")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = build_items(args.inputs, args.max_originals)
    print(f"{sum(x['test'] == 'human' for x in items)} human-labelled statements, "
          f"{sum(x['test'] == 'seeded' for x in items)} seeded-test statements")

    if args.dry_run:
        answers = ['{"type": "rule", "supported": %s, "excerpt": 1}' % ("true" if i % 2 else "false")
                   for i in range(len(items))]
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto")
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
        model.eval()
        prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                            {"role": "user", "content": prompt_text(x)}],
                                           tokenize=False, add_generation_prompt=True) for x in items]
        order = np.argsort([len(p) for p in prompts], kind="stable")      # order-safe batching
        answers = [None] * len(prompts)
        with torch.no_grad():
            for s in range(0, len(order), args.bs):
                idx = order[s:s + args.bs]
                enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(model.device)
                gen = model.generate(**enc, max_new_tokens=60, do_sample=False, pad_token_id=tok.pad_token_id)
                for i, t in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                    answers[i] = t.strip()
                print(f"  {min(s + args.bs, len(order))}/{len(order)}", flush=True)

    for x, a in zip(items, answers):
        x["judge"], x["judge_raw"] = parse(a), a
    with open(args.out_dir / "judge_v2_details.jsonl", "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps({k: v for k, v in x.items() if k != "passages"}, ensure_ascii=False) + "\n")
    summary = summarise(items)
    summary["judge_model"] = args.model
    (args.out_dir / "judge_v2_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Test 1 - your 20 verdicts ===")
    for k, v in summary["human"].items():
        print(f"  {k}: {v}")
    print("=== Test 2 - seeded errors ===")
    for k, v in summary["seeded"].items():
        print(f"  {k}: {v}")
    print(f"unparsed answers: {summary['unparsed']}\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
