"""
Module 5, Step 3: LLM-as-judge - is each draft reply faithful to the rules it was given?

Runs on Kaggle (GPU T4 x2, Internet ON). Self-contained - does not import `cfpb`.
    !python m5_02_judge.py --inputs /kaggle/input/<dataset with the .jsonl draft files> --out-dir /kaggle/working/m5

The judge (Qwen2.5-7B-Instruct, larger than the 3B model that wrote the drafts) reads, for every draft:
the retrieved excerpts, the complaint and the draft. It then
  1. lists every statement the draft makes about what a rule requires or allows,
  2. marks each one SUPPORTED only if an excerpt clearly says it,
  3. rates 1-5 how well the draft addresses the customer's actual problem,
  4. flags drafts that contradict facts in the complaint.

Faithfulness = supported statements / all statements (the RAGAS definition, implemented directly so it runs
on an open model with no paid API).
Writes judged_<name>.jsonl per input file and m5_judge_summary.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "You are a strict, careful fact-checker of draft replies to US consumer-finance complaints."
TASK = """Tasks:
1. List every statement in the DRAFT REPLY about what a rule, regulation or law requires, allows or forbids
   (deadlines, obligations, rights, liability, fees). Ignore greetings, apologies, requests for information and
   general promises to look into the matter. If there are none, return an empty list.
2. For each statement decide whether the EXCERPTS clearly support it. Mark "supported": true only if an excerpt
   says the same thing. Wrong numbers, wrong deadlines, rules applied to the wrong situation, or statements not
   found in any excerpt are "supported": false. Give the number of the supporting excerpt, or null.
3. Rate 1-5 how well the draft addresses the customer's actual problem (5 = directly and completely).
4. Does the draft contradict any fact the customer states in the complaint? true/false.
Reply with JSON only, exactly in this shape:
{"claims": [{"claim": "...", "supported": true, "excerpt": 1}], "addresses_problem": 4, "contradicts_complaint": false}"""


def clip(text, n):
    w = str(text).split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def build_prompt(tok, rec):
    excerpts = "\n\n".join(f"[{i + 1}] ({p['citation']})\n{clip(p['text'], 260)}" for i, p in enumerate(rec["retrieved"]))
    user = (f"REGULATION EXCERPTS:\n{excerpts}\n\nCUSTOMER COMPLAINT:\n{clip(rec['complaint'], 250)}\n\n"
            f"DRAFT REPLY:\n{clip(rec['draft'], 400)}\n\n{TASK}")
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def parse(answer):
    a, b = answer.find("{"), answer.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        d = json.loads(answer[a:b + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or not isinstance(d.get("claims", []), list):
        return None
    return d


@torch.no_grad()
def generate_all(model, tok, prompts, bs, max_new_tokens):
    """Order-safe batched generation: answer i belongs to prompt i."""
    order = np.argsort([len(p) for p in prompts], kind="stable")
    out = [None] * len(prompts)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, t in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out[i] = t.strip()
        print(f"  {min(s + bs, len(order))}/{len(order)}", flush=True)
    return out


def summarise(judged):
    ok = [j for j in judged if j["judge"] is not None]
    claims = [c for j in ok for c in j["judge"].get("claims", [])]
    sup = [bool(c.get("supported")) for c in claims]
    per_draft = [np.mean([bool(c.get("supported")) for c in j["judge"]["claims"]])
                 for j in ok if j["judge"].get("claims")]
    rel = [j["judge"].get("addresses_problem") for j in ok if isinstance(j["judge"].get("addresses_problem"), (int, float))]
    return {
        "drafts": len(judged), "judged_ok": len(ok),
        "statements": len(claims), "supported": int(sum(sup)),
        "faithfulness": float(np.mean(sup)) if sup else None,
        "faithfulness_mean_per_draft": float(np.mean(per_draft)) if per_draft else None,
        "drafts_with_unsupported_statement": int(sum(any(not c.get("supported") for c in j["judge"].get("claims", []))
                                                     for j in ok)),
        "drafts_without_rule_statements": int(sum(not j["judge"].get("claims") for j in ok)),
        "addresses_problem_mean": float(np.mean(rel)) if rel else None,
        "contradicts_complaint": int(sum(bool(j["judge"].get("contradicts_complaint")) for j in ok)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True, help="folder with one or more draft .jsonl files")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.inputs.rglob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files under {args.inputs}")
    print("Draft files:", [f.name for f in files])

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Judge: {args.model} on {gpu} x{torch.cuda.device_count()}")

    summary = {"judge_model": args.model, "files": {}}
    for f in files:
        recs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"\n{f.name}: {len(recs)} drafts")
        t0 = time.time()
        answers = generate_all(model, tok, [build_prompt(tok, r) for r in recs], args.bs, 700)
        judged = []
        for r, a in zip(recs, answers):
            judged.append({"complaint_id": r.get("complaint_id"), "product": r.get("product"), "issue": r.get("issue"),
                           "settings": r.get("settings"), "draft": r["draft"],
                           "retrieved": [{"citation": p["citation"], "text": p["text"]} for p in r["retrieved"]],
                           "judge": parse(a), "judge_raw": a})
        s = summarise(judged)
        s["minutes"] = round((time.time() - t0) / 60, 1)
        summary["files"][f.stem] = s
        print("  ", s)
        with open(args.out_dir / f"judged_{f.stem}.jsonl", "w", encoding="utf-8") as fh:
            for j in judged:
                fh.write(json.dumps(j, ensure_ascii=False) + "\n")
        bad = sum(j["judge"] is None for j in judged)
        if bad:
            print(f"  WARNING: {bad} judge answers could not be parsed as JSON (kept in judge_raw).")

    (args.out_dir / "m5_judge_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved to", args.out_dir)


if __name__ == "__main__":
    main()
