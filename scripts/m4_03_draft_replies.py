"""
Module 4, Step 3: draft cited replies to real complaints with RAG.

Needs Ollama running on your machine with a model pulled, e.g.:
    ollama pull qwen2.5:3b          (fits a 4 GB GPU; qwen2.5:7b needs ~6 GB)

Try one complaint of your own:
    uv run python scripts/m4_03_draft_replies.py --text "I was charged twice for ..." --product "Bank account"

Draft replies for a sample of test complaints (3 per product) and write a report:
    uv run python scripts/m4_03_draft_replies.py
Writes reports/m4_drafts.jsonl (used by Module 5) and reports/m4_drafts.html (open in a browser)
"""
import argparse
import html
import json

import pandas as pd

from cfpb import config
from cfpb.rag import DEFAULT_LLM, Retriever, answer_complaint, clip, ollama_ok
from cfpb.report import build_html, pct


def show(res):
    print("\nSEARCH QUERY:", res["query"])
    print("\nRETRIEVED:")
    for r in res["passages"].itertuples():
        print(f"  {r.rank}. [{r.citation}] {r.title[:80]}")
    print("\n" + res["draft"])
    c = res["check"]
    print(f"\nCitation check: {c['n_valid']}/{c['n_cited']} citations point to retrieved passages"
          + (f" | NOT retrieved: {c['invalid']}" if c["invalid"] else "") + f" | {res['seconds']}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="draft a reply to this complaint text")
    ap.add_argument("--product", default=None)
    ap.add_argument("--issue", default="")
    ap.add_argument("--n-per-product", type=int, default=3)
    ap.add_argument("--model", default=DEFAULT_LLM)
    ap.add_argument("--k", type=int, default=5, help="passages given to the LLM")
    ap.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-filter", action="store_true", help="search all regulations, not just the product's")
    ap.add_argument("--no-rewrite", action="store_true",
                    help="search with the raw complaint instead of an LLM-rewritten question")
    args = ap.parse_args()

    ok, msg = ollama_ok(args.model)
    if not ok:
        raise SystemExit(msg)
    print("Loading search indexes and models ...")
    retriever = Retriever(rerank=not args.no_rerank)
    opts = dict(model=args.model, k=args.k, mode=args.mode, filter_product=not args.no_filter, rewrite=not args.no_rewrite)

    if args.text:
        show(answer_complaint(retriever, args.text, args.product, args.issue, **opts))
        return

    test = pd.read_parquet(config.PROCESSED / "m1_test.parquet", columns=["complaint_id", "product", "issue", "text", "n_chars"])
    sample = (test[test["n_chars"].between(300, 4000)].sample(frac=1, random_state=config.SEED)
                  .groupby("product").head(args.n_per_product).sort_values("product"))
    records = []
    for i, row in enumerate(sample.itertuples(), 1):
        print(f"\n===== {i}/{len(sample)}  {row.product} / {row.issue}")
        res = answer_complaint(retriever, row.text, row.product, row.issue, **opts)
        show(res)
        records.append({"complaint_id": int(row.complaint_id), "product": row.product, "issue": row.issue,
                        "complaint": clip(row.text, 300), "query": res["query"],
                        "retrieved": res["passages"][["citation", "title", "text"]].to_dict("records"),
                        "draft": res["draft"], "check": res["check"], "seconds": res["seconds"], "settings": opts})

    out = config.REPORTS / "m4_drafts.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- readable report
    n = len(records)
    with_cite = sum(r["check"]["has_citation"] for r in records)
    cited = sum(r["check"]["n_cited"] for r in records)
    valid = sum(r["check"]["n_valid"] for r in records)
    fmt = sum(r["check"]["format_ok"] for r in records)
    secs = sum(r["seconds"] for r in records) / max(1, n)
    summary = [f"{n} test complaints ({args.n_per_product} per product). Model: <b>{args.model}</b>; search: "
               f"<b>{args.mode}</b>{'' if args.no_rerank else ' + reranking'}"
               f"{'' if args.no_filter else ', limited to the regulations for the product'}.",
               f"<b>{with_cite}/{n}</b> drafts cite at least one rule; <b>{valid}/{cited}</b> citations point to a "
               f"passage that was actually retrieved ({pct(valid / max(1, cited))}). Answers followed the required "
               f"format in {fmt}/{n} cases. Average time per draft: {secs:.0f} seconds.",
               "These are automatic checks only. Whether the cited rules are the <i>right</i> ones, and whether "
               "the reply is faithful to them, is measured in Module 5."]
    sections = [("Summary", summary, "", [])]
    for r in records:
        c = r["check"]
        cites = "".join(f"<li>[{html.escape(p['citation'])}] {html.escape(p['title'][:90])}</li>" for p in r["retrieved"])
        block = (f"<p><b>Complaint</b> (first 300 words):</p><blockquote>{html.escape(r['complaint'])}</blockquote>"
                 f"<p><b>Search query:</b> {html.escape(r['query'])}</p><p><b>Retrieved rules:</b></p><ol>{cites}</ol>"
                 f"<p><b>Draft:</b></p><pre style='white-space:pre-wrap;background:#f3f2ef;padding:12px'>"
                 f"{html.escape(r['draft'])}</pre>")
        bullet = [f"Citations: {c['n_valid']}/{c['n_cited']} point to retrieved passages"
                  + (f" - NOT retrieved: {html.escape(', '.join(c['invalid']))}" if c["invalid"] else "")
                  + f". Time: {r['seconds']}s."]
        sections.append((html.escape(f"{r['product']} - {r['issue']}"), [], block, bullet))
    rep = config.REPORTS / "m4_drafts.html"
    rep.write_text(build_html("RAG draft replies", "Regulation-grounded draft replies for agents to review.", sections),
                   encoding="utf-8")
    print(f"\n{with_cite}/{n} drafts with citations | {valid}/{cited} citations valid | ~{secs:.0f}s per draft")
    print(f"Saved {out} and {rep}")


if __name__ == "__main__":
    main()
