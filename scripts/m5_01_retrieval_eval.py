"""
Module 5, Step 1: does the search find the right rules?   (laptop, ~10 minutes)

    uv run python scripts/m5_01_retrieval_eval.py

Uses the answer key eval/golden_retrieval.json - 34 realistic complaints, each paired with the rule section(s)
that clearly govern it (e.g. "collector threatened arrest" -> 12 CFR 1006.18 / 15 U.S.C. 1692e).
For several search setups it measures:
  Found in top 5  - share of complaints where a correct section is among the 5 passages the LLM is given
  Recall@5        - share of all correct sections that appear in those 5
  MRR@10          - how high the first correct section ranks (1 = 1st place, 0.5 = 2nd, 0.33 = 3rd ... 0 = not in top 10)
Official interpretations of a section count as that section.

Writes reports/m5_retrieval.csv, reports/m5_retrieval_details.csv; caches rewritten questions in eval/rewrites_cache.json
"""
import json

import pandas as pd

from cfpb import config
from cfpb.rag import DEFAULT_LLM, Retriever, base_citation, build_query, ollama_ok, rewrite_query

GOLDEN = config.ROOT / "eval" / "golden_retrieval.json"
CACHE = config.ROOT / "eval" / "rewrites_cache.json"
FINAL = "Hybrid + rerank + rewrite (final)"
MULTI = "Final + original complaint (multi-query)"
CONFIGS = [  # name, search options, use the LLM-rewritten question?
    ("Keywords only (BM25)", dict(mode="bm25", rerank=False), False),
    ("Meaning only (embeddings)", dict(mode="dense", rerank=False), False),
    ("Hybrid (keywords + meaning)", dict(mode="hybrid", rerank=False), False),
    ("Hybrid + rerank", dict(mode="hybrid", rerank=True), False),
    (FINAL, dict(mode="hybrid", rerank=True), True),
    ("Final, without product filter", dict(mode="hybrid", rerank=True, filter_product=False), True),
    ("Final, with model forms (appendices)", dict(mode="hybrid", rerank=True, include_appendices=True), True),
    (MULTI, dict(mode="hybrid", rerank=True), "multi"),
]


def rewrites(scenarios, model):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    key = f"{model}"
    cache.setdefault(key, {})
    for s in scenarios:
        sid = str(s["id"])
        if sid not in cache[key]:
            cache[key][sid] = rewrite_query(s["complaint"], s["product"], s["issue"], model=model)
            print(f"  rewrote {sid}: {cache[key][sid]}")
    CACHE.write_text(json.dumps(cache, indent=2))
    return cache[key]


def main():
    scenarios = json.loads(GOLDEN.read_text())["scenarios"]
    print(f"{len(scenarios)} answer-key complaints")
    ok, msg = ollama_ok(DEFAULT_LLM)
    rw = None
    if ok:
        print(f"Rewriting complaints into questions with {DEFAULT_LLM} (cached after the first run) ...")
        rw = rewrites(scenarios, DEFAULT_LLM)
    else:
        print(f"Skipping the rewrite setups: {msg}")

    retriever = Retriever(rerank=True)
    rows, details = [], []
    for name, opts, use_rw in CONFIGS:
        if use_rw and rw is None:
            continue
        hits = recall = rr = 0.0
        for s in scenarios:
            expected = {base_citation(c) for c in s["expected"]}
            original = build_query(s["complaint"], s["product"], s["issue"])
            if use_rw == "multi":
                q = rw[str(s["id"])]
                res = retriever.search_multi([q, original], product=s["product"], k=10,
                                             rerank_query=f"{q} {s['complaint'][:300]}", **opts)
                query = f"{q}  ||  (original complaint)"
            else:
                query = rw[str(s["id"])] if use_rw else original
                res = retriever.search(query, product=s["product"], k=10, **opts)
            got = [base_citation(c) for c in res["citation"]]
            top5 = got[:5]
            hit = any(g in expected for g in top5)
            first = next((i + 1 for i, g in enumerate(got) if g in expected), None)
            hits += hit
            recall += len(expected & set(top5)) / len(expected)
            rr += 1 / first if first else 0
            details.append({"config": name, "id": s["id"], "product": s["product"], "complaint": s["complaint"],
                            "expected": "; ".join(sorted(expected)), "query": query,
                            "top5": "; ".join(top5), "found_in_top5": hit, "first_correct_rank": first})
        n = len(scenarios)
        rows.append({"setup": name, "found_in_top5": f"{int(hits)}/{n}", "found_rate": hits / n,
                     "recall_at_5": recall / n, "mrr_at_10": rr / n})

    res = pd.DataFrame(rows)
    res.round(3).to_csv(config.REPORTS / "m5_retrieval.csv", index=False)
    det = pd.DataFrame(details)
    det.to_csv(config.REPORTS / "m5_retrieval_details.csv", index=False)
    print("\n", res.round(3).to_string(index=False))

    final = det[det["config"] == (FINAL if rw is not None else "Hybrid + rerank")]
    missed = final[~final["found_in_top5"]]
    print(f"\nMissed by '{FINAL}' ({len(missed)}):")
    for r in missed.itertuples():
        print(f"  #{r.id} [{r.product}] expected {r.expected}\n     query: {r.query}\n     got:   {r.top5}")
    multi = det[det["config"] == MULTI]
    if len(multi):
        fixed = set(missed["id"]) - set(multi[~multi["found_in_top5"]]["id"])
        broke = set(multi[~multi["found_in_top5"]]["id"]) - set(missed["id"])
        print(f"\nMulti-query vs final: fixed {sorted(fixed) or 'none'} | newly missed {sorted(broke) or 'none'}")
    print(f"\nSaved {config.REPORTS / 'm5_retrieval.csv'} and m5_retrieval_details.csv")


if __name__ == "__main__":
    main()
