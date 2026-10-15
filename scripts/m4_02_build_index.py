"""
Module 4, Step 2: cut the regulations into chunks and build the two search indexes.

    uv add sentence-transformers faiss-cpu rank-bm25
    uv run python scripts/m4_02_build_index.py

Two ways to search, combined in step 3 ("hybrid search"):
  - meaning search  : each chunk becomes a vector (embedding); FAISS finds the chunks whose meaning is
                      closest to the question, even if they use different words
  - keyword search  : BM25 finds chunks that share the exact words, e.g. "reinvestigation" or "1681i"

Writes models/m4_index/: chunks.parquet, dense.faiss, bm25.pkl, info.json
"""
import json
import pickle

import faiss
import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from cfpb import config
from cfpb.rag import INDEX_DIR, tokenize

EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # small, fast, strong for retrieval; runs fine on CPU
CHUNK_WORDS, OVERLAP = 220, 40


def chunks_of(text, size=CHUNK_WORDS, overlap=OVERLAP):
    words = text.split()
    if len(words) <= size:
        return [text]
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)]


def main():
    units = pd.read_parquet(config.INTERIM / "reg_units.parquet")
    rows = []
    for u in units.itertuples():
        pieces = chunks_of(u.text)
        for j, piece in enumerate(pieces):
            rows.append({"chunk_id": len(rows), "citation": u.citation, "title": u.title, "source": u.source,
                         "kind": u.kind,
                         "products": "|".join(u.products), "part": f"{j + 1}/{len(pieces)}",
                         # the title is kept with every chunk so each piece still says what it is about
                         "text": f"{u.title}\n{piece}"})
    chunks = pd.DataFrame(rows)
    print(f"{len(units):,} sections -> {len(chunks):,} chunks (~{CHUNK_WORDS} words each)")
    print(chunks.groupby("source").size().rename("chunks").to_string(), "\n")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print("Keyword index (BM25) ...")
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(BM25Okapi([tokenize(t) for t in chunks["text"]]), f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Meaning index: embedding with {EMBED_MODEL} on {device} ...")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    emb = model.encode(chunks["text"].tolist(), batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])     # inner product on unit vectors = cosine similarity
    index.add(emb)
    faiss.write_index(index, str(INDEX_DIR / "dense.faiss"))

    chunks.to_parquet(INDEX_DIR / "chunks.parquet", index=False)
    (INDEX_DIR / "info.json").write_text(json.dumps(
        {"embed_model": EMBED_MODEL, "chunk_words": CHUNK_WORDS, "overlap": OVERLAP,
         "n_chunks": len(chunks), "dim": int(emb.shape[1])}, indent=2))
    print(f"\nSaved to {INDEX_DIR}")


if __name__ == "__main__":
    main()
