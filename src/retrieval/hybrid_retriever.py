import json
import chromadb
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rbi_sebi_circulars"

def load_chunks(chunks_path: str = "data/chunks/chunks.json") -> list:
    return json.loads(Path(chunks_path).read_text(encoding="utf-8"))

def build_bm25(chunks: list) -> tuple:
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, chunks

def vector_search(query: str, collection, model: SentenceTransformer, n: int = 10) -> list:
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        hits.append({
            "chunk_id": meta.get("chunk_id", ""),
            "text": doc,
            "source": meta.get("title") or meta.get("doc_id", ""),
            "url": meta.get("url", ""),
            "vector_score": round(1 - dist, 4)
        })
    return hits

def bm25_search(query: str, bm25: BM25Okapi, chunks: list, n: int = 10) -> list:
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    hits = []
    for idx in top_indices:
        hits.append({
            "chunk_id": chunks[idx]["chunk_id"],
            "text": chunks[idx]["text"],
            "source": chunks[idx].get("title") or chunks[idx].get("doc_id", ""),
            "url": chunks[idx].get("url", ""),
            "bm25_score": round(float(scores[idx]), 4)
        })
    return hits

def reciprocal_rank_fusion(vector_hits: list, bm25_hits: list, k: int = 60) -> list:
    scores = {}
    docs = {}

    for rank, hit in enumerate(vector_hits):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        docs[cid] = hit

    for rank, hit in enumerate(bm25_hits):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        if cid not in docs:
            docs[cid] = hit

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for cid, score in ranked:
        entry = docs[cid].copy()
        entry["rrf_score"] = round(score, 6)
        results.append(entry)
    return results

def hybrid_search(query: str, collection, model: SentenceTransformer,
                  bm25: BM25Okapi, chunks: list, n: int = 5) -> list:
    vector_hits = vector_search(query, collection, model, n=10)
    bm25_hits = bm25_search(query, bm25, chunks, n=10)
    fused = reciprocal_rank_fusion(vector_hits, bm25_hits)
    return fused[:n]

if __name__ == "__main__":
    chunks = load_chunks()
    bm25, chunks = build_bm25(chunks)
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path="data/vectorstore")
    collection = client.get_collection(COLLECTION_NAME)

    queries = [
        "What are the cybersecurity requirements for SEBI regulated entities?",
        "CSCRF compliance audit requirements",
        "data breach incident reporting"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        results = hybrid_search(query, collection, model, bm25, chunks)
        for i, r in enumerate(results):
            print(f"  [{i+1}] RRF: {r['rrf_score']} | {r['source'][:50]}")
            print(f"       {r['text'][:120]}")