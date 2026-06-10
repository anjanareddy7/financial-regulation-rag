import os
import json
import requests
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "rbi_sebi_circulars"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an AI research assistant specializing in machine learning and NLP papers.

STRICT RULES:
1. Answer ONLY using the provided context chunks.
2. Every claim must cite the chunk it came from using [Chunk X] format.
3. If the answer is not in the context, respond with exactly: "I cannot find this in the provided documents."
4. Never use outside knowledge. Never hallucinate.
5. Be precise and concise."""

def load_chunks() -> list:
    return json.loads(
        Path("data/chunks/chunks.json").read_text(encoding="utf-8")
    )

def build_bm25(chunks: list):
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)

def vector_search(query: str, collection, model: SentenceTransformer, n: int = 10) -> list:
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )
    hits = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        hits.append({
            "chunk_id": meta.get("chunk_id", f"vec_{i}"),
            "text": doc,
            "source": meta.get("title") or meta.get("doc_id", ""),
            "url": meta.get("url", ""),
            "vector_score": round(1 - dist, 4)
        })
    return hits

def bm25_search(query: str, bm25: BM25Okapi, chunks: list, n: int = 10) -> list:
    scores = bm25.get_scores(query.lower().split())
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

def rerank(query: str, chunks: list, reranker: CrossEncoder, top_n: int = 5) -> list:
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)
    return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_n]

def build_context(chunks: list) -> str:
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Chunk {i+1}] Source: {chunk['source']}\n{chunk['text']}\n"
    return context

def answer_query_groq(query: str, chunks: list, groq_client: Groq) -> str:
    context = build_context(chunks)
    prompt = f"""Context:
{context}

Question: {query}

Answer (cite chunks used):"""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

def answer_query_ollama(query: str, chunks: list, ollama_url: str) -> str:
    context = build_context(chunks)
    prompt = f"""You are an AI research assistant.
Answer ONLY using the provided context chunks.
Cite chunks using [Chunk X] format.
If the answer is not in the context, respond with: "I cannot find this in the provided documents."

Context:
{context}

Question: {query}

Answer (cite chunks used):"""
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )
    return response.json()["response"]

def query_pipeline(query: str):
    embed_model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    client = chromadb.PersistentClient(path="data/vectorstore")
    collection = client.get_collection(COLLECTION_NAME)
    chunks = load_chunks()
    bm25 = build_bm25(chunks)

    # step 1: hybrid retrieval
    vec_hits = vector_search(query, collection, embed_model, n=10)
    bm25_hits = bm25_search(query, bm25, chunks, n=10)
    fused = reciprocal_rank_fusion(vec_hits, bm25_hits)[:10]

    # step 2: rerank
    reranked = rerank(query, fused, reranker, top_n=5)

    # step 3: answer with configurable backend
    backend = os.getenv("INFERENCE_BACKEND", "groq")

    if backend == "ollama":
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        response = answer_query_ollama(query, reranked, ollama_url)
        print(f"[Backend: Ollama/Mistral]")
    else:
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = answer_query_groq(query, reranked, groq_client)
        print(f"[Backend: Groq/llama-3.3-70b]")

    print(f"\nQuery: {query}")
    print("-" * 60)
    print(f"\nAnswer:\n{response}")
    print("\nSources (after reranking):")
    for i, chunk in enumerate(reranked):
        print(f"  [{i+1}] {chunk['source'][:50]} "
              f"(rerank: {chunk['rerank_score']} | rrf: {chunk.get('rrf_score', 'n/a')})")

    return {"query": query, "answer": response, "chunks": reranked}

if __name__ == "__main__":
    query_pipeline("How does retrieval augmented generation reduce hallucination?")