import os
import json
import chromadb
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "rbi_sebi_circulars"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a financial regulation assistant specializing in SEBI and RBI guidelines.

STRICT RULES:
1. Answer ONLY using the provided context chunks.
2. Every claim must cite the chunk it came from using [Chunk X] format.
3. If the answer is not in the context, respond with exactly: "I cannot find this in the provided documents."
4. Never use outside knowledge. Never hallucinate.
5. Be precise and concise."""

# global state — models load once at startup
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    state["embed_model"] = SentenceTransformer(MODEL_NAME)
    state["reranker"] = CrossEncoder(RERANKER_MODEL, max_length=512)
    state["groq"] = Groq(api_key=os.getenv("GROQ_API_KEY"))
    state["collection"] = chromadb.PersistentClient(
        path="data/vectorstore"
    ).get_collection(COLLECTION_NAME)
    chunks = json.loads(Path("data/chunks/chunks.json").read_text(encoding="utf-8"))
    state["chunks"] = chunks
    state["bm25"] = BM25Okapi([c["text"].lower().split() for c in chunks])
    print("All models loaded. API ready.")
    yield
    state.clear()

app = FastAPI(
    title="SEBI/RBI Regulation RAG API",
    description="Ask questions over SEBI and RBI regulatory documents",
    version="1.0.0",
    lifespan=lifespan
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class ChunkSource(BaseModel):
    chunk_id: str
    source: str
    url: str
    rerank_score: float
    text_preview: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[ChunkSource]

def vector_search(query: str, n: int = 10) -> list:
    embedding = state["embed_model"].encode([query]).tolist()
    results = state["collection"].query(
        query_embeddings=embedding,
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

def bm25_search(query: str, n: int = 10) -> list:
    scores = state["bm25"].get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    chunks = state["chunks"]
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

def reciprocal_rank_fusion(vec_hits: list, bm25_hits: list, k: int = 60) -> list:
    scores = {}
    docs = {}
    for rank, hit in enumerate(vec_hits):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        docs[cid] = hit
    for rank, hit in enumerate(bm25_hits):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        if cid not in docs:
            docs[cid] = hit
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**docs[cid], "rrf_score": round(score, 6)} for cid, score in ranked]

def rerank(query: str, chunks: list, top_n: int = 5) -> list:
    pairs = [(query, c["text"]) for c in chunks]
    scores = state["reranker"].predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)
    return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_n]

def build_context(chunks: list) -> str:
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Chunk {i+1}] Source: {chunk['source']}\n{chunk['text']}\n"
    return context

def get_answer(query: str, chunks: list) -> str:
    context = build_context(chunks)
    prompt = f"""Context:
{context}

Question: {query}

Answer (cite chunks used):"""
    response = state["groq"].chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(state.get("chunks", []))}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    vec_hits = vector_search(request.question, n=10)
    bm25_hits = bm25_search(request.question, n=10)
    fused = reciprocal_rank_fusion(vec_hits, bm25_hits)[:10]
    reranked = rerank(request.question, fused, top_n=request.top_k)
    answer = get_answer(request.question, reranked)

    sources = [ChunkSource(
        chunk_id=c["chunk_id"],
        source=c["source"],
        url=c.get("url", ""),
        rerank_score=c["rerank_score"],
        text_preview=c["text"][:200]
    ) for c in reranked]

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources
    )