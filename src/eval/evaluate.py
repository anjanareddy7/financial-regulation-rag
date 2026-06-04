import os
import json
import time
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "rbi_sebi_circulars"
GROQ_MODEL = "llama-3.1-8b-instant"  # use faster model for eval runs

SYSTEM_PROMPT = """You are an AI research assistant.
Answer ONLY using the provided context chunks.
Cite chunks using [Chunk X] format.
If the answer is not in the context, respond with exactly: "I cannot find this in the provided documents."
Never use outside knowledge."""

def load_chunks():
    return json.loads(Path("data/chunks/chunks.json").read_text(encoding="utf-8"))

def build_bm25(chunks):
    return BM25Okapi([c["text"].lower().split() for c in chunks])

def vector_search(query, collection, model, n=10):
    embedding = model.encode([query]).tolist()
    results = collection.query(
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
            "vector_score": round(1 - dist, 4)
        })
    return hits

def bm25_search(query, bm25, chunks, n=10):
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [{
        "chunk_id": chunks[idx]["chunk_id"],
        "text": chunks[idx]["text"],
        "source": chunks[idx].get("title") or chunks[idx].get("doc_id", ""),
        "bm25_score": round(float(scores[idx]), 4)
    } for idx in top_indices]

def reciprocal_rank_fusion(vec_hits, bm25_hits, k=60):
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

def rerank(query, chunks, reranker, top_n=5):
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)
    return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_n]

def get_answer(query, chunks, groq_client):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Chunk {i+1}] Source: {chunk['source']}\n{chunk['text']}\n"
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer (cite chunks):"
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

def run_eval():
    print("Loading models...")
    embed_model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    client = chromadb.PersistentClient(path="data/vectorstore")
    collection = client.get_collection(COLLECTION_NAME)
    chunks = load_chunks()
    bm25 = build_bm25(chunks)

    questions = json.loads(
        Path("data/eval_questions.json").read_text(encoding="utf-8")
    )

    print(f"Running eval on {len(questions)} questions...")

    results = {
        "questions": [],
        "answers": [],
        "contexts": [],
        "ground_truths": []
    }

    for i, item in enumerate(questions):
        query = item["question"]
        print(f"  [{i+1}/{len(questions)}] {query[:60]}...")

        vec_hits = vector_search(query, collection, embed_model, n=10)
        bm25_hits = bm25_search(query, bm25, chunks, n=10)
        fused = reciprocal_rank_fusion(vec_hits, bm25_hits)[:10]
        reranked = rerank(query, fused, reranker, top_n=5)
        answer = get_answer(query, reranked, groq_client)

        results["questions"].append(query)
        results["answers"].append(answer)
        results["contexts"].append([c["text"] for c in reranked])
        results["ground_truths"].append(item.get("ground_truth", ""))

        time.sleep(2)  # respect Groq rate limits

    # save raw results
    Path("data/eval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("Raw results saved to data/eval_results.json")

    from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# configure RAGAS to use Groq instead of OpenAI
groq_llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
))
hf_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

faithfulness.llm = groq_llm
answer_relevancy.llm = groq_llm
answer_relevancy.embeddings = hf_embeddings
context_precision.llm = groq_llm

print("\nRunning RAGAS scoring...")
dataset = Dataset.from_dict({
    "question": results["questions"],
    "answer": results["answers"],
    "contexts": results["contexts"],
    "ground_truth": results["ground_truths"]
})

scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
)

    print("\n" + "="*50)
    print("EVAL RESULTS")
    print("="*50)
    print(f"Faithfulness:      {scores['faithfulness']:.4f}")
    print(f"Answer Relevancy:  {scores['answer_relevancy']:.4f}")
    print(f"Context Precision: {scores['context_precision']:.4f}")
    print("="*50)

    Path("data/eval_scores.json").write_text(
        json.dumps({
            "faithfulness": scores["faithfulness"],
            "answer_relevancy": scores["answer_relevancy"],
            "context_precision": scores["context_precision"]
        }, indent=2),
        encoding="utf-8"
    )
    print("Scores saved to data/eval_scores.json")

if __name__ == "__main__":
    run_eval()