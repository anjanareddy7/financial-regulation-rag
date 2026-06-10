
# AI Research Paper Assistant

A production-grade RAG (Retrieval Augmented Generation) system that allows you to ask questions over 423 AI research papers and receive cited, grounded answers. Built to demonstrate real-world AI engineering patterns including hybrid retrieval, reranking, citation enforcement, and evaluation.

---

## Demo

> **Q: How does retrieval augmented generation improve factual accuracy of LLMs?**
>
> Retrieval-augmented models can improve factual correctness of LLMs [Chunk 1]. Memory augmentation strategies, such as retrieving from an external knowledge source, help the language model avoid producing non-factual and out-of-date information [Chunk 2]. Retrieval-augmented models generally decrease performance the least with respect to knowledge F1 scores, indicating the augmentation can effectively retrieve knowledge on these topics [Chunk 3].
>
> **Sources:** OPT: Open Pre-trained Transformer Language Models | Rethinking with Retrieval | Is ChatGPT Good at Search?

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Hybrid Retrieval              │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │ BM25 Search │  │  Vector Search   │  │
│  │ (keywords)  │  │  (semantic)      │  │
│  └──────┬──────┘  └────────┬─────────┘  │
│         └────────┬─────────┘            │
│                  ▼                      │
│      Reciprocal Rank Fusion             │
└──────────────────┬──────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│         Cross-Encoder Reranking          │
│   ms-marco-MiniLM-L-6-v2               │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│      LLM Generation with Citation        │
│   Enforcement (Groq llama-3.3-70b)      │
│   Refuses to answer if not in context   │
└──────────────────┬───────────────────────┘
                   ▼
            Cited Answer
```

---

## Evaluation Results

Evaluated on 30 manually curated questions using LLM-as-judge scoring:

| Metric | Score |
|--------|-------|
| Faithfulness | 0.88 |
| Answer Relevancy | 0.72 |
| Context Precision | 0.80 |

**Faithfulness (0.88):** 88% of answer claims are grounded in retrieved chunks — citation enforcement is working.

**Answer Relevancy (0.72):** 72% of answers directly address the question. Gap is partly from correct refusals on out-of-corpus questions.

**Context Precision (0.80):** 80% of retrieved chunks were relevant — hybrid search + reranking is finding the right passages.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq (llama-3.3-70b-versatile) |
| API | FastAPI |
| UI | Streamlit |
| Dataset | jamescalam/ai-arxiv (423 papers) |
| Observability | Langfuse |
---

## Project Structure

```
case-law-rag/
├── src/
│   ├── ingestion/
│   │   ├── dataset_loader.py    # loads arxiv papers from HuggingFace
│   │   ├── pdf_extractor.py     # extracts text from PDFs
│   │   └── cleaner.py           # cleans and normalizes text
│   ├── retrieval/
│   │   ├── chunker.py           # chunks documents with metadata
│   │   ├── embedder.py          # builds ChromaDB vectorstore
│   │   ├── hybrid_retriever.py  # BM25 + vector + RRF fusion
│   │   ├── reranker.py          # cross-encoder reranking
│   │   └── query_engine.py      # end-to-end query pipeline
│   ├── api/
│   │   └── main.py              # FastAPI REST endpoint
│   └── eval/
│       ├── evaluate.py          # runs queries over eval dataset
│       └── score.py             # LLM-as-judge scoring
├── app.py                       # Streamlit chat UI
├── data/
│   └── eval_questions.json      # 30 evaluation questions
└── requirements.txt
```

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/anjanareddy7/financial-regulation-rag.git
cd financial-regulation-rag
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Set up environment

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key
```

Get a free Groq API key at console.groq.com

### 3. Build the corpus

```bash
# Download 423 AI research papers
python src/ingestion/dataset_loader.py

# Clean and chunk documents
python src/ingestion/cleaner.py
python src/retrieval/chunker.py

# Build vector store (takes ~70 mins on CPU)
python src/retrieval/embedder.py
```

### 4. Run the API

```bash
uvicorn src.api.main:app --reload
```

### 5. Run the UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## API Reference

### POST /query

```json
{
  "question": "How does retrieval augmented generation work?",
  "top_k": 5
}
```

Response:
```json
{
  "question": "How does retrieval augmented generation work?",
  "answer": "RAG improves factual accuracy by retrieving relevant chunks... [Chunk 1]",
  "sources": [
    {
      "chunk_id": "1234_chunk_5",
      "source": "Retrieval Augmentation Reduces Hallucination",
      "rerank_score": 8.45,
      "text_preview": "..."
    }
  ]
}
```

### GET /health

```json
{"status": "ok", "chunks_loaded": 79200}
```

---

## How It Works

**1. Hybrid Retrieval**
Every query runs both BM25 keyword search and vector similarity search in parallel. BM25 catches exact technical terms (paper names, model names, acronyms). Vector search catches semantic meaning. Results are fused using Reciprocal Rank Fusion.

**2. Cross-Encoder Reranking**
The top 10 fused results are passed to a cross-encoder reranker that jointly scores the query and each chunk together. This is more accurate than bi-encoder similarity but too slow to run on the full corpus — hence the two-stage pipeline.

**3. Citation Enforcement**
The LLM is instructed to only use provided context chunks and cite them explicitly. If the answer isn't in the retrieved chunks, it responds with "I cannot find this in the provided documents" rather than hallucinating.

---

## Evaluation

Run the evaluation pipeline:

```bash
# Generate answers for 30 eval questions
python src/eval/evaluate.py

# Score with LLM-as-judge
python src/eval/score.py
```
## Inference Backend Comparison

| Backend | Model | Avg Latency | Cost | Privacy |
|---------|-------|-------------|------|---------|
| Groq (cloud) | llama-3.3-70b-versatile | 0.24s | Free tier | Data sent to API |
| Ollama (local) | Mistral-7B | 4.17s | Free | Fully private |

Benchmarked on Google Colab T4 GPU. Switch backends via `INFERENCE_BACKEND=groq/ollama` in `.env`.
---

## What I'd Add Next

- **Langfuse observability** — trace every request with latency, cost, and quality metrics
- **Fine-tuning with LoRA** — domain-specific fine-tuning on AI research QA pairs to close the relevancy gap
- **Semantic Scholar API** — expand corpus to 500+ papers with approved API access
- **Query routing** — classify queries and route to different retrieval strategies

---

## About

Built as a portfolio project to demonstrate production RAG engineering patterns. The pipeline prioritizes retrieval quality over generation quality — the biggest wins in RAG come from better retrieval, not bigger models.
```

