import json
import chromadb
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for now
COLLECTION_NAME = "rbi_sebi_circulars"

def load_chunks(chunks_path: str = "data/chunks/chunks.json") -> list:
    return json.loads(Path(chunks_path).read_text(encoding="utf-8"))

def build_vectorstore(chunks: list, db_path: str = "data/vectorstore"):
    Path(db_path).mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=db_path)

    # delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Embedding {len(chunks)} chunks...")
    batch_size = 32

    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [{
            "doc_id": c["doc_id"],
            "title": c["title"],
            "source": c["source"],
            "url": c["url"],
            "chunk_index": c["chunk_index"],
            "total_chunks": c["total_chunks"],
        } for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    print(f"Vectorstore built. {collection.count()} chunks indexed.")
    return collection

def test_query(query: str = "What are the cybersecurity requirements for SEBI?"):
    client = chromadb.PersistentClient(path="data/vectorstore")
    collection = client.get_collection(COLLECTION_NAME)

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\nQuery: {query}")
    print("-" * 60)
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\nResult {i+1} (score: {1 - dist:.3f})")
        print(f"Source: {meta['title'] or meta['doc_id']}")
        print(f"Text: {doc[:200]}")

if __name__ == "__main__":
    chunks = load_chunks()
    build_vectorstore(chunks)
    test_query()