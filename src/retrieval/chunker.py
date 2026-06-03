import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(processed_dir: str = "data/processed") -> list:
    docs = []
    for f in Path(processed_dir).glob("*.json"):
        doc = json.loads(f.read_text(encoding="utf-8"))
        if doc.get("text") and doc.get("word_count", 0) > 100:
            docs.append(doc)
    print(f"Loaded {len(docs)} documents")
    return docs

def chunk_documents(docs: list, out_dir: str = "data/chunks") -> list:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " "],
    )

    all_chunks = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}_chunk_{i}",
                "doc_id": doc["doc_id"],
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "url": doc.get("url", ""),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk,
            })

    out_path = Path(out_dir) / "chunks.json"
    out_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Created {len(all_chunks)} chunks from {len(docs)} documents")
    print(f"Avg chunk size: {sum(len(c['text'].split()) for c in all_chunks) // len(all_chunks)} words")
    return all_chunks

if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"\nSample chunk:")
    print(f"  ID: {chunks[0]['chunk_id']}")
    print(f"  Text: {chunks[0]['text'][:200]}")