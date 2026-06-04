import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def load_and_save(out_dir: str = "data/processed"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("Loading jamescalam/ai-arxiv dataset...")
    ds = load_dataset("jamescalam/ai-arxiv", split="train")
    print(f"Total papers: {len(ds)}")

    saved = 0
    skipped = 0

    for paper in tqdm(ds, desc="Processing papers"):
        content = paper.get("content", "").strip()
        if len(content.split()) < 200:
            skipped += 1
            continue

        doc_id = paper["id"].replace("/", "_").replace(".", "_")
        out = {
            "doc_id": doc_id,
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "published": paper.get("published", ""),
            "categories": paper.get("categories", []),
            "summary": paper.get("summary", ""),
            "source": "arxiv",
            "url": f"https://arxiv.org/abs/{paper['id']}",
            "text": content,
            "word_count": len(content.split())
        }

        out_path = Path(out_dir) / f"{doc_id}.json"
        out_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        saved += 1

    print(f"\nDone. {saved} saved, {skipped} skipped.")
    print(f"Total words in corpus: {sum(json.loads(f.read_text(encoding='utf-8'))['word_count'] for f in Path(out_dir).glob('*.json')):,}")

if __name__ == "__main__":
    load_and_save()