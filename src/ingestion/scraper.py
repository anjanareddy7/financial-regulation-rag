import arxiv
import json
import time
import requests
from pathlib import Path

TOPICS = [
    "retrieval augmented generation",
    "large language models hallucination",
    "LLM evaluation benchmarks",
    "vector search embeddings semantic",
    "transformer fine tuning instruction"
]

PAPERS_PER_TOPIC = 60

def download_pdf(url: str, out_path: Path) -> bool:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research project)"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False

def scrape_arxiv(out_dir: str = "data/raw/arxiv", papers_per_topic: int = PAPERS_PER_TOPIC):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    client = arxiv.Client(
        page_size=10,
        delay_seconds=8,
        num_retries=3
    )

    seen_ids = set()
    total = 0

    for topic in TOPICS:
        print(f"\nSearching: '{topic}'")
        time.sleep(15)  # pause before each topic to avoid rate limit

        search = arxiv.Search(
            query=topic,
            max_results=papers_per_topic,
            sort_by=arxiv.SortCriterion.Relevance
        )

        topic_count = 0
        try:
            for paper in client.results(search):
                paper_id = paper.entry_id.split("/")[-1]

                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)

                pdf_path = Path(out_dir) / f"{paper_id}.pdf"
                meta_path = Path(out_dir) / f"{paper_id}.json"

                if pdf_path.exists():
                    topic_count += 1
                    total += 1
                    continue

                # use requests to download instead of paper.download_pdf()
                pdf_url = f"https://arxiv.org/pdf/{paper_id}"
                success = download_pdf(pdf_url, pdf_path)

                if success:
                    meta = {
                        "doc_id": paper_id,
                        "title": paper.title,
                        "authors": [a.name for a in paper.authors[:3]],
                        "published": str(paper.published.date()),
                        "topic": topic,
                        "url": pdf_url,
                        "abstract": paper.summary[:500]
                    }
                    meta_path.write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2),
                        encoding="utf-8"
                    )
                    topic_count += 1
                    total += 1
                    print(f"  [{topic_count}] {paper.title[:70]}")

                time.sleep(5)  # respectful delay between downloads

        except Exception as e:
            print(f"  Search failed: {e}")
            time.sleep(30)  # longer wait if rate limited

        print(f"  Done: {topic_count} papers for '{topic}'")

    print(f"\nTotal papers downloaded: {total}")

if __name__ == "__main__":
    scrape_arxiv()