import fitz
import json
from pathlib import Path
from tqdm import tqdm

def extract_pdf(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n".join(pages)

def process_all_pdfs(raw_dirs: list = ["data/raw/rbi", "data/raw/sebi"],
                     out_dir: str = "data/processed"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    skipped = 0
    processed = 0

    for raw_dir in raw_dirs:
        pdfs = list(Path(raw_dir).glob("*.pdf"))
        print(f"\nProcessing {len(pdfs)} PDFs from {raw_dir}...")

        for pdf_path in tqdm(pdfs):
            meta_path = pdf_path.with_suffix(".json")
            meta = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))

            try:
                text = extract_pdf(pdf_path)
            except Exception as e:
                print(f"Failed to extract {pdf_path.name}: {e}")
                skipped += 1
                continue

            # skip if too short — likely a scanned image PDF
            if len(text.split()) < 100:
                print(f"Skipping {pdf_path.name} — likely scanned, too short")
                skipped += 1
                continue

            out = {
                "doc_id": pdf_path.stem,
                "title": meta.get("title", pdf_path.stem),
                "source": meta.get("source", "unknown"),
                "url": meta.get("url", ""),
                "filename": pdf_path.name,
                "text": text,
                "word_count": len(text.split())
            }

            out_path = Path(out_dir) / f"{pdf_path.stem}.json"
            out_path.write_text(
                json.dumps(out, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            processed += 1

    print(f"\nDone. {processed} processed, {skipped} skipped.")

if __name__ == "__main__":
    process_all_pdfs()