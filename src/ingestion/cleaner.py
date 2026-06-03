import json
import re
from pathlib import Path

NOISE_PATTERNS = [
    r"Page \d+ of \d+",
    r"^\d+$",                          # lone page numbers
    r"IN THE SUPREME COURT OF INDIA",  # repeated headers
    r"CIVIL APPELLATE JURISDICTION",
    r"\[.*?\d{4}.*?\]",               # citation brackets like [2019] 4 SCC 123
    r"={3,}",                          # separator lines
]

def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # drop noise lines
        if any(re.search(p, line) for p in NOISE_PATTERNS):
            continue
        # drop very short lines that are likely artifacts
        if len(line) < 10 and not line.endswith(":"):
            continue
        cleaned.append(line)

    # Rejoin, collapsing multiple blank lines
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()

def clean_corpus(processed_dir: str = "data/processed"):
    files = list(Path(processed_dir).glob("*.json"))
    print(f"Cleaning {len(files)} documents...")

    for f in files:
        doc = json.loads(f.read_text(encoding='utf-8'))
        doc["text"] = clean_text(doc["text"])
        f.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')

    print("Cleaning complete.")

if __name__ == "__main__":
    clean_corpus()