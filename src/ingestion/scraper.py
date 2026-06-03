import requests
import json
import time
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_rbi_links_for_year(year: int) -> list:
    url = f"https://www.rbi.org.in/Scripts/BS_ViewMasCirculardetails.aspx?yr={year}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(resp.text, "html.parser")
    links = soup.find_all("a", href=True)
    
    results = []
    for a in links:
        href = a.get("href", "")
        if "rbidocs" in href and ".PDF" in href.upper():
            results.append({
                "url": href if href.startswith("http") else "https:" + href,
                "title": a.text.strip(),
                "source": "RBI",
                "year": year
            })
    return results

def scrape_rbi(out_dir: str = "data/raw/rbi", years: list = None):
    if years is None:
        years = [2019, 2020, 2021, 2022, 2023, 2024]
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    all_links = []

    print("Collecting RBI circular links...")
    for year in years:
        links = get_rbi_links_for_year(year)
        print(f"  {year}: {len(links)} circulars")
        all_links.extend(links)
        time.sleep(0.5)

    print(f"\nTotal: {len(all_links)} circulars found")
    download_pdfs(all_links, out_dir)

def download_pdfs(links: list, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    failed = 0

    for item in tqdm(links, desc=f"Downloading PDFs"):
        url = item["url"]
        filename = url.split("/")[-1]
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"

        out_path = Path(out_dir) / filename
        if out_path.exists():
            continue

        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)

            meta_path = out_path.with_suffix(".json")
            meta_path.write_text(
                json.dumps({
                    "title": item.get("title", ""),
                    "source": item.get("source", "RBI"),
                    "url": url,
                    "year": item.get("year", ""),
                    "filename": filename
                }, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            print(f"\nFailed: {url} — {e}")
            failed += 1
        time.sleep(0.3)

    print(f"\nDone. {len(links) - failed} downloaded, {failed} failed.")

if __name__ == "__main__":
    scrape_rbi(years=[2019, 2020, 2021, 2022, 2023, 2024])