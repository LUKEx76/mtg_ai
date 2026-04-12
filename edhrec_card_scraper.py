import json
import os
import re
import time
import unicodedata
import pandas as pd
import requests
from typing import List, Dict, Any

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MTG-EDHREC-Scraper/0.1; +https://example.com/your-contact)",
    "Accept-Language": "en-US,en;q=0.9",
}

def get(url: str, max_retries: int = 3, backoff: float = 1.5) -> requests.Response:
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url=url, headers=HEADERS, timeout=20)
            if 200 <= resp.status_code < 300:
                return resp
            # backoff for rate limits or errors
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff * attempt)
                continue
            resp.raise_for_status()
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            time.sleep(backoff * attempt)
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to GET {url}")

def sluggify(sub_name: str) -> str:
    normalized = unicodedata.normalize('NFKD', sub_name)
    name_ascii = normalized.encode('ascii', 'ignore').decode('utf-8')
    name_lower = name_ascii.lower()
    name_clean = str(name_lower).replace("//","-")
    name_clean = re.sub(pattern=r'[^\w\s-]', repl='', string=name_clean)
    slug = re.sub(pattern=r'[-\s]+', repl='-', string=name_clean).strip('-')
    return slug

def sluggify_card_name(name: str) -> List[str]:
    slug_list = []
    slug_list.append(sluggify(sub_name=name))

    if "//" in name:
        split_name = name.split("//")[0]
        slug_list.append(sluggify(sub_name=split_name))

    # name exceptions
    if name == "+2 Mace":
        slug_list.append("plus-2-mace")
    elif name == "Summon: Choco/Mog":
        slug_list.append("summon-choco-mog")
    elif name == "Welcome to . . . // Jurassic Park":
        slug_list.append("welcome-to-")
    elif name == "SP//dr, Piloted by Peni":
        slug_list.append("sp-dr-piloted-by-peni")
    
    return slug_list


if __name__ == "__main__":
    base_url = "https://json.edhrec.com/pages/cards/"
    output_dir = "data/edhrec/cards"
    os.makedirs(name=output_dir, exist_ok=True)
    
    df = pd.read_excel(io="data/commander_legal_pixel_art.xlsx")
    total = len(df)

    for i, (index, row) in enumerate(iterable=df.iterrows(), start=1):
        print(f"\r({i}/{total}) {int((i/total)*100)}% - ", end="")
        name = row["name"]
        clean_names = sluggify_card_name(name=name)

        for clean_name in clean_names:
            file_path = os.path.join(output_dir, f"{clean_name}.json")

            if os.path.exists(path=file_path):
                print(f"Skipping {clean_name} (already exists)")
                continue

            url = f"{base_url}{clean_name}.json"

            try:
                print(f"Fetching: {clean_name}...")
                response = get(url=url)
                data = response.json()
                
                with open(file=file_path, mode="w", encoding="utf-8") as f:
                    json.dump(obj=data, fp=f, indent=4)
                
                print(f"Successfully saved to {file_path}")

            except Exception as e:
                print(f"Failed to process {clean_name}")
                print(f"Error details: {e}")
                print(f"Card name {row['name']}\n")
                continue
            
            time.sleep(0.1)

    print("Done scraping cards")

