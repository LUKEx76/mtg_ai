import json
import os
import re
import time
import unicodedata
import pandas as pd
import requests
from typing import Dict, Any

HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (compatible; MTG-EDHREC-Scraper/0.1; +https://example.com/your-contact)",
    "Accept-Language": "en-US,en;q=0.9",
}

def get(url: str, max_retries: int = 3, backoff: float = 1.5) -> requests.Response:
    last_exc: Exception = None
    for attempt in range(1, max_retries + 1):
        try:
            resp: requests.Response = requests.get(url=url, headers=HEADERS, timeout=20)
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

def sluggify_commander_name(name: str) -> str:
    normalized: str = unicodedata.normalize('NFKD', name)
    name_ascii: str = normalized.encode('ascii', 'ignore').decode('utf-8')
    name_lower: str = name_ascii.lower()
    name_clean: str = str(name_lower).replace("//","-")
    name_clean = re.sub(pattern=r'[^\w\s-]', repl='', string=name_clean)
    slug: str = re.sub(pattern=r'[-\s]+', repl='-', string=name_clean).strip('-')
    
    return slug


if __name__ == "__main__":
    base_url: str = "https://json.edhrec.com/pages/decks/"
    output_dir: str = "data/edhrec/decks"
    os.makedirs(name=output_dir, exist_ok=True)
    
    df: pd.DataFrame = pd.read_parquet(path="data/cards_with_embeddings.parquet")
    # todo: add planeswalkers that can be commanders
    legendary_creatures: pd.DataFrame = df[df['type_line'].str.contains("Legendary") & df['type_line'].str.contains("Creature")]
    total: int = len(legendary_creatures)

    for i, (index, row) in enumerate(iterable=legendary_creatures.iterrows(), start=1):
        print(f"\r({i}/{total}) ", end="")
        type_line: str = row["type_line"].split("//")[0].lower()
        name: str = row["name"]

        # exclude double sided cards with legendary backside
        if not ("legendary" in type_line and "creature" in type_line):
            print(f"Skipping {name}, with type line {row['type_line']}")
            continue

        # query first name for double sided cards
        # exception for sp//dr
        if "//" in name and name.lower() != "SP//dr, Piloted by Peni".lower():
            name = name.split("//")[0]

        clean_name: str = sluggify_commander_name(name=name)
        file_path: str = os.path.join(output_dir, f"{clean_name}.json")

        if os.path.exists(path=file_path):
            print(f"Skipping {clean_name} (already exists)")
            continue

        if clean_name in ["mishra-lost-to-phyrexia", "titania-gaea-incarnate", "brisela-voice-of-nightmares", "hanweir-the-writhing-township"]:
            print(f"Skipping {clean_name} (non legal single commander)")
            continue

        url: str = f"{base_url}{clean_name}.json"

        try:
            print(f"Fetching: {clean_name}...")
            response: requests.Response = get(url=url)
            data: Any = response.json()
            
            with open(file=file_path, mode="w", encoding="utf-8") as f:
                json.dump(obj=data, fp=f, indent=4)
            
            print(f"Successfully saved to {file_path}")

        except Exception as e:
            print(f"Failed to process {clean_name}")
            print(f"Error details: {e}")
            print(row["name"])
            print(row["type_line"])
            exit()
        
        time.sleep(0.1)

    print("Done scraping commanders")
