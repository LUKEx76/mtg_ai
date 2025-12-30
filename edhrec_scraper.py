import json
import os
import re
import time
import unicodedata
import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MTG-EDHREC-Scraper/0.1; +https://example.com/your-contact)",
    "Accept-Language": "en-US,en;q=0.9",
}

def get(url: str, max_retries: int = 3, backoff: float = 1.5) -> requests.Response:
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if 200 <= resp.status_code < 300:
                return resp
            # 429/5xx backoff
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
    normalized = unicodedata.normalize('NFKD', name)
    name_ascii = normalized.encode('ascii', 'ignore').decode('utf-8')
    name_lower = name_ascii.lower()
    name_clean = str(name_lower).replace("//","-")
    name_clean = re.sub(r'[^\w\s-]', '', name_clean)
    slug = re.sub(r'[-\s]+', '-', name_clean).strip('-')
    
    return slug


if __name__ == "__main__":
    # Static API enpoints:
    # Decks of a commander: https://json.edhrec.com/pages/decks/{sluggified_commander_name}.json
    # Average deck of a commander: https://json.edhrec.com/pages/average-decks/{sluggified_commander_name}.json
    # Stats of a commander: https://json.edhrec.com/pages/commanders/{sluggified_commander_name}.json
    # Card: https://json.edhrec.com/pages/cards/{sluggified_card_name}.json

    # Dynamic API endpoints:
    # Decklists
    
    # for the deck with id : https://edhrec.com/deckpreview/RZ_HK4XKENoKMywO-8nWeQ
    # the link to copy the decklist to the clipboard is: "https://edhrec.com/deckpreview/RZ_HK4XKENoKMywO-8nWeQ#"
    # test_url = "https://edhrec.com/deckpreview/RZ_HK4XKENoKMywO-8nWeQ#"
    # commander_url = "https://edhrec.com/commanders/the-wise-mothman"  # change me

    # To scrape decks, use this url:
    # https://edhrec.com/_next/data/MuZ2MLc8SEtUXZB9xfwdh/deckpreview/wQqvoCVrJi54gH65huggag.json?deckId=wQqvoCVrJi54gH65huggag
    # the MuZ2MLc8SEtUXZB9xfwdh seems to be the hash of the commanders name Avacyn, Angel of Hope

    base_url = "https://json.edhrec.com/pages/decks/"
    output_dir = "data/edhrec/decks"
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_parquet("data/cards_with_embeddings.parquet")
    # TODO: add planeswalkers that say "can be your commander"
    legendary_creatures = df[df['type_line'].str.contains("Legendary") & df['type_line'].str.contains("Creature")]
    total = len(legendary_creatures)

    for i, (index, row) in enumerate(legendary_creatures.iterrows(), start=1):
        print(f"\r({i}/{total}) ", end="")
        type_line = row["type_line"].split("//")[0].lower()
        name = row["name"]

        # make sure double sided cards with legendary backside are excluded
        if not ("legendary" in type_line and "creature" in type_line):
            print(f"Skipping {name}, with type line {row["type_line"]}")
            continue

        # for double sided cards only query the first name
        # exception, the card "SP//dr, Piloted by Peni"
        if "//" in name and name.lower() != "SP//dr, Piloted by Peni".lower():
            name = name.split("//")[0]

        clean_name = sluggify_commander_name(name)

        file_path = os.path.join(output_dir, f"{clean_name}.json")

        if os.path.exists(file_path):
            print(f"Skipping {clean_name} (already exists)")
            continue

        if clean_name in ["mishra-lost-to-phyrexia", "titania-gaea-incarnate", "brisela-voice-of-nightmares", "hanweir-the-writhing-township"]:
            print(f"Skipping {clean_name} (non legal single commander)")
            continue

        url = f"{base_url}{clean_name}.json"

        try:
            print(f"Fetching: {clean_name}...")
            response = get(url)
            data = response.json()
            
            file_path = f"{output_dir}/{clean_name}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            
            print(f"Successfully saved to {file_path}")

        except Exception as e:
            print(f"Failed to process {clean_name}")
            print(f"Error details: {e}")
            print(row["name"])
            print(row["type_line"])
            exit()
        
        time.sleep(0.1)

    print("Done scraping commanders")
