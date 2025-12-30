import json
import pandas as pd
import ftfy

# Layouts to exclude outright
EXCLUDE_LAYOUTS = {
    "art_series", "token", "double_faced_token", "emblem", "scheme", 
    "plane", "planar", "phenomenon", "vanguard", "reversible_card"
}

def filter_dataset(input_path) -> pd.DataFrame:
    print("Loading dataset...")
    with open(input_path, "r", encoding="utf-8") as f:
        cards = json.load(f)

    df = pd.DataFrame(cards)

    # 1. Exclude layouts
    if "layout" in df.columns:
        df = df[~df["layout"].isin(EXCLUDE_LAYOUTS)]

    # 2. Filter for Commander legal cards
    df = df[df["legalities"].apply(lambda l: isinstance(l, dict) and l.get("commander") == "legal")]

    # 3. Remove duplicates
    df = df.drop_duplicates(subset='oracle_id', keep='last')

    # --- TEXT CLEANING START ---
    print("Cleaning text encodings (this may take a moment)...")

    # Helper: Fixes text using ftfy
    def clean_val(val):
        if isinstance(val, str):
            # fix_text repairs 'ù' -> '—' and other encoding errors
            return ftfy.fix_text(val) 
        return val

    # Helper: Specifically handles the 'card_faces' list of dictionaries
    def clean_faces(faces_list):
        if not isinstance(faces_list, list):
            return faces_list
        
        cleaned_list = []
        for face in faces_list:
            if isinstance(face, dict):
                # We copy the face dict to avoid SettingWithCopy warnings
                new_face = face.copy()
                # Clean specific text fields inside the face
                for key in ["name", "type_line", "oracle_text", "flavor_text"]:
                    if key in new_face:
                        new_face[key] = clean_val(new_face[key])
                cleaned_list.append(new_face)
            else:
                cleaned_list.append(face)
        return cleaned_list

    # Apply cleaning to main columns
    text_cols = ["name", "type_line", "oracle_text"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_val)

    # Apply cleaning to the nested card_faces column
    if "card_faces" in df.columns:
        df["card_faces"] = df["card_faces"].apply(clean_faces)
    
    # --- TEXT CLEANING END ---

    useful_columns = [
        "oracle_id", "name", "card_faces", "mana_cost", "cmc", "type_line", "oracle_text",
        "power", "toughness", "loyalty", "colors", "color_identity",
        "keywords", "produced_mana", "prices", "game_changer", "edhrec_rank"
    ]

    # Filter columns safely (only keep ones that actually exist)
    existing_cols = [c for c in useful_columns if c in df.columns]
    
    print("Dataset has been filtered and cleaned.")
    return df[existing_cols].copy()