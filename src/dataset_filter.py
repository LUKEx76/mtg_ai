import pandas as pd
import ftfy
from typing import List, Dict, Any, Optional

# layouts to exclude
EXCLUDE_LAYOUTS = {
    "art_series", "token", "double_faced_token", "emblem", "scheme", 
    "plane", "planar", "phenomenon", "vanguard", "reversible_card"
}

USEFUL_COLUMNS = [
        "oracle_id", "name", "card_faces", "mana_cost", "cmc", "type_line", "oracle_text", "power", 
        "toughness", "loyalty", "colors", "color_identity", "keywords", "game_changer", "set"
]

def clean_text_encoding(text: Optional[str]) -> Optional[str]:
    if isinstance(text, str):
        # repair encoding errors
        return ftfy.fix_text(text=text) 
    return text

def clean_card_faces(faces_list: Any) -> Any:
    if not isinstance(faces_list, list):
        return faces_list
    
    cleaned_list = []
    for face in faces_list:
        if not isinstance(face, dict):
            cleaned_list.append(face)
            continue

        new_face = {key: face[key] for key in USEFUL_COLUMNS if key in face}
        
        for key in ["name", "type_line", "oracle_text", "flavor_text"]:
            if key in new_face:
                new_face[key] = clean_text_encoding(text=new_face[key])
                
        cleaned_list.append(new_face)
            
    return cleaned_list

def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("Loading dataset...")

    if "layout" in df.columns:
        df = df[~df["layout"].isin(element=EXCLUDE_LAYOUTS)]

    df = df[df["legalities"].apply(func=lambda l: isinstance(l, dict) and l.get("commander") == "legal")]

    df = df.drop_duplicates(subset='oracle_id', keep='last')

    print("Cleaning text encodings...")
    text_cols = ["name", "type_line", "oracle_text"]
    for col in text_cols:
        df[col] = df[col].apply(func=clean_text_encoding)

    if "card_faces" in df.columns:
        df["card_faces"] = df["card_faces"].apply(func=clean_card_faces)

    print("Dataset has been filtered and cleaned.")
    return df[USEFUL_COLUMNS].copy(deep=True).reset_index(drop=True)
