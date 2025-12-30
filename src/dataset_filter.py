import pandas as pd
import ftfy

# Layouts to exclude outright
EXCLUDE_LAYOUTS = {
    "art_series", "token", "double_faced_token", "emblem", "scheme", 
    "plane", "planar", "phenomenon", "vanguard", "reversible_card"
}

def clean_text_encoding(text:str):
    if isinstance(text, str):
        # fix_text repairs 'ù' -> '—' and other encoding errors
        return ftfy.fix_text(text) 
    return text

def clean_card_faces(faces_list):
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
                    new_face[key] = clean_text_encoding(new_face[key])
            cleaned_list.append(new_face)
        else:
            cleaned_list.append(face)
    return cleaned_list

def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("Loading dataset...")

    # Exclude layouts
    if "layout" in df.columns:
        df = df[~df["layout"].isin(EXCLUDE_LAYOUTS)]

    # Filter for Commander legal cards
    df = df[df["legalities"].apply(lambda l: isinstance(l, dict) and l.get("commander") == "legal")]

    # Remove duplicates
    df = df.drop_duplicates(subset='oracle_id', keep='last')

    print("Cleaning text encodings")

    # Apply cleaning to main columns
    text_cols = ["name", "type_line", "oracle_text"]
    for col in text_cols:
        df[col] = df[col].apply(clean_text_encoding)

    # Apply cleaning to the nested card_faces column
    if "card_faces" in df.columns:
        df["card_faces"] = df["card_faces"].apply(clean_card_faces)
    
    useful_columns = [
        "oracle_id", "name", "card_faces", "mana_cost", "cmc", "type_line", "oracle_text",
        "power", "toughness", "loyalty", "colors", "color_identity",
        "keywords", "produced_mana", "prices", "game_changer", "edhrec_rank"
    ]

    print("Dataset has been filtered and cleaned.")
    return df[useful_columns].copy()
