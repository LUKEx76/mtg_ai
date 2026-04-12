import pandas as pd
import ftfy
from typing import List, Dict, Any, Optional

INPUT_PATH: str = "data/scryfall/oracle-cards-20251022090300.json"

EXCLUDE_LAYOUTS: set[str] = {
    "art_series", "token", "double_faced_token", "emblem", "scheme", 
    "plane", "planar", "phenomenon", "vanguard", "reversible_card"
}

def clean_text_encoding(text: Optional[str]) -> Optional[str]:
    if isinstance(text, str):
        # repair encoding errors
        return ftfy.fix_text(text=text) 
    return text

def clean_card_faces(faces_list: Any) -> Any:
    if not isinstance(faces_list, list):
        return faces_list
    
    cleaned_list: List[Dict[str, Any]] = []
    for face in faces_list:
        if not isinstance(face, dict):
            cleaned_list.append(face)
            continue

        # copy face to avoid warnings
        new_face: Dict[str, Any] = face.copy()
        for key in ["name", "type_line", "oracle_text", "flavor_text"]:
            if key in new_face:
                new_face[key] = clean_text_encoding(text=new_face[key])
        cleaned_list.append(new_face)
    return cleaned_list

def extract_large_image_uri(row: pd.Series) -> Optional[str]:
    """extract large image uri"""
    # check single-faced cards
    image_uris = row.get("image_uris")
    if isinstance(image_uris, dict):
        return image_uris.get("large")
    
    # check double-faced cards
    card_faces = row.get("card_faces")
    if isinstance(card_faces, list) and len(card_faces) > 0:
        front_face: Any = card_faces[0]
        if isinstance(front_face, dict):
            face_image_uris = front_face.get("image_uris")
            if isinstance(face_image_uris, dict):
                return face_image_uris.get("large")
            
    return None

def main() -> None:
    # load data
    df: pd.DataFrame = pd.read_json(path_or_buf=INPUT_PATH)

    # filter rows
    if "layout" in df.columns:
        df = df[~df["layout"].isin(element=EXCLUDE_LAYOUTS)]

    df = df[df["legalities"].apply(func=lambda l: isinstance(l, dict) and l.get("commander") == "legal")]

    # remove duplicates
    df = df.drop_duplicates(subset='oracle_id', keep='first')

    # clean text
    text_cols: List[str] = ["name", "type_line", "oracle_text"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(func=clean_text_encoding)

    if "card_faces" in df.columns:
        df["card_faces"] = df["card_faces"].apply(func=clean_card_faces)

    # extract images
    df["image_uri"] = df.apply(func=extract_large_image_uri, axis=1)

    # drop missing images
    df = df.dropna(subset=["image_uri"])

    # export
    useful_columns: List[str] = ["oracle_id", "name", "image_uri", "color_identity"]
    df_done: pd.DataFrame = df[useful_columns].copy(deep=True)

    df_done.to_excel(excel_writer="commander_legal_pixel_art.xlsx", index=False)
    print("Exported commander legal pixel art data.")

if __name__ == "__main__":
    main()
