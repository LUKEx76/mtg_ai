import json
import pandas as pd

FACE_JOINER = "\n//\n"

# Layouts that actually store rules text across faces and should be concatenated
FACE_LAYOUTS = {
    "transform",
    "modal_dfc",
    "split",
    "flip",
    "adventure",
    "meld",
}

# Layouts to exclude outright (not in your list, but keep for safety)
EXCLUDE_LAYOUTS = {
    "art_series",
    "token",
    "double_faced_token",
    "emblem",
    "scheme",
    "plane",
    "planar",
    "phenomenon",
    "vanguard",
    "reversible_card",
}

def filter_dataset(input_path) -> pd.DataFrame:
    with open(input_path, "r", encoding="utf-8") as f:
        cards = json.load(f)

    df = pd.DataFrame(cards)

    # For now only legal cards in commander
    df = df[df["legalities"].apply(lambda l: isinstance(l, dict) and l.get("commander") == "legal")]

    # Remove duplicates
    df = df.drop_duplicates(subset='oracle_id', keep='first')

    # Exclude layouts outright (art series, tokens, emblems, planes, etc.)
    if "layout" in df.columns:
        df = df[~df["layout"].isin(EXCLUDE_LAYOUTS)]

    # Handle double-sided / multi-face cards:
    df["oracle_text"] = df.apply(_combined_oracle_text, axis=1)

    useful_columns = [
        "oracle_id", "name", "mana_cost", "cmc", "type_line", "oracle_text",
        "power", "toughness", "loyalty", "colors", "color_identity",
        "keywords", "produced_mana", "prices", "game_changer", "edhrec_rank"
    ]

    print("Dataset has been filtered.")
    return df[useful_columns].copy()


def _clean_str(x):
    return x.strip() if isinstance(x, str) else ""

def _face_oracle(face: dict) -> str:
    return _clean_str(face.get("oracle_text", "")) if isinstance(face, dict) else ""

def _join_faces(faces) -> str:
    if not isinstance(faces, list) or not faces:
        return ""
    parts = []
    for f in faces:
        ot = _face_oracle(f)
        if ot:
            parts.append(ot)
    return FACE_JOINER.join(parts) if parts else ""

def _combined_oracle_text(row) -> str:
    layout = row.get("layout")
    top = _clean_str(row.get("oracle_text", ""))
    faces = row.get("card_faces") or []
    joined = _join_faces(faces)

    if layout in FACE_LAYOUTS:
        return joined if joined else top
    else:
        return top if top else joined
    