import pandas as pd
import ftfy

INPUT_PATH = "data/scryfall/oracle-cards-20251022090300.json"

EXCLUDE_LAYOUTS = {
    "art_series", "token", "double_faced_token", "emblem", "scheme", 
    "plane", "planar", "phenomenon", "vanguard", "reversible_card"
}

def clean_text_encoding(text: str):
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

def extract_large_image_uri(row):
    """Safely extracts the large image URI, checking both single and double-faced cards."""
    # 1. Check standard single-faced cards
    if isinstance(row.get("image_uris"), dict):
        return row["image_uris"].get("large")
    
    # 2. Check double-faced cards (e.g., Transform, MDFC)
    if isinstance(row.get("card_faces"), list) and len(row["card_faces"]) > 0:
        front_face = row["card_faces"][0]
        if isinstance(front_face, dict) and isinstance(front_face.get("image_uris"), dict):
            return front_face["image_uris"].get("large")
            
    # Return None if no image is found
    return None


# --- 1. Load Data ---
df = pd.read_json(INPUT_PATH)

# --- 2. Filter Rows ---
if "layout" in df.columns:
    df = df[~df["layout"].isin(EXCLUDE_LAYOUTS)]

df = df[df["legalities"].apply(lambda l: isinstance(l, dict) and l.get("commander") == "legal")]

# Remove duplicates to get one physical card representation
df = df.drop_duplicates(subset='oracle_id', keep='first')

# --- 3. Clean Text ---
text_cols = ["name", "type_line", "oracle_text"]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_text_encoding)

if "card_faces" in df.columns:
    df["card_faces"] = df["card_faces"].apply(clean_card_faces)

# --- 4. Extract Images ---
# Use apply with axis=1 to pass the entire row to our extraction function
df["image_uri"] = df.apply(extract_large_image_uri, axis=1)

# Drop any weird rows that completely lacked an image
df = df.dropna(subset=["image_uri"])

# --- 5. Export ---
useful_columns = ["oracle_id", "name", "image_uri", "color_identity"]
df_done = df[useful_columns].copy()

df_done.to_excel("commander_legal_pixel_art.xlsx", index=False)