import json
import pandas as pd
import re
from sentence_transformers import SentenceTransformer

def create_modern_bert_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    model = SentenceTransformer('Alibaba-NLP/gte-modernbert-base', trust_remote_code=True)

    print("Expanding card faces...")
    expanded_rows = []
    for _, row in df.iterrows():
        expanded_rows.extend(process_card_row(row))
    
    expanded_df = pd.DataFrame(expanded_rows)

    print("Generating JSON inputs...")
    
    # TODO: could improve prompt engineering:
    # add low, medium, high mana cost
    # add keywords, ...
    # if want to strip color entirely cut mana_cost, then add something like contains X (to capture spells with X as casting cost)
    def row_to_clean_json(row):
        data = {
            #"name": row['name'], # remove for training
            "rule_text": row['oracle_text'],
            "type": row['type_line'],
            "mana_cost": row['mana_cost'],
            "loyalty": row['loyalty'],
            "toughness": row['toughness'],
            "power": row['power'],
            "cmc": row['cmc'],
        }
        return clean_and_serialize(data)

    json_inputs = expanded_df.apply(row_to_clean_json, axis=1).tolist()

    print(f"Embedding {len(json_inputs)} faces...")
    embeddings = model.encode(json_inputs, batch_size=32, show_progress_bar=True)

    expanded_df['embedding'] = list(embeddings)
    expanded_df['json_input'] = json_inputs
    
    return expanded_df

def process_card_row(row):
    """
    Takes a single row from the original Scryfall DF.
    Returns a LIST of dicts (1 dict for normal cards, 2+ for multi-face).
    """
    faces_data = []

    # Multi-face cards (Transform, Modal, etc.)
    if 'card_faces' in row and isinstance(row['card_faces'], list):
        for face in row['card_faces']:
            faces_data.append({
                "name": face.get('name', row['name']), 
                "parent_name": row['name'],       
                "oracle_id": row.get('oracle_id'),
                "type_line": face.get('type_line', row.get('type_line', '')),
                "oracle_text": clean_mtg_text(face.get('oracle_text', ''), face.get('name', row['name'])),
                "mana_cost": face.get('mana_cost', ''),
                "cmc": row.get('cmc', ''), # CMC is usually on the parent for old DFCs, but safe here
                "colors": row.get('colors', ''),
                "power": face.get('power', ''), # Empty string if missing
                "toughness": face.get('toughness', ''),
                # Check face first for loyalty (Planeswalkers), fallback to row
                "loyalty": face.get('loyalty', row.get('loyalty', '')), 
                "face_type": "multi"
            })
    else:
        # Normal single-faced card
        faces_data.append({
            "name": row['name'],
            "parent_name": row['name'],
            "oracle_id": row.get('oracle_id'),
            "type_line": row.get('type_line', ''),
            "oracle_text": clean_mtg_text(row.get('oracle_text', ''), row['name']),
            "mana_cost": row.get('mana_cost', ''),
            "cmc": row.get('cmc', ''),
            "colors": row.get('colors', ''),
            "power": row.get('power', ''),
            "toughness": row.get('toughness', ''),
            "loyalty": row.get('loyalty', ''),
            "face_type": "single"
        })
        
    return faces_data

def clean_mtg_text(text, card_name):
    if not text or not isinstance(text, str): 
        return ""
    
    # Remove Reminder Text
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Normalize "Self-Reference"
    if card_name:
        names_to_replace = [card_name]
        if "," in card_name:
            names_to_replace.append(card_name.split(",")[0])
            
        for name in names_to_replace:
            text = re.sub(re.escape(name), 'CARDNAME', text, flags=re.IGNORECASE)

    # Remove generic transform keywords (Uncommented and Fixed)
    keywords_to_strip = ["Daybound", "Nightbound", "Transform"] 
    for k in keywords_to_strip:
        text = text.replace(k, "")
        
    # Fix whitespaces
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def clean_and_serialize(row_dict):
    """
    Removes Null/Empty fields and converts floats to ints to save tokens.
    """
    clean_data = {}
    
    for key, value in row_dict.items():
        # Skip NaNs, Nones, and Empty Strings
        if pd.isna(value) or value is None or value == "":
            continue
            
        # Convert 2.0 -> 2 (integers save token space)
        if isinstance(value, float) and value.is_integer():
            clean_data[key] = int(value)
        else:
            clean_data[key] = value
            
    return json.dumps(clean_data)