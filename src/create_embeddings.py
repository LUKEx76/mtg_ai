import json
import pandas as pd
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

def create_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    model = SentenceTransformer(model_name_or_path='Alibaba-NLP/gte-modernbert-base', trust_remote_code=True)

    print("Expanding card faces...")
    expanded_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        expanded_rows.extend(process_card_row(row=row))
    
    expanded_df = pd.DataFrame(data=expanded_rows)

    print("Generating JSON inputs...")
    def row_to_clean_json(row: pd.Series) -> str:
        data = {
            "rule_text": row['oracle_text'],
            "type": row['type_line'],
            "mana_cost": row['mana_cost'],
            "loyalty": row['loyalty'],
            "toughness": row['toughness'],
            "power": row['power'],
            "cmc": row['cmc'],
        }
        return clean_and_serialize(row_dict=data)

    json_inputs = expanded_df.apply(func=row_to_clean_json, axis=1).tolist()

    print(f"Embedding {len(json_inputs)} faces...")
    embeddings = model.encode(sentences=json_inputs, batch_size=32, show_progress_bar=True)

    expanded_df['embedding'] = list(embeddings)
    expanded_df['json_input'] = json_inputs
    
    return expanded_df

def process_card_row(row: pd.Series) -> List[Dict[str, Any]]:
    faces_data: List[Dict[str, Any]] = []

    if 'card_faces' not in row or not isinstance(row['card_faces'], list):
        faces_data.append({
            "oracle_id": row.get('oracle_id'),
            "name": row['name'],
            "parent_name": row['name'],
            "mana_cost": row.get('mana_cost', ''),
            "cmc": row.get('cmc', ''),
            "type_line": row.get('type_line', ''),
            "oracle_text": clean_mtg_text(text=row.get('oracle_text', ''), card_name=row['name']),
            "power": row.get('power', ''),
            "toughness": row.get('toughness', ''),
            "loyalty": row.get('loyalty', ''),
            "colors": row.get('colors', ''),
            "color_identity": row.get('color_identity', ''),
            "face_type": "single"
        })
        return faces_data

    for face in row['card_faces']:
        faces_data.append({
            "oracle_id": row.get('oracle_id'),
            "name": face.get('name', row['name']), 
            "parent_name": row['name'],
            "mana_cost": face.get('mana_cost', ''),
            "cmc": row.get('cmc', ''),
            "type_line": face.get('type_line', row.get('type_line', '')),
            "oracle_text": clean_mtg_text(text=face.get('oracle_text', ''), card_name=face.get('name', row['name'])),
            "power": face.get('power', ''),
            "toughness": face.get('toughness', ''),
            "loyalty": face.get('loyalty', row.get('loyalty', '')),
            "colors": row.get('colors', ''),
            "color_identity": row.get('color_identity', ''),
            "face_type": "multi"
        })
        
    return faces_data

def clean_mtg_text(text: str, card_name: str) -> str:
    if not text or not isinstance(text, str): 
        return ""
    
    # remove reminder text
    text = re.sub(pattern=r'\([^)]*\)', repl='', string=text)
    
    # normalize self-reference
    if card_name:
        names_to_replace = [card_name]
        if "," in card_name:
            names_to_replace.append(card_name.split(",")[0])
            
        for name in names_to_replace:
            text = re.sub(pattern=re.escape(string=name), repl='CARDNAME', string=text, flags=re.IGNORECASE)

    # remove transform keywords
    keywords_to_strip = ["Daybound", "Nightbound", "Transform"] 
    for k in keywords_to_strip:
        text = text.replace(k, "")
        
    # fix whitespaces
    text = re.sub(pattern=r' +', repl=' ', string=text)
    
    return text.strip()

def clean_and_serialize(row_dict: Dict[str, Any]) -> str:
    """removes empty fields and converts floats to ints"""
    clean_data: Dict[str, Any] = {}
    
    for key, value in row_dict.items():
        # skip empty values
        if pd.isna(value) or value is None or value == "":
            continue
            
        # convert floats to ints
        if isinstance(value, float) and value.is_integer():
            clean_data[key] = int(value)
        else:
            clean_data[key] = value
            
    return json.dumps(obj=clean_data)