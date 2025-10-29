import json
import pandas as pd

def filter_dataset(input_path) -> pd.DataFrame:
    with open(input_path, "r", encoding="utf-8") as f:
        cards = json.load(f)

    df = pd.DataFrame(cards)

    # For now only legal cards in commander
    df = df[df["legalities"].apply(lambda l: isinstance(l, dict) and l.get("commander") == "legal")]

    # Remove duplicates
    df = df.drop_duplicates(subset='oracle_id', keep='first')

    # TODO: handle double sided cards

    useful_columns = [
        "oracle_id", "name", "mana_cost", "cmc", "type_line", "oracle_text",
        "power", "toughness", "loyalty", "colors", "color_identity",
        "keywords", "produced_mana", "prices", "game_changer", "edhrec_rank"
    ]

    print("Dataset has been filtered.")
    return df[useful_columns].copy()
