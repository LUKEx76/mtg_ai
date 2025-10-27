import json
import pandas as pd

with open("default-cards-20251022090902.json", "r", encoding="utf-8") as f:
    cards = json.load(f)

df = pd.DataFrame(cards)

# Paper format
df = df[df["games"].apply(lambda g: isinstance(g, list) and "paper" in g)]

# For now only legal cards in commander
df = df[df["legalities"].apply(lambda l: isinstance(l, dict) and l.get("commander") == "legal")]

useful_columns = [
    "name", "mana_cost", "cmc", "type_line", "oracle_text",
    "power", "toughness", "loyalty", "colors", "color_identity",
    "keywords", "set", "set_name", "rarity", "artist",
    "produced_mana", "prices", "game_changer", "edhrec_rank"
]

# Remove duplicates
df = df.drop_duplicates(subset='oracle_id', keep='first')

df_clean = df[useful_columns].copy()

df_clean.to_parquet("cards_commander.parquet", index=False)
print("cleaned commander csv has been exported!")
