import pandas as pd
import numpy as np

df = pd.read_parquet("cards_commander.parquet")

print(f"Loaded cleaned MTG card set {df.shape}")

display_columns = [ "name", "mana_cost", "type_line", "power", "toughness", "oracle_text" ]
short_display_columns = ["name", "keyword_count", "keywords"]

print(df.columns)

# Quick look
# print(df.sample(5)[display_columns])

# Most keywords
df['keyword_count'] = df['keywords'].apply(
    lambda k: len(k) if isinstance(k, np.ndarray) else 0
)

top10_keywords = df.nlargest(10, 'keyword_count')

print(top10_keywords[short_display_columns])
