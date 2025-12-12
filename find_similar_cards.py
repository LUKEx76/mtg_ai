import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DIMENSIONS = 64
card_name = "Avacyn, Angel of Hope"

df = pd.read_parquet("data/cards_with_embeddings.parquet")

# pick a card
row = df[df['name'] == card_name].iloc[0]
vec = row[f'embedding_{DIMENSIONS}'].reshape(1, -1)

# compare against all cards
all_vecs = np.vstack(df[f'embedding_{DIMENSIONS}'].to_numpy())
sims = cosine_similarity(vec, all_vecs)[0]

# top 5 similar cards (excluding itself)
print(df.assign(sim=sims).sort_values('sim', ascending=False).head(5)[['name', 'sim', 'oracle_text_clean']])