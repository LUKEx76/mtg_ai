import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

card_name = "Kalamax, the Stormsire"

df = pd.read_parquet("data/cards_with_embeddings.parquet")

row = df[df['name'] == card_name].iloc[0]
vec = row[f'embedding'].reshape(1, -1)

# compare against all cards
all_vecs = np.vstack(df[f'embedding'].to_numpy())
sims = cosine_similarity(vec, all_vecs)[0]

print(df.assign(sim=sims).sort_values('sim', ascending=False).head(5)[['name', 'sim', 'oracle_text']])
