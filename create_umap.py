import umap.umap_ as umap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

N_NEAREST_NEIGHBORS = 15
MIN_DISTANCE = 0.1
METRIC = 'cosine'
SEED = 1893

df = pd.read_parquet("data/cards_with_embeddings.parquet")
print(f"Extracting embeddings for {len(df)} cards")

embedding_matrix = np.stack(df['embedding'].values)
print("Running UMAP projection")

# metric='cosine': Best for semantic embeddings (BERT models)
# n_neighbors: Controls cluster size (Higher = focus on global structure, Lower = focus on local clumps)
# min_dist: Controls how tightly points are packed (Lower = tighter clumps)
reducer = umap.UMAP(
    n_components=2, 
    n_neighbors=N_NEAREST_NEIGHBORS, 
    min_dist=MIN_DISTANCE,
    metric=METRIC, 
    random_state=SEED,
    verbose=True
)

umap_coords = reducer.fit_transform(embedding_matrix)

df['x'] = umap_coords[:, 0]
df['y'] = umap_coords[:, 1]

# Normalize coordinates to a specific range (e.g., -100 to 100)
scaler = StandardScaler()
scaled_coords = scaler.fit_transform(df[['x', 'y']])
df['x'] = scaled_coords[:, 0] * 100
df['y'] = scaled_coords[:, 1] * 100

# Compross data
df['x'] = df['x'].round(4)
df['y'] = df['y'].round(4)

print("UMAP coordinates attached.")
df.to_parquet("cards_with_umap_coords.parquet", index=False)