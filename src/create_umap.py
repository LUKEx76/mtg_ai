import umap.umap_ as umap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

N_NEAREST_NEIGHBORS = 15
MIN_DISTANCE = 0.1
METRIC = 'cosine'
SEED = 1893

def create_umap(df: pd.DataFrame) -> pd.DataFrame:
    print("Running UMAP projection")
    embedding_matrix = np.stack(arrays=df['embedding'].values)

    # use cosine metric for bert
    # controls cluster size
    # controls packing density
    reducer = umap.UMAP(
        n_components=2, 
        n_neighbors=N_NEAREST_NEIGHBORS, 
        min_dist=MIN_DISTANCE,
        metric=METRIC, 
        random_state=SEED,
        verbose=True
    )

    umap_coords = reducer.fit_transform(X=embedding_matrix)
    df['x'] = umap_coords[:, 0]
    df['y'] = umap_coords[:, 1]

    # normalize coordinates
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(X=df[['x', 'y']])
    df['x'] = scaled_coords[:, 0] * 100
    df['y'] = scaled_coords[:, 1] * 100

    df['x'] = df['x'].round(decimals=4)
    df['y'] = df['y'].round(decimals=4)

    print(f"Max x value of map {df['x'].max()}")
    print(f"Min x value of map {df['x'].min()}")
    print(f"Max y value of map {df['y'].max()}")
    print(f"Min y value of map {df['y'].min()}")

    print("UMAP coordinates attached.")
    return df
