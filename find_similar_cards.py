import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

def find_similar_cards(card_name: str, top_n: int = 5) -> None:
    df: pd.DataFrame = pd.read_parquet(path="data/cards_with_embeddings.parquet")

    if card_name not in df['name'].values:
        print(f"Card '{card_name}' not found in dataset.")
        return

    row: pd.Series = df[df['name'] == card_name].iloc[0]
    vec: np.ndarray = row['embedding'].reshape(1, -1)

    all_vecs: np.ndarray = np.vstack(tup=df['embedding'].to_numpy())
    sims: np.ndarray = cosine_similarity(X=vec, Y=all_vecs)[0]

    result_df: pd.DataFrame = df.assign(sim=sims).sort_values(by='sim', ascending=False).head(n=top_n)
    print(result_df[['name', 'sim', 'oracle_text']])

if __name__ == "__main__":
    target_card: str = "Goblin King"
    find_similar_cards(card_name=target_card, top_n=5)
