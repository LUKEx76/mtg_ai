import pandas as pd

def save_dataset(df:pd.DataFrame):
    df.to_parquet("cards_with_embeddings.parquet", index=False)
