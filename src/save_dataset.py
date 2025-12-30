import pandas as pd
import os

def save_dataset(df: pd.DataFrame, filepath: str):
    df.to_parquet(filepath, index=False)
    print(f"Successfully saved dataframe to {filepath}")
