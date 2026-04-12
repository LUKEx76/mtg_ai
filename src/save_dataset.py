import pandas as pd
import os

def save_dataset(df: pd.DataFrame, filepath: str) -> None:
    df.to_parquet(path=filepath, index=False)
    print(f"Successfully saved dataframe to {filepath}")
