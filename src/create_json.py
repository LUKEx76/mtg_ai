import pandas as pd
from typing import List

def clean_color(c_list: List[str]) -> str:
    if not isinstance(c_list, list) or len(c_list) == 0:
        return "C" # colorless
    return "".join(iterable=c_list)

def create_json(df: pd.DataFrame, output_path: str) -> None:
    df["x"] = df["x"].round(decimals=4)
    df["y"] = df["y"].round(decimals=4)
    df["ci"] = df["color_identity"].apply(func=clean_color)
    df["cmc"] = df["cmc"].astype(dtype=int)

    # rename to short columns
    final_df: pd.DataFrame = df.rename(columns={
        "name": "n",
        "type_line": "t",
        "oracle_id": "oid"
    })

    cols_to_keep: List[str] = ["n", "x", "y", "ci", "cmc", "t", "oid"]
    final_df = final_df[cols_to_keep].dropna(subset=["x", "y"])

    final_df.to_json(path_or_buf=output_path, orient="records")

    print(f"Exported {len(final_df)} cards to {output_path}")
