import pandas as pd

# Convert list ['R', 'G'] to string "RG"
def clean_color(c_list):
    if not isinstance(c_list, list) or len(c_list) == 0:
        return "C" # Colorless
    return "".join(c_list)

def create_json(df: pd.DataFrame, output_path: str):
    df["x"] = df["x"].round(4)
    df["y"] = df["y"].round(4)
    df["ci"] = df["color_identity"].apply(clean_color)
    df["cmc"] = df["cmc"].astype(int)

    # Rename columns to be short (save bytes)
    final_df = df.rename(columns={
        "name": "n",
        "type_line": "t",
        "oracle_id": "oid"
    })

    cols_to_keep = ["n", "x", "y", "ci", "cmc", "t", "oid"]
    final_df = final_df[cols_to_keep].dropna(subset=["x", "y"])

    final_df.to_json(output_path, orient="records")

    print(f"Exported {len(final_df)} cards to {output_path}")
