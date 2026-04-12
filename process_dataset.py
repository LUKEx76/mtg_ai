import argparse
import os
import pandas as pd
from src import dataset_filter, save_dataset, create_embeddings, create_umap, create_json

def process_dataset(df: pd.DataFrame) -> None:
    df = dataset_filter.filter_dataset(df=df)
    df = create_embeddings.create_embeddings(df=df)
    df = create_umap.create_umap(df=df)
    create_json.create_json(df=df, output_path="mtg_map.json")
    save_dataset.save_dataset(df=df, filepath="mtg_cards_processed.parquet")

def main() -> None:
    parser = argparse.ArgumentParser(description="Process a dataset JSON and save a new file.")
    parser.add_argument("input_json", help="Path to the input scryfall JSON dataset")

    args = parser.parse_args()
    input_path = args.input_json

    if not os.path.isfile(path=input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Processing {input_path} ...")
    df = pd.read_json(path_or_buf=input_path)
    process_dataset(df=df)
    print("Done.")

if __name__ == "__main__":
    main()
