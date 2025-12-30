import argparse
import os
from src import dataset_filter, save_dataset, create_modern_bert_embeddings

def process_dataset(input_path):
    df = dataset_filter.filter_dataset(input_path)
    df = create_modern_bert_embeddings.create_modern_bert_embeddings(df)
    save_dataset.save_dataset(df)

def main():
    parser = argparse.ArgumentParser(
        description="Process a dataset JSON and save a new file."
    )
    parser.add_argument(
        "input_json",
        help="Path to the input scryfall JSON dataset",
    )

    args = parser.parse_args()
    input_path = args.input_json

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Processing {input_path} ...")
    process_dataset(input_path)
    print("Done.")

if __name__ == "__main__":
    main()
