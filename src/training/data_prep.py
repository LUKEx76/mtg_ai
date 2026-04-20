import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.embeddings.mtg_dataset import MTGDataset
from src.training import config

def load_and_prep_data(input_path: str, label_path: str):
    df_input = pd.read_parquet(input_path)
    df_label = pd.read_excel(label_path)

    cols_to_use = df_label.columns.difference(df_input.columns).tolist()
    df = df_input.merge(df_label[cols_to_use + ["oracle_id"]], on="oracle_id", how="left")

    df = df.dropna(subset=["log_power_score"])
    df = df[np.isfinite(df["log_power_score"])]
    return df

def get_feature_columns(df: pd.DataFrame):
    embedding_cols = [col for col in df.columns if col.startswith('embed_')]
    numerical_cols = [col for col in df.columns if col.startswith('num_')]
    boolean_cols = [col for col in df.columns if col.startswith('bool_')]
    return embedding_cols, numerical_cols, boolean_cols

def create_dataloaders(df: pd.DataFrame, embedding_cols: list, numerical_cols: list, boolean_cols: list, target_col: str, seed: int):
    text_embeddings = df[embedding_cols].values
    tabular_features = df[numerical_cols + boolean_cols].fillna(-1).values

    scaler = StandardScaler()
    tabular_features = scaler.fit_transform(tabular_features)

    targets = df[target_col].values

    X_text_train_val, X_text_test, X_tab_train_val, X_tab_test, y_train_val, y_test = train_test_split(
        text_embeddings, tabular_features, targets, test_size=0.20, random_state=seed
    )

    X_text_train, X_text_val, X_tab_train, X_tab_val, y_train, y_val = train_test_split(
        X_text_train_val, X_tab_train_val, y_train_val, test_size=0.25, random_state=seed
    )

    train_ds = MTGDataset(X_text_train, X_tab_train, y_train)
    val_ds = MTGDataset(X_text_val, X_tab_val, y_val)
    test_ds = MTGDataset(X_text_test, X_tab_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)
    
    datasets = {
        "train": (X_text_train, X_tab_train, y_train),
        "val": (X_text_val, X_tab_val, y_val),
        "test": (X_text_test, X_tab_test, y_test)
    }

    return train_loader, val_loader, test_loader, datasets, scaler
