import torch
import numpy as np

from src.embeddings.mtg_fusion_model import MTGFusionModel
from src.training import config
from src.training import data_prep
from src.training import trainer

INPUT_DATA_PATH = "data/training/card_strength_input.parquet"
LABEL_DATA_PATH = "data/training/card_strength_metric.xlsx"
OUTPUT_MODEL_PATH = "src/best_fusion_model.pth"
SEED = 1893
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    df = data_prep.load_and_prep_data(input_path=config.INPUT_DATA_PATH, label_path=config.LABEL_DATA_PATH)
    embedding_cols, numerical_cols, boolean_cols = data_prep.get_feature_columns(df=df)
    
    train_loader, val_loader, _, _, _ = data_prep.create_dataloaders(
        df=df, 
        embedding_cols=embedding_cols, 
        numerical_cols=numerical_cols, 
        boolean_cols=boolean_cols, 
        target_col='log_power_score', 
        seed=config.SEED
    )
    
    tabular_dim = len(numerical_cols) + len(boolean_cols)

    model = MTGFusionModel(
        text_dim=len(embedding_cols), 
        tabular_dim=tabular_dim
    ).to(config.DEVICE)

    criterion = torch.nn.HuberLoss(delta=3.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)

    print("Training MTG Fusion Model...")
    best_model, history = trainer.train_torch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=config.DEVICE,
        epochs=config.EPOCHS,
    )
    
    print(f"Training complete. Best Val Loss: {min(history['val_loss']):.4f}")
    
    torch.save(best_model.state_dict(), config.OUTPUT_MODEL_PATH)
    print(f"Model saved to {config.OUTPUT_MODEL_PATH}")

if __name__ == '__main__':
    main()
