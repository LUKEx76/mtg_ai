import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.embeddings.mtg_fusion_model import MTGFusionModel
from src.training import config
from src.training import data_prep
from src.training import trainer
from src.training.models import EmbeddingLinearModel

def evaluate_models():
    np.random.seed(seed=config.SEED)
    torch.manual_seed(seed=config.SEED)

    df = data_prep.load_and_prep_data(input_path=config.INPUT_DATA_PATH, label_path=config.LABEL_DATA_PATH)
    embedding_cols, numerical_cols, boolean_cols = data_prep.get_feature_columns(df=df)
    
    train_loader, val_loader, test_loader, datasets, _ = data_prep.create_dataloaders(
        df=df, 
        embedding_cols=embedding_cols, 
        numerical_cols=numerical_cols, 
        boolean_cols=boolean_cols, 
        target_col='log_power_score', 
        seed=config.SEED
    )
    
    X_text_train, X_tab_train, y_train = datasets["train"]
    X_text_val, X_tab_val, y_val = datasets["val"]
    X_text_test, X_tab_test, y_test = datasets["test"]

    # 1. full xgboost baseline
    print("Evaluating Full XGBoost Baseline...")
    X_train_xgb = np.hstack(tup=[X_text_train, X_tab_train])
    X_val_xgb = np.hstack(tup=[X_text_val, X_tab_val])
    X_test_xgb = np.hstack(tup=[X_text_test, X_tab_test])

    xgb_full = XGBRegressor(random_state=config.SEED, device=config.DEVICE)
    xgb_full.fit(X=X_train_xgb, y=y_train, eval_set=[(X_val_xgb, y_val)], verbose=False)
    y_pred_xgb = xgb_full.predict(X=X_test_xgb)

    # 2. tabular-only xgboost baseline
    print("Evaluating Tabular-only XGBoost Baseline...")
    xgb_tab = XGBRegressor(random_state=config.SEED, device=config.DEVICE)
    xgb_tab.fit(X=X_tab_train, y=y_train, eval_set=[(X_tab_val, y_val)], verbose=False)
    y_pred_tab_xgb = xgb_tab.predict(X=X_tab_test)

    # 3. embedding-only linear baseline
    print("Evaluating Embedding Linear Baseline...")
    linear_model = EmbeddingLinearModel(text_dim=len(embedding_cols)).to(device=config.DEVICE)
    linear_optimizer = torch.optim.SGD(params=linear_model.parameters(), lr=0.001)
    linear_criterion = torch.nn.HuberLoss(delta=3.0)
    
    best_linear_model, _ = trainer.train_torch_model(
        model=linear_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=linear_optimizer,
        criterion=linear_criterion,
        device=config.DEVICE,
        epochs=config.EPOCHS,
        verbose=10,
        is_baseline=True
    )
    
    linear_preds = []
    best_linear_model.eval()
    with torch.no_grad():
        for texts, _, _ in test_loader:
            texts = texts.to(device=config.DEVICE)
            outputs = best_linear_model(texts, None)
            linear_preds.extend(outputs.cpu().numpy())
    y_pred_linear = np.array(object=linear_preds)

    # 4. late fusion model
    print("Evaluating Late Fusion Model...")
    fusion_model = MTGFusionModel(
        text_dim=len(embedding_cols), 
        tabular_dim=X_tab_train.shape[1]
    ).to(device=config.DEVICE)
    
    try:
        fusion_model.load_state_dict(state_dict=torch.load(f=config.OUTPUT_MODEL_PATH))
        fusion_model.eval()
        fusion_preds = []
        with torch.no_grad():
            for texts, tabs, _ in test_loader:
                texts, tabs = texts.to(device=config.DEVICE), tabs.to(device=config.DEVICE)
                outputs = fusion_model(texts, tabs)
                fusion_preds.extend(outputs.cpu().numpy())
        y_pred_fusion = np.array(object=fusion_preds)
    except FileNotFoundError:
        print(f"Warning: Fusion model not found at {config.OUTPUT_MODEL_PATH}. skipping.")
        y_pred_fusion = None

    # compare results
    preds = {
        "Full XGBoost": y_pred_xgb,
        "Tabular XGBoost": y_pred_tab_xgb,
        "Embedding Baseline": y_pred_linear
    }
    if y_pred_fusion is not None:
        preds["Late Fusion"] = y_pred_fusion

    metrics = {}
    for name, y_pred in preds.items():
        metrics[name] = {
            "MAE": mean_absolute_error(y_true=y_test, y_pred=y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred)),
            "R2": r2_score(y_true=y_test, y_pred=y_pred)
        }
        print(f"\n{name} Metrics:")
        print(f"  Test MAE: {metrics[name]['MAE']:.4f}")
        print(f"  Test RMSE: {metrics[name]['RMSE']:.4f}")
        print(f"  Test R2: {metrics[name]['R2']:.4f}")

    # plotting
    if y_pred_fusion is not None:
        plot_results(y_test, y_pred_xgb, y_pred_fusion, metrics)

def plot_results(y_test, y_pred_xgb, y_pred_fusion, metrics):
    xgb_r2 = metrics["Full XGBoost"]["R2"]
    fusion_r2 = metrics["Late Fusion"]["R2"]

    plt.figure(figsize=(12, 5))
    
    # 1. prediction vs actual
    plt.subplot(1, 2, 1)
    plt.scatter(x=y_test, y=y_pred_xgb, alpha=0.3, label=f'XGBoost (R2={xgb_r2:.2f})', color='tab:blue')
    plt.scatter(x=y_test, y=y_pred_fusion, alpha=0.3, label=f'Late Fusion (R2={fusion_r2:.2f})', color='tab:orange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel(xlabel='Actual Power Score')
    plt.ylabel(ylabel='Predicted Power Score')
    plt.title(label='Prediction vs Actual')
    plt.legend()

    # 2. residual distribution
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=y_test - y_pred_xgb, label='XGBoost Residuals', fill=True, color='tab:blue')
    sns.kdeplot(data=y_test - y_pred_fusion, label='Late Fusion Residuals', fill=True, color='tab:orange')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel(xlabel='Residual (Actual - Predicted)')
    plt.title(label='Residual Distribution')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 3. metrics comparison
    metrics_df = pd.DataFrame(data={
        'Model': ['XGBoost', 'Late Fusion', 'XGBoost', 'Late Fusion'],
        'Metric': ['MAE', 'MAE', 'RMSE', 'RMSE'],
        'Value': [metrics["Full XGBoost"]["MAE"], metrics["Late Fusion"]["MAE"], 
                  metrics["Full XGBoost"]["RMSE"], metrics["Late Fusion"]["RMSE"]]
    })

    plt.figure(figsize=(8, 5))
    sns.barplot(data=metrics_df, x='Metric', y='Value', hue='Model')
    plt.title(label='Model Comparison: MAE and RMSE')
    plt.show()

if __name__ == '__main__':
    evaluate_models()
