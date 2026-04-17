import torch
import torch.nn as nn


class MTGFusionModel(nn.Module):
    """ 
    Simple baseline linear regression model for concatenated text embeddings 
    and tabular features.
    """

    def __init__(self, text_dim: int, tabular_dim: int):
        super(MTGFusionModel, self).__init__()
        # Single linear layer combining all inputs
        self.regression = nn.Linear(text_dim + tabular_dim, 1)
        
    def forward(self, text_embeddings: torch.Tensor, tabular_features: torch.Tensor) -> torch.Tensor:
        # Concatenate text and tabular features
        combined = torch.cat((text_embeddings, tabular_features), dim=1)
        out = self.regression(combined)
        return out.squeeze(-1)
