import torch
import torch.nn as nn


class MTGFusionModel(nn.Module):
    """ 
    late fusion model for sentence embeddings and tabular features
    only train the tabular encoder and fusion head, sentence embeddings stay untouched
    """

    def __init__(self, text_dim=768, tabular_dim=20, hidden_dim=512, bottleneck_dim=256):
        super(MTGFusionModel, self).__init__()
        
        # Tabular encoder
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fusion layers
        fusion_input_dim = text_dim + 64
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, 1)
        )
        
    def forward(self, text_embeddings, tabular_features):
        tab_out = self.tabular_encoder(tabular_features)
        combined = torch.cat((text_embeddings, tab_out), dim=1)
        out = self.fusion_head(combined)
        return out.squeeze()

