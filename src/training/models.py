import torch.nn as nn

class EmbeddingLinearModel(nn.Module):
    def __init__(self, text_dim: int):
        super(EmbeddingLinearModel, self).__init__()
        self.linear = nn.Linear(in_features=text_dim, out_features=1)

    def forward(self, text_embeddings, _):
        out = self.linear(text_embeddings)
        return out.squeeze(dim=-1)
