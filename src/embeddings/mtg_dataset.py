import torch
from torch.utils.data import Dataset

class MTGDataset(Dataset):
    def __init__(self, embeddings, tabular, targets):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.tabular = torch.tensor(tabular, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.tabular[idx], self.targets[idx]
    