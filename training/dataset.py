import numpy as np
import torch
from torch.utils.data import Dataset

class ASLDataset(Dataset):
    def __init__(self, data_dir:str, labels):
        self.files = list(data_dir.glob("*.npy"))
        self.labels = labels
    
    def __getitem__(self, idx):
        x = np.load(self.files[idx])
        y = self.labels[self.files[idx].stem]
        return torch.tensor(x).float(), torch.tensor(y)
    
    def __len__(self):
        return len(self.files)