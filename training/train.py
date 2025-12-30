import torch
from torch.utils.data import DataLoader
from model.st_pose_model import STPoseModel
from training.dataset import ASLDataset
from utils.skeleton import build_skeleton_adjacency

device = "xpu" if torch.xpu.is_available() else "cuda"

adj = build_skeleton_adjacency()
model = STPoseModel(adj, num_classes=2000).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
criterion = torch.nn.CrossEntropyLoss()

dataset = ASLDataset("data/processed")