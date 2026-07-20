# scripts/debug_batch.py
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

# adjust path if needed to import local modules
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from training.dataset import ASLDataset
from model.st_pose_model import STPoseModel
from utils.skeleton import build_skeleton_adjacency

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

ds = ASLDataset(processed_dir="data/old_processed", wsasl_json="data/WLASL_v0.3.json", split="train")
print("dataset size:", len(ds), "num_joints:", ds.num_joints, "num_classes:", ds.num_classes)

loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

# build model consistent with training
adj = build_skeleton_adjacency(num_joints=ds.num_joints)
model = STPoseModel(adj=adj, num_classes=ds.num_classes, hidden_dim=128, num_heads=4, T=30).to(device)
model.eval()

# fetch one batch
batch = next(iter(loader))
# batch may be ((x, mask), y)
if isinstance(batch[0], (list, tuple)):
    (x, mask), y = batch
else:
    x, y = batch
    mask = None

print("x dtype:", x.dtype, "shape:", x.shape)
if mask is not None:
    print("mask dtype:", mask.dtype, "shape:", mask.shape)
print("y dtype:", y.dtype, "shape:", y.shape)
print("x min/max/mean/std:", float(x.min()), float(x.max()), float(x.mean()), float(x.std()))
print("x has NaN:", bool(torch.isnan(x).any()), "has Inf:", bool(torch.isinf(x).any()))
print("y min/max:", int(y.min()), int(y.max()))

x = x.to(device)
if mask is not None:
    mask = mask.to(device)
y = y.to(device)

# forward pass checks
with torch.no_grad():
    logits = model(x.to(device), mask)
print("logits shape:", logits.shape)
print("logits min/max/mean/std:", float(logits.min()), float(logits.max()), float(logits.mean()), float(logits.std()))
print("logits has NaN:", bool(torch.isnan(logits).any()), "has Inf:", bool(torch.isinf(logits).any()))

# compute loss and try backward with anomaly detection if forward looks OK
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

logits = model(x.to(device), mask)
if torch.isnan(logits).any() or torch.isinf(logits).any():
    print("Bad logits detected; aborting backward.")
else:
    loss = criterion(logits, y)
    print("loss:", float(loss))
    try:
        # run backward under anomaly detection to get a traceback if it errors
        with torch.autograd.detect_anomaly():
            optimizer.zero_grad()
            loss.backward()
            print("backward completed (no anomaly).")
    except Exception as e:
        print("Anomaly during backward:", repr(e))