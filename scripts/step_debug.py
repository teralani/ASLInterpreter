import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from training.dataset import ASLDataset
from model.st_pose_model import STPoseModel
from utils.skeleton import build_skeleton_adjacency
from model.spatial_attention import SpatialAttentionWithBias
from utils.velocity import add_velocity

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

ds = ASLDataset(processed_dir="data/processed", wsasl_json="data/WLASL_v0.3.json", split="train")
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
batch = next(iter(loader))
x, y = batch
B, T, J, C = x.shape
print("batch x shape:", x.shape, "y shape:", y.shape)
print("x finite:", torch.isfinite(x).all().item(),
      "min/max/mean/std:", float(x.min()), float(x.max()), float(x.mean()), float(x.std()))

x = x.to(device)
y = y.to(device)

# step 1: add_velocity
x1 = add_velocity(x)
print("after add_velocity finite:", torch.isfinite(x1).all().item(),
      "min/max/mean/std:", float(x1.min()), float(x1.max()), float(x1.mean()), float(x1.std()))

# step 2: embed
model = STPoseModel(adj=build_skeleton_adjacency(num_joints=ds.num_joints), num_classes=ds.num_classes).to(device)
embed = model.embed
x2 = embed(x1)
print("after embed shape:", x2.shape, "finite:", torch.isfinite(x2).all().item(),
      "min/max/mean/std:", float(x2.min()), float(x2.max()), float(x2.mean()), float(x2.std()))

# step 3: spatial (call attn directly to capture weights)
B, T, J, D = x2.shape
x3 = x2.view(B*T, J, D)
sp = model.spatial
bias = sp.bias.to(dtype=x3.dtype, device=x3.device)
out_sp, attn_w = sp.attn(x3, x3, x3, attn_mask=bias)
print("spatial out shape:", out_sp.shape, "finite:", torch.isfinite(out_sp).all().item(),
      "min/max/mean/std:", float(out_sp.min()), float(out_sp.max()), float(out_sp.mean()), float(out_sp.std()))
print("spatial attn_w shape:", attn_w.shape, "finite:", torch.isfinite(attn_w).all().item(),
      "attn_w min/max/mean/std:", float(attn_w.min()), float(attn_w.max()), float(attn_w.mean()), float(attn_w.std()))

# reshape and pool
x4 = out_sp.view(B, T, J, D)
x5 = x4.mean(dim=2)
print("after joint pool shape:", x5.shape, "finite:", torch.isfinite(x5).all().item(),
      "min/max/mean/std:", float(x5.min()), float(x5.max()), float(x5.mean()), float(x5.std()))

# pos add
x6 = x5 + model.pos[:, :T].to(device)
print("after pos add finite:", torch.isfinite(x6).all().item(),
      "min/max/mean/std:", float(x6.min()), float(x6.max()), float(x6.mean()), float(x6.std()))

# temporal attention
valid_frames = (x1.abs().sum(dim=(2,3))> 0)  # computed on x1
key_padding_mask = ~valid_frames.to(device)
print("valid_frames sum per-batch:", valid_frames.sum(dim=1).tolist())
temp_out, temp_attn_w = model.temporal(x6, x6, x6, key_padding_mask=key_padding_mask)
print("temporal out shape:", temp_out.shape, "finite:", torch.isfinite(temp_out).all().item(),
      "min/max/mean/std:", float(temp_out.min()), float(temp_out.max()), float(temp_out.mean()), float(temp_out.std()))
print("temporal attn_w shape:", temp_attn_w.shape if temp_attn_w is not None else None,
      "finite:", torch.isfinite(temp_attn_w).all().item() if temp_attn_w is not None else None)

# aggregate and classifier
valid_frames_u = valid_frames.unsqueeze(-1).to(device)
x_final = (temp_out * valid_frames_u).sum(dim=1)
x_final = x_final / valid_frames_u.sum(dim=1).clamp(min=1)
print("x_final finite:", torch.isfinite(x_final).all().item(),
      "min/max/mean/std:", float(x_final.min()), float(x_final.max()), float(x_final.mean()), float(x_final.std()))

logits = model.cls(x_final)
print("logits finite:", torch.isfinite(logits).all().item(),
      "min/max/mean/std:", float(logits.min()), float(logits.max()), float(logits.mean()), float(logits.std()))