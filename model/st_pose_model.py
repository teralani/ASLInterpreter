import torch
import torch.nn as nn
from model.spatial_attention import SpatialAttentionWithBias
from utils.velocity import add_velocity

class STPoseModel(nn.Module):
    def __init__(self, adj, num_classes:int = 2000, hidden_dim:int=128, num_heads:int = 4, T:int = 30):
        super().__init__()
        self.embed = nn.Linear(6, hidden_dim)
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.zeros_(self.embed.bias)

        self.spatial = SpatialAttentionWithBias(hidden_dim, num_heads, adj)

        self.temporal = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        # initialize small positional embeddings to avoid large activations
        self.pos = nn.Parameter(torch.zeros(1, T, hidden_dim))

        self.cls = nn.Linear(hidden_dim, num_classes)
        nn.init.xavier_uniform_(self.cls.weight)
        nn.init.zeros_(self.cls.bias)

    def _assert_finite(self, t: torch.Tensor, name: str):
        if not torch.isfinite(t).all():
            finite_mask = torch.isfinite(t)
            if finite_mask.any():
                vals = t[finite_mask]
                stats = {
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "mean": float(vals.mean()),
                    "std": float(vals.std())
                }
            else:
                stats = {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
            raise RuntimeError(f"Non-finite values detected in {name}: {stats}")

    def forward(self, x, mask=None):
        x = add_velocity(x)
        self._assert_finite(x, "after_add_velocity")
        assert x.shape[-1] == 6, f"Expected 6 channels, got {x.shape[-1]}"

        B, T, J, C = x.shape

        # If mask is provided use it; otherwise infer validity from coordinates
        # mask shape expected: (B, T, J) with 1.0 for observed joints, 0.0 otherwise
        if mask is not None:
            # ensure boolean mask
            valid_joints = (mask > 0.5)
        else:
            valid_joints = (x.abs().sum(dim=-1) > 0)  # (B, T, J)

        # Masking bad inputs/frames (frame valid if any joint observed)
        valid_frames = valid_joints.any(dim=2) if valid_joints.ndim == 3 else valid_joints.any(dim=2)

        # Ensure at least one valid frame per sample to avoid attention over entirely-masked sequences
        # (mark first frame valid if none are valid)
        # clone to avoid accidental in-place issues with upstream tensors
        valid_frames = valid_frames.clone()
        no_valid = (~valid_frames).all(dim=1)
        if no_valid.any():
            valid_frames[no_valid, 0] = True

        key_padding_mask = ~valid_frames

        x = self.embed(x) # (B, T, J, D)
        self._assert_finite(x, "after_embed")
        x = x.view(B*T, J, -1) # (B*T, J, D)
        # Build joint-level key_padding_mask for spatial attention: True where to mask
        if mask is not None:
            joint_kpm = ~valid_joints.view(B*T, J)
        else:
            joint_kpm = ~(x.abs().sum(dim=-1) > 0)

        x = self.spatial(x, key_padding_mask=joint_kpm) # (B*T, J, D)
        self._assert_finite(x, "after_spatial")
        x = x.view(B, T, J, -1)

        # Pooling across Joints
        x = x.mean(dim=2) # (B, T, D)
        self._assert_finite(x, "after_joint_pool")

        # Adds previous positions until timestep
        x = x + self.pos[:, :T]
        self._assert_finite(x, "after_pos_add")

        # Create causal attention mask (prevent attending to future frames)
        device = x.device
        attn_mask = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

        x, _ = self.temporal(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        self._assert_finite(x, "after_temporal")

        valid_frames = valid_frames.unsqueeze(-1)  # (B, T, 1)

        # aggregates over time to yield a clip-level representation (B, D)
        x = (x * valid_frames).sum(dim=1)
        x = x / valid_frames.sum(dim=1).clamp(min=1)

        self._assert_finite(x, "before_classifier")
        return self.cls(x)