import torch
import torch.nn as nn
import numpy as np
from attn_with_bias import SpatialAttentionWithBias

# TODO:
# Positional Encoding
# Masking
# Motion features
# Graph bias

class SpatialTemporalModel(nn.Module):
    def __init__(self, joints=51, coords=3, hidden_dim=128, num_classes=2000, adj = None):
        super().__init__()

        self.embed = nn.Linear(coords * 2, hidden_dim)
        # (B, T, J, 3) -> (B, T, J, D) where D = hidden_dim

        self.spatial_attn = SpatialAttentionWithBias(
            hidden_dim, num_heads=4, batch_first=True, adj = adj
        )

        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )

        self.temporal_pos = nn.Parameter(torch.randn(1, 64, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = add_velocity(x)
        # x: (B, T, J, C)
        B, T, J, C = x.shape

        x = self.embed(x)

        # Spatial attention
        x = x.view(B*T, J, -1)
        x, _ = self.spatial_attn(x, x, x)
        x = x.view(B, T, J, -1)

        # Aggregate joints (collapses joints)
        x = x.mean(dim=2)

        # Add temporal position
        x = x + self.temporal_pos[:, :T]
        # new shape: (B, T, D)

        # Temporal attention
        x, _ = self.temporal_attn(x, x, x)

        # Pool + classify
        x = x.mean(dim=1)
        return self.classifier(x) # logits

def add_velocity(x):
    """
    
    :param x: (B, T, J, C)
    
    :returns: (B, T-1, J, 2*C)
    """
    velocity = x[:, 1:] - x[:, :-1] # (B, T-1, J, C)
    x_mid = x[:, :-1] # align
    return torch.cat([x_mid, velocity], dim=1)