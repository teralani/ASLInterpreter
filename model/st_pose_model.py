import torch
import torch.nn as nn
from model.spatial_attention import SpatialAttentionWithBias
from utils.velocity import add_velocity

class STPoseModel(nn.Module):
    def __init__(self, adj, num_classes:int = 2000, hidden_dim:int=128, num_heads:int = 4, T:int = 30):
        super().__init__()
        self.embed = nn.Linear(6, hidden_dim) # embeds coords into higher dim space
        self.spatial = SpatialAttentionWithBias(hidden_dim, num_heads, adj) # attention across joints (space) within a single frame
        self.temporal = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True # batch_first means tensors are [Batch num, rest of dimensions...]
        ) # attention across time (attends to other frames)
        self.pos = nn.Parameter(torch.randn(1, T, hidden_dim)) # a tensor that is used as a module parameter when backpropagating
        # self.pos is a learnable lookup table for time positions (1, max_t, D) so order of events matter
        self.cls = nn.Linear(hidden_dim, num_classes) # softmax classifier

    def forward(self, x):
        x = add_velocity(x)
        B, T, J, C = x.shape # Batchs, Time steps, Joints, Coordinates

        x = self.embed(x) # (B, T, J, C) -> (B, T, J, D)
        x = x.view(B*T, J, -1) # collapses across batch and time to apply spatial reasoning equally to each frame
        x = self.spatial(x) # outputs: (B*T, J, D)
        x = x.view(B, T, J, -1) # Undo flattening

        # Pooling across Joints
        x = x.mean(dim=2) # (B, T, D)

        # Adds previous positions until timestep
        x = x + self.pos[:, :T]

        x, _ = self.temporal(x, x, x)

        return self.cls(x.mean(dim=1)) # mean across Timestep dimension
