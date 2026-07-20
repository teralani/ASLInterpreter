import torch
import torch.nn as nn

class SpatialAttentionWithBias(nn.Module):
    def __init__(self, dim:int, heads:int, adj):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        # Convert adjacency 0/1 to additive bias
        bias = torch.where(adj == 1, torch.zeros_like(adj), torch.full_like(adj, -1e9))
        self.register_buffer("bias", bias)  # (J, J)

    def forward(self, x):
        """
        x: (B*T, J, D)
        bias: (J, J)  --> automatically broadcast across batch
        """
        # ensure bias matches input dtype/device to avoid unexpected casts
        bias = self.bias.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, attn_mask=bias)[0]