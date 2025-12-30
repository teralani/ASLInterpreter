import torch
import torch.nn as nn

class SpatialAttentionWithBias(nn.Module):
    def __init__(self, dim:int, heads:int, adj):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.register_buffer("adj", adj)
        # creates a non-learnable persistent tensor to a module
        # used for the positional/relational encodings for the edges (adjacency matrix)

    def forward(self, x):
        bias = self.adj.unsqueeze(0) 
        # inserts dimension at 0th index -> [1, num_joints, num_joints]

        return self.attn(x, x, x, attn_mask=bias)[0]    
        #returns attn_output from the linear transformation instead of attn_output_weights (tensor with attention weights)