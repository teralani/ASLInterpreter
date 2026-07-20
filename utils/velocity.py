import torch
def add_velocity(x):
    """
    x: (B, T, J, 3)
    returns: (B, T, J, 6)
    """
    v = x[:, 1:] - x[:, :-1]           # (B, T-1, J, 3)
    v = torch.cat([v[:, :1], v], dim=1)  # pad first frame

    return torch.cat([x, v], dim=-1)   # (B, T, J, 6)
