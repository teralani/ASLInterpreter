import torch
def add_velocity(x, mask = None):
    """
    x: (B, T, J, 3)
    returns: (B, T, J, 6)
    """
    # compute raw differences
    v = x[:, 1:] - x[:, :-1]           # (B, T-1, J, 3)

    # mask out velocities where either previous or current joint is missing (all zeros)
    if mask is not None:
        prev_valid = mask[:, :-1] > 0.5
        curr_valid = mask[:, 1:] > 0.5
    else:
        prev = x[:, :-1]  # (B, T-1, J, 3)
        curr = x[:, 1:]
        prev_valid = (prev.abs().sum(dim=-1) > 0)  # (B, T-1, J)
        curr_valid = (curr.abs().sum(dim=-1) > 0)
        
    valid_pair = (prev_valid & curr_valid).unsqueeze(-1)  # (B, T-1, J, 1)

    v = v * valid_pair.to(v.dtype)

    # pad first frame velocity with zeros
    v = torch.cat([torch.zeros_like(v[:, :1]), v], dim=1)  # (B, T, J, 3)

    return torch.cat([x, v], dim=-1)   # (B, T, J, 6)
