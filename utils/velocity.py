import torch
def add_velocity(x, mask = None):
    """
    x: (B, T, J, 3)
    returns: (B, T, J, 6)
    """

    v = x[:, 1:] - x[:, :-1] # (B, T-1, J, 3)


    if mask is not None:
        prev_valid = mask[:, :-1] > 0.5
        curr_valid = mask[:, 1:] > 0.5
    else:
        prev = x[:, :-1]  # (B, T-1, J, 3)
        curr = x[:, 1:]
        prev_valid = (prev.abs().sum(dim=-1) > 0) # (B, T-1, J)
        curr_valid = (curr.abs().sum(dim=-1) > 0)
        
    valid_pair = (prev_valid & curr_valid).unsqueeze(-1) # (B, T-1, J, 1)

    v = v * valid_pair.to(v.dtype)

    # pad first frame velocity with zeros
    v = torch.cat([torch.zeros_like(v[:, :1]), v], dim=1) # (B, T, J, 3)

    return torch.cat([x, v], dim=-1) # (B, T, J, 6)

def add_bone(x, mask, parents):
    """
    x: (B, T, J, 3), mask: (B, T, J), parents: (J,) LongTensor
    returns: (B, T, J, 3) bone vectors
    """
    parent_x = x[:, :, parents, :] # (B, T, J, 3)
    bone = x - parent_x

    parent_valid = mask[:, :, parents] > 0.5
    self_valid = mask > 0.5
    valid_pair = (parent_valid & self_valid).unsqueeze(-1)
    bone = bone * valid_pair.to(bone.dtype)

    is_root = (parents == torch.arange(parents.shape[0], device=parents.device))
    bone = bone * (~is_root).view(1, 1, -1, 1).to(bone.dtype)

    return bone