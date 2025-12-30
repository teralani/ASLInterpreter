import torch

def add_velocity(x):
    velocity = x[:, 1:] - x[:, :-1]

    x = x[:, :-1]

    return torch.cat([x, velocity], dim = 1)