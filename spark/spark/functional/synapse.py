import torch

def synapse_step(z, params, mask=None):
    return torch.matmul(z, params['weight'])
