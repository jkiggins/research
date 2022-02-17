import torch

def poisson(rate, steps=None):

    if not (type(rate) == torch.Tensor):
        rate = torch.as_tensor(rate)
        
    if not (steps is None):
        rate = rate.repeat(steps)
        
    spike_vector = torch.poisson(rate)

    return (spike_vector > 0.1)*1.0
