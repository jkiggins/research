import torch

def poisson(rates, steps=None):
    if not (type(rates) == torch.Tensor):
        rates = torch.as_tensor(rates)
        
    if not (steps is None):
        rates = rates.repeat(steps, 1).transpose(1,0)

    spike_vector = torch.poisson(rates)

    return (spike_vector > 0.1)*1.0


################### TESTS ######################
import pytest

def test_poisson_encode():
    # Test scalar rate
    rate = 0.5
    spikes = poisson(rate, steps=1000)
    mean_rate = torch.mean(spikes)

    assert spikes.shape[1] == 1000

    # Test list of rates
    rates = [0.5, 0.4, 0.3]
    spikes = poisson(rates, steps=1000)
    assert spikes.shape[0] == 3
    assert spikes.shape[1] == 1000

    # Test tensor or rates
    rates = torch.as_tensor(rates)
    spikes = poisson(rates, steps=1000)
    assert spikes.shape[0] == 3
    assert spikes.shape[1] == 1000
