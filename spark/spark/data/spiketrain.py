import torch
from ..functional import encode


def impulse(start, size, duration):
    spikes = torch.zeros((1, duration))
    spikes[0, start:start+size] = 1

    return spikes


def poisson(rates, duration):    
    random_spikes = encode.poisson(rates, steps=duration)

    if len(random_spikes.shape) == 1:
        random_spikes = random_spikes.reshape(1, -1)

    return random_spikes


def uniform_noise(shape, prob):
    noise_spikes = torch.rand(shape) < prob

    return noise_spikes * 1.0


def pre_post_pair(
        spike_deltas,
        dt,
        fixed_duration=False,
        padding = 10
):
    """
    Generate steps number of pulse pairs, with the time between varying linearly
    between time_delta_range[0] and time_delta_range[1]. A positive pulse pair
    time delta implies the presynaptic spike cam before the post
    """

    # If scalar, make list with single element
    if not (hasattr(spike_deltas, '__iter__')):
        spike_deltas = [spike_deltas]

    if len(spike_deltas.shape) == 0:
        spike_deltas = spike_deltas.reshape(1)

    all_spike_trains = []

    for delta_t in spike_deltas:
        inter_spike_interval = int(torch.round(delta_t / dt))
        steps_for_sim = abs(inter_spike_interval) + int(padding * 2)
        spikes = torch.zeros((steps_for_sim, 2))

        if inter_spike_interval >= 0:
            spikes[padding][0] = 1
            spikes[padding+inter_spike_interval][1] = 1
        else:
            spikes[padding][1] = 1
            spikes[padding+abs(inter_spike_interval)][0] = 1

        all_spike_trains.append(spikes)

    return all_spike_trains


################### tests ######################
import pytest

def test_pre_post_pair():
    spike_deltas = torch.linspace(-50e-3, 50e-3, 101)
    dt = 0.001

    pulse_pair_spikes = pre_post_pair(spike_deltas, dt)

    for delta_t, spikes in zip(spike_deltas, pulse_pair_spikes):
        z_pre_idx = torch.where(torch.isclose(spikes[:, 0], torch.as_tensor(1.0)))[0][0]
        z_post_idx = torch.where(torch.isclose(spikes[:, 1], torch.as_tensor(1.0)))[0][0]

        assert torch.isclose((z_post_idx - z_pre_idx)*dt, delta_t)
