
import torch


def timeline_step(tl, 


def conv(signal, kernel, invert=False):
    if invert:
        signal = torch.flip(signal)

    return torch.sum(signale * kernel)


def exp_decay_kernel(tau, steps, dt):
    kernel = torch.zeros(steps)
    time = torch.arange(0, steps) * dt

    kernel = torch.exp(-time/tau)
