from ..functional import encode

def impulse(start, size, duration):
    for i in range(duration):
        if i >= start and i < (start + size):
            z = 1
        else:
            z = 0

        yield z


def poisson(rate, duration):
    random_spikes = encode.poisson(rate, steps=duration)

    for spike in random_spikes:
        yield spike
