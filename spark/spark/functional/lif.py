import torch

from .threshold import threshold

def lif_step(z, state, params, dt):
    """
    """
    # compute voltage updates
    dv = dt * params['tau_mem'] * ((params['v_leak'] - state['v']) + state['i'])
    v_decayed = state['v'] + dv

    # compute current updates
    di = -dt * params['tau_syn'] * state['i']
    i_decayed = state['i'] + di

    # compute new spikes
    z_new = threshold(v_decayed - params['v_th'], params['method'], params['alpha'])
    # compute reset
    v_new = (1 - z_new.detach()) * v_decayed + z_new.detach() * params['v_reset']
    # compute current jumps
    i_new = i_decayed + z

    state['i'] = i_new
    state['v'] = v_new

    return z_new, state


################### TESTS ######################

import pytest
from pathlib import Path

@pytest.fixture
def save_path():
    import os
    
    spath = (Path(__file__).parent/".test").absolute()
    if not spath.exists():
        os.makedirs(str(pspath))
    return spath


def test_lif_step(save_path):
    from matplotlib import pyplot as plt
    from . import encode
    import numpy as np
        
    print("Saving graphs in ", str(save_path))

    dt = 0.001
    lif_params = {
        'v_reset': -0.5,
        'v_leak': -0.02,
        'v_th': 1.0,
        'method': 'heaviside',
        'alpha': 1.0,  # doesn't matter if ^ is heaviside, see threshold.py
        'tau_mem': 1.0 / 1e-2,
        'tau_syn': 1.0 / 5e-3,
    }
    state = {'v': torch.as_tensor(0.0), 'i': torch.as_tensor(0.0)}

    random_spikes = encode.poisson(0.2, steps=100)
    timeline = {
        'z': [],
        'v': [],
        'i': []
    }
            
    for s in random_spikes:
        z, state = lif_step(s, state, lif_params, dt)
        
        timeline['z'].append(float(z))
        timeline['v'].append(float(state['v']))
        timeline['i'].append(float(state['i']))


    fig = plt.Figure(figsize=(6.4, 7))
    
    ax = fig.add_subplot(211)
    ax.set_title("LIF neuron membrane Voltage and Current over time")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage")
    ax.plot(timeline['v'], label='Membrane Voltage')
    ax.plot(timeline['i'], label='Post-synaptic Current')
    ax.set_xlim((0, len(timeline['z'])))
    ax.legend()
    
    z_in = random_spikes
    z_out = np.array(timeline['z'])

    event_z = [
        np.where(z_in > 0)[0].tolist(),
        np.where(z_out > 0)[0].tolist()
    ]

    ax = fig.add_subplot(212)
    ax.set_title("LIF neuron output events")
    ax.eventplot(event_z, lineoffsets=[0, 1], linelengths=[0.5, 0.5])
    ax.set_xlim((0, len(timeline['z'])))
    ax.legend(['z in', 'z out'])

    fig.tight_layout()
    fig.savefig(str(save_path/'test_lif_step.jpg'))    
