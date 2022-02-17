import torch

from .threshold import threshold

def astro_step_decay(state, params, dt):
    du = dt * params['tau_u'] * -state['u']
    u_decayed = max(state['u'] + du, 0)

    state['u'] = u_decayed

    return state


# Perform a single step on the pre-synaptic input pathway
def astro_step_z_pre(z_pre, state, params, dt):
    # Current update

    di = dt * params['tau_i_pre'] * -state['i_pre']
    i_decayed = max(state['i_pre'] + di, 0.0)
    i_new = i_decayed + z_pre * params['alpha_pre']

    state['u'] = state['u'] + state['i_pre'] * dt
    state['i_pre'] = i_new

    return state


# Perform a single step on the posy-synaptic input pathway
def astro_step_z_post(z_post, state, params, dt):
    # Current update

    di = dt * params['tau_i_pre'] * -state['i_pre']
    i_decayed = state['i_pre'] + di
    i_new = i_decayed + z_post * params['alpha_post']

    state['u'] = state['u'] + state['i_post']
    state['i_post'] = i_new

    return state


# Apply a threshold
def astro_step_thr(state, params):
    u_spike = threshold(state['u'] - params['u_th'], 'heaviside', 0.0)

    # if u exceeded the threshold
    if u_spike > 0.5:
        state['u'] = 0.0

    return state, u_spike > 0.5


# Step astro effects, based on the value of u
def astro_step_effect(u, state, params, dt):
    return None, state


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


def test_astro_step(save_path):
    from matplotlib import pyplot as plt
    from . import encode
    import numpy as np
        
    print("Saving graphs in ", str(save_path))

    dt = 0.001
    astro_params = {
        'tau_u': 1/1e-1,
        'tau_i_pre': 1/1e-4,
        'alpha_pre': 100.0,
        'alpha_post': 1.0,
        'u_th': 1.0,
    }

    # Simulate for 1000 time steps, constant spiking input
    z = torch.as_tensor(1)
    state = {
        'u': torch.as_tensor(0.0),
        'i_pre': 0.0,
        'i_post': 0.0
    }

    timeline = {'u': [], 'i_pre': [], 'z': [], 'u_spike': []}
    
    for i in range(150):
        if i > 2:
            z = 1
        else:
            z = 0

        state = astro_step_decay(state, astro_params, dt)
        state = astro_step_z_pre(z, state, astro_params, dt)
        
        # state, u_spike = astro_step_thr(state, astro_params)
        u_spike = 0

        timeline['u'].append(state['u'])
        timeline['i_pre'].append(state['i_pre'])
        timeline['z'].append(z)
        timeline['u_spike'].append(u_spike)

    fig = plt.Figure(figsize=(6.4, 7))
    
    ax = fig.add_subplot(211)
    ax.set_title("Astrocyte State U and Current Over Time")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Value")
    ax.plot(timeline['u'], label='Astrocyte State')
    ax.plot(timeline['i_pre'], label='pre-state Current')
    ax.set_xlim((0, len(timeline['z'])))
    ax.legend()

    z_in = np.array(timeline['z'])
    u_out = np.array(timeline['u_spike'])

    event_z = [
        np.where(z_in > 0)[0].tolist(),
        np.where(u_out > 0)[0].tolist()
    ]

    ax = fig.add_subplot(212)
    ax.set_title("Astrocyte Output Events")
    ax.eventplot(event_z, lineoffsets=[0, 1], linelengths=[0.5, 0.5])
    ax.set_xlim((0, len(timeline['z'])))
    ax.legend(['z in', 'u out'])

    fig.tight_layout()
    fig.savefig(str(save_path/'test_astro_step.jpg'))

    print(timeline)
        

    
    
