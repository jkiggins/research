import torch

from .threshold import threshold

import torch.jit

def _clip_decay_across_zero(orig, decayed):
    orig_is_gt_zero = orig > 0.0
    decayed_is_lt_zero = decayed < 0.0
    where_invalid = torch.where(orig_is_gt_zero == decayed_is_lt_zero)

    decayed[where_invalid] = torch.as_tensor(0.0)

    return decayed
    

def astro_step_decay(state, params, dt):
    
    du = dt * params['tau_u'] * -state['u']
    u_decayed = state['u'] + torch.as_tensor(du)
    u_decayed = _clip_decay_across_zero(state['u'], u_decayed)

    state['u'] = u_decayed

    return state


# Perform a single step on the pre-synaptic input pathway
def astro_step_z_pre(z_pre, state, params, dt):
    # Current update

    di = dt * params['tau_i_pre'] * -state['i_pre']
    i_decayed = state['i_pre'] + di
    i_decayed = _clip_decay_across_zero(state['i_pre'], i_decayed)

    i_new = z_pre * params['alpha_pre']
    
    if params['pre_reset_on_spike'] and z_pre:
        # In this case, don't add i_decayed (forget past spikes)
        pass
    else:
        i_new = i_new + i_decayed

    
    state['i_pre'] = i_new

    return state


# Perform a single step on the posy-synaptic input pathway
def astro_step_z_post(z_post, state, params, dt):
    # Current update

    di = dt * params['tau_i_post'] * -state['i_post']
    i_decayed = state['i_post'] + di
    i_decayed = _clip_decay_across_zero(state['i_post'], i_decayed)

    i_new = z_post * params['alpha_post']
    
    if params['post_reset_on_spike'] and z_post:
        # In this case, don't add i_decayed (forget past spikes)
        pass
    else:
        i_new = i_new + i_decayed

    state['i_post'] = i_new

    return state


# Update u based on other signals
def astro_step_u_prod(state):
    """ Update u by adding the product of pre and post signals """
    du = state['i_pre'] * state['i_post']
    state['u'] = state['u'] + du

    return state


def astro_step_u_ordered_prod(state, params):
    du = state['i_pre'] * state['i_post']

    post_pre_diff = state['i_post'] - state['i_pre']

    if post_pre_diff < params['u_step_params']['ltd']:
        du = -du
    elif post_pre_diff > params['u_step_params']['ltp']:
        pass
    else:
        du = torch.as_tensor(0.0)

    state['u'] = state['u'] + du

    return state


def astro_step_u_stdp(state, params, z_pre=None, z_post=None):
    du = torch.as_tensor(0.0)

    # for LTP, we want to fully apply du when u is close to u_thr
    # and restruct du when u is close to -u_thr
    # (u + u_thr) / (2*u_thr)

    du = torch.zeros_like(state['u'])

    bool_z_post = torch.logical_and(z_pre == 0, z_post == 1)
    bool_z_pre = torch.logical_and(z_pre == 1, z_post == 0)

    # Handle z_post=1
    # LTP Step
    wh_z_post = torch.where(bool_z_post)
    du[wh_z_post] = state['i_pre'][wh_z_post].clone()
    state['i_pre'][wh_z_post] = torch.as_tensor(0.0)
    ltp_mult = (state['u'][wh_z_post] + params['u_th']) / (2*params['u_th'])

    # Handle z_post=0, z_pre=1
    # LTD Step
    wh_z_pre = torch.where(bool_z_pre)
    du[wh_z_pre] = -state['i_post'][wh_z_pre].clone()
    state['i_post'][wh_z_pre] = torch.as_tensor(0.0)
    ltp_mult = (state['u'][wh_z_pre] + params['u_th']) / (2*params['u_th'])

    # Apply LTP/LTD thresholds
    # bool_ltp = du[wh_z_post] < params['u_step_params']['ltp']
    # bool_ltd = du[wh_z_pre] > params['u_step_params']['ltd']
    # wh_ltp = torch.where(bool_ltp)
    # wh_ltd = torch.where(bool_ltd)

    # du[wh_z_post][wh_ltp] = -torch.abs(du[wh_z_post][wh_ltp])
    # du[wh_z_pre][wh_ltd] = torch.abs(du[wh_z_pre][wh_ltd])

    state['u'] = state['u'] + du
        
    return state


def astro_step_u_signal(state, params, dt):
    du = state['i_pre'] + state['i_post']

    state['u'] = state['u'] + du

    return state
        

# Apply a threshold
def astro_step_thr(state, params):
    u_spike_low = state['u'] < -(params['u_th'])
    u_spike_high = state['u'] > params['u_th']

    l_no_spike = torch.logical_not(
        torch.logical_or(u_spike_low,  u_spike_high))

    wh_no_spike = torch.where(l_no_spike)
    wh_spike = torch.where(torch.logical_not(l_no_spike))

    u_spike = u_spike_low * -1.0 + u_spike_high * 1.0
    u_spike[wh_no_spike] = torch.as_tensor(0.0)

    state['u'][wh_spike] = torch.as_tensor(0.0)
    
    return state, u_spike


# Step astro effects, based on the value of u
def astro_step_effect_weight(u_spike, params):
    weight_mod = torch.ones_like(u_spike)
    wh_ltd = u_spike < 0.0
    wh_ltp = u_spike > 0.0

    weight_mod[wh_ltd] = 0.95
    weight_mod[wh_ltp] = 1.05

    return weight_mod


################### tests ######################

import pytest
from pathlib import Path

@pytest.fixture
def save_path():
    import os
    
    spath = (Path(__file__).parent/".test").absolute()
    if not spath.exists():
        os.makedirs(str(pspath))
    return spath


def test_pre_post_signals():
    state = {
        'i_pre': torch.as_tensor(1.0),
        'i_post': torch.as_tensor(1.0),
    }

    astro_params = {
        'tau_i_pre': 200.0,
        'tau_i_post': 200.0,
        'alpha_pre': 1.0,
        'alpha_post': 1.0,
    }

    exp_values = [
        1.0,
        0.8000000119,
        0.6399999857,
        0.5119999647,
        0.4095999599,
        0.3276799619,
        0.2621439695,
        0.2097151726,
        0.1677721441,
        0.1342177093,
        0.1073741689,
        0.08589933813,
        0.06871946901,
        0.0549755767,
        0.04398046061,
        0.03518436849,
        0.02814749442,
        0.02251799591,
        0.01801439747,
        0.01441151835,
        0.01152921468,
    ]

    state = {
        'i_pre': torch.as_tensor(1.0),
        'i_post': torch.as_tensor(1.0),
    }

    for i in range(len(exp_values)):
        assert (exp_values[i] - state['i_pre']) < 1e-6, "I-PRE: Expected {}, got {}".format(exp_values[i], state['i_pre'])
        assert (exp_values[i] - state['i_post']) < 1e-6, "I-POST: Expected {}, got {}".format(exp_values[i], state['i_post'])

        astro_step_z_pre(0, state, astro_params, 0.001)
        astro_step_z_post(0, state, astro_params, 0.001)


    state = {
        'i_pre': torch.as_tensor(-1.0),
        'i_post': torch.as_tensor(-1.0),
    }

    for i in range(len(exp_values)):
        assert abs(-exp_values[i] - state['i_pre']) < 1e-6, "I-PRE: Expected {}, got {}".format(exp_values[i], state['i_pre'])
        assert abs(-exp_values[i] - state['i_post']) < 1e-6, "I-POST: Expected {}, got {}".format(exp_values[i], state['i_post'])

        astro_step_z_pre(0, state, astro_params, 0.001)
        astro_step_z_post(0, state, astro_params, 0.001)
        
    
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
    
