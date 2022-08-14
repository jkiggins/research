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


def astro_step_u_stdp(state, params, z_pre=None, z_post=None, reward=None):
    du = torch.zeros_like(state['u'])

    bool_ltd = torch.logical_and(
        z_pre > 0.0,
        torch.isclose(z_post, torch.tensor(0.0))
    )        
    wh_ltd = torch.where(bool_ltd)
    
    bool_ltp = torch.logical_and(
        z_post > 0.0,
        torch.isclose(z_pre, torch.tensor(0.0))
    )
    wh_ltp = torch.where(bool_ltp)

    # Peform LTP/LTD across astrocyte processes
    du[wh_ltd] = -state['i_post'][wh_ltd]
    du[wh_ltp] = state['i_pre'][wh_ltp]

    # Apply band
    du = torch.where(
        torch.abs(du) > params['dca_max'],
        torch.as_tensor(0.0),
        du)

    # Apply reward
    if reward is None:
        reward = torch.ones_like(du)
    du = du * reward

    state['u'] = state['u'] + du
    
    # When ip3 -> u or k+ -> u, that input trace is set to zero
    # These input traces effectivley "give" their value to u
    state['i_post'][wh_ltd] = torch.as_tensor(0.0)
    state['i_pre'][wh_ltp] = torch.as_tensor(0.0)

    return state



def astro_step_u_signal(state, params, dt):
    du = state['i_pre'] + state['i_post']

    state['u'] = state['u'] + du

    return state
        

def astro_step_reward_effect(state, params, reward):
    # reward is either 0.0, 1.0, or -1.0
    dw_prop = state['u'] * reward
    dw_prop = (dw_prop / params['u_th']) * 0.2 + 1.0

    # where reward is not 0.0
    wh_reward = torch.where(torch.logical_not(
            torch.isclose(reward, torch.as_tensor(0.0))
    ))

    # Reset u when updating weight
    state['u'][wh_reward] = torch.as_tensor(0.0)

    return state, dw_prop


# Apply activity
def astro_step_activity(state, params):
    # Detect when ip3 and k+ drop below a threshold, from above that same threshold
    act_lt_thr = (state['i_pre'] + state['i_post']) < params['ip3_kp_activity_thr']
    falling_edge = torch.logical_and(act_lt_thr, state['act_gt_thr'])

    u_spike = torch.zeros_like(state['u'])
    u_spike[falling_edge] = 1.0
    state['act_gt_thr'][falling_edge] = False

    return state, u_spike


def astro_track_activity(state, params):
    act_gt_thr = (state['i_pre'] + state['i_post']) > params['ip3_kp_activity_thr']

    if not ('act_gt_thr' in state):
        state['act_gt_thr'] = torch.logical_and(act_gt_thr, torch.as_tensor(False))
    
    rising_edge = torch.logical_and(
        torch.logical_not(state['act_gt_thr']),
        act_gt_thr)
        
    state['act_gt_thr'][rising_edge] = True

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


def astro_step_and_coupling(state, params):
    """
    AND - coupling between synapse
    This function will add logic to the simple threshold normally used for
    weight update

    If (syn0_ca>thr) and (syn1_ca>thr) - This is the desired behavior, no need
    to change weights

    If (syn0_ca>thr) xor (syn1_ca<thr) reverse sign of syn0_ca (anti-stdp)
    """

    syns = params['coupling']['and']
    if (syns is None) or len(syns) == 0:
        return state

    ca_gt_thr = torch.abs(state['u'][syns]) > params['ca_th']
    ca_gt_thr_ltp = state['u'][syns] > params['ca_th']
    ca_gt_thr_ltd = state['u'][syns] < -params['ca_th']

    wh_ca_gt_thr = torch.where(ca_gt_thr)
    wh_ca_gt_thr_ltp = torch.where(ca_gt_thr_ltp)
    wh_ca_gt_thr_ltd = torch.where(ca_gt_thr_ltd)

    if torch.all(torch.logical_not(ca_gt_thr)):
        # Do nothing, no thr exceeded
        pass

    elif torch.all(ca_gt_thr):
        # Zero out Ca, no need for thr events
        # u_before = state['u'].tolist()
        u_syns = state['u'][syns]
        u_syns[wh_ca_gt_thr] = 0.0
        state['u'][syns] = u_syns
        # print(
        #     "ca_gt_thr: {} -> {}: {} - {}".format(
        #         u_before,
        #         state['u'].tolist(),
        #         syns, ca_gt_thr))

    elif False: # torch.any(ca_gt_thr):
        # Invert all Ca values where thr was exceeded, weight update's must
        # follow anti-stdp
        # u_before = state['u'].tolist()
        u_syns = state['u'][syns]
        u_syns[wh_ca_gt_thr] = u_syns[wh_ca_gt_thr] * -1.0
        state['u'][syns] = u_syns

        # print(
        #     "ca_gt_thr: {} -> {}: {} - {}".format(
        #     u_before,
        #     state['u'].tolist(),
        #     syns, ca_gt_thr))


    return state

# Step astro effects, based on the value of u
def astro_step_effect_weight(u_spike, params):
    weight_mod = torch.ones_like(u_spike)
    wh_ltd = u_spike < 0.0
    wh_ltp = u_spike > 0.0

    weight_mod[wh_ltd] = params['u_step_params']['dw_ltd']
    weight_mod[wh_ltp] = params['u_step_params']['dw_ltp']

    return weight_mod


def astro_step_effect_weight_prop(u_spike, state, params):
    # u_spike is 1.0 when conditions are such that weights
    # should be updated

    # Weight update magnitude and direction are proportional to ca
    ca = state['u']
    dw = ca / params['u_th']
    dw = torch.clamp(dw, -1.0, 1.0)  # -1.0 to 1.0
    dw = dw * 0.5  # -0.5 to 0.5
    # 1.0 -> 0.5 to 1.5, 0.0 -> -0.5 to 0.5
    dw = u_spike + dw

    wh_zero = torch.where(torch.isclose(u_spike, torch.as_tensor(0.0)))
    wh_u_spike = torch.where(u_spike > 0.5)

    state['u'][wh_u_spike] = 0.0
    
    dw[wh_zero] = 1.0

    return state, dw


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
    
