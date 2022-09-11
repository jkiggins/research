import torch

from .threshold import threshold

import torch.jit

def _get_syns(params, name):
    if name in params['coupling'] and len(params['coupling'][name]) > 0:
        return params['coupling'][name]

def _clip_decay_across_zero(orig, decayed):
    orig_is_gt_zero = orig > 0.0
    decayed_is_lt_zero = decayed < 0.0
    where_invalid = torch.where(orig_is_gt_zero == decayed_is_lt_zero)

    decayed[where_invalid] = torch.as_tensor(0.0)

    return decayed


def astro_step_decay(state, params, dt):
    
    du = dt * params['tau_ca'] * -state['ca']
    u_decayed = state['ca'] + torch.as_tensor(du)
    u_decayed = _clip_decay_across_zero(state['ca'], u_decayed)

    state['ca'] = u_decayed

    return state


# Perform a single step on the pre-synaptic input pathway
def astro_step_z_pre(z_pre, state, params, dt):
    # Current update

    di = dt * params['tau_ip3'] * -state['ip3']
    i_decayed = state['ip3'] + di
    i_decayed = _clip_decay_across_zero(state['ip3'], i_decayed)

    i_new = z_pre * params['alpha_pre']
    
    if params['pre_reset_on_spike'] and z_pre:
        # In this case, don't add i_decayed (forget past spikes)
        pass
    else:
        i_new = i_new + i_decayed

    state['ip3'] = i_new

    return state


# Perform a single step on the posy-synaptic input pathway
def astro_step_z_post(z_post, state, params, dt):
    # Current update

    di = dt * params['tau_kp'] * -state['kp']
    i_decayed = state['kp'] + di
    i_decayed = _clip_decay_across_zero(state['kp'], i_decayed)

    i_new = z_post * params['alpha_post']
    
    if params['post_reset_on_spike'] and z_post:
        # In this case, don't add i_decayed (forget past spikes)
        pass
    else:
        i_new = i_new + i_decayed

    state['kp'] = i_new

    return state


# Update u based on other signals
def astro_step_prod_ca(state, params):
    """ Update u by adding the product of pre and post signals """

    syns = _get_syns(params, 'prod')
    if syns is None:
        return state

    raise NotImplementedError("Synapse selection not implemented")
        
    du = state['ip3'] * state['kp']
    state['ca'] = state['ca'] + du

    return state


def astro_step_ordered_prod_ca(state, params):

    syns = _get_syns(params, 'ordered_prod')
    if syns is None:
        return state

    raise NotImplementedError("Synapse selection not implemented")

    du = state['ip3'] * state['kp']

    post_pre_diff = state['kp'] - state['ip3']

    if post_pre_diff < params['u_step_params']['ltd']:
        du = -du
    elif post_pre_diff > params['u_step_params']['ltp']:
        pass
    else:
        du = torch.as_tensor(0.0)

    state['ca'] = state['ca'] + du

    return state


def astro_step_stdp_ca(state, params, z_pre=None, z_post=None, reward=None):

    syns = _get_syns(params, 'stdp')
    if syns is None:
        return state

    dca = torch.zeros_like(state['ca'][syns])

    bool_ltd = torch.logical_and(
        z_pre[syns] > 0.0,
        torch.isclose(z_post, torch.tensor(0.0))
    )        
    wh_ltd = torch.where(bool_ltd)
    
    bool_ltp = torch.logical_and(
        z_post > 0.0,
        torch.isclose(z_pre[syns], torch.tensor(0.0))
    )
    wh_ltp = torch.where(bool_ltp)

    # Peform LTP/LTD across astrocyte processes
    dca[wh_ltd] = -state['kp'][syns][wh_ltd]
    dca[wh_ltp] = state['ip3'][syns][wh_ltp]

    # Apply band
    dca = torch.where(
        torch.abs(dca) > params['dca_max'],
        torch.as_tensor(0.0),
        dca)

    # Apply reward
    if reward is None:
        reward = torch.ones_like(dca)
    dca = dca * reward

    state['ca'][syns] = state['ca'][syns] + dca
    
    # When ip3 -> u or k+ -> u, that input trace is set to zero
    # These input traces effectivley "give" their value to u
    state['kp'][syns][wh_ltd] = torch.as_tensor(0.0)
    state['ip3'][syns][wh_ltp] = torch.as_tensor(0.0)

    return state



def astro_step_u_signal(state, params, dt):
    du = state['ip3'] + state['kp']

    state['ca'] = state['ca'] + du

    return state
        

def astro_step_reward_effect(state, params, reward):
    # reward is either 0.0, 1.0, or -1.0
    dw_prop = state['ca'] * reward
    dw_prop = (dw_prop / params['ca_th']) * 0.2 + 1.0

    # where reward is not 0.0
    wh_reward = torch.where(torch.logical_not(
            torch.isclose(reward, torch.as_tensor(0.0))
    ))

    # Reset u when updating weight
    state['ca'][wh_reward] = torch.as_tensor(0.0)

    return state, dw_prop


# Apply activity
def astro_step_activity(state, params):
    # Detect when ip3 and k+ drop below a threshold, from above that same threshold
    act_lt_thr = (state['ip3'] + state['kp']) < params['ip3_kp_activity_thr']
    falling_edge = torch.logical_and(act_lt_thr, state['act_gt_thr'])

    u_spike = torch.zeros_like(state['ca'])
    u_spike[falling_edge] = 1.0
    state['act_gt_thr'][falling_edge] = False

    return state, u_spike


def astro_track_activity(state, params):
    act_gt_thr = (state['ip3'] + state['kp']) > params['ip3_kp_activity_thr']

    if not ('act_gt_thr' in state):
        state['act_gt_thr'] = torch.logical_and(act_gt_thr, torch.as_tensor(False))
    
    rising_edge = torch.logical_and(
        torch.logical_not(state['act_gt_thr']),
        act_gt_thr)
        
    state['act_gt_thr'][rising_edge] = True

    return state
    

# Apply a threshold
def astro_step_thr(state, params):
    u_spike_low = state['ca'] < -(params['ca_th'])
    u_spike_high = state['ca'] > params['ca_th']

    l_no_spike = torch.logical_not(
        torch.logical_or(u_spike_low,  u_spike_high))

    wh_no_spike = torch.where(l_no_spike)
    wh_spike = torch.where(torch.logical_not(l_no_spike))

    u_spike = u_spike_low * -1.0 + u_spike_high * 1.0
    u_spike[wh_no_spike] = torch.as_tensor(0.0)

    state['ca'][wh_spike] = torch.as_tensor(0.0)
    
    return state, u_spike


def astro_step_and_coupling(state, params):
    """
    AND - coupling between synapse
    
    This function will consider IP3 values from N synapses, along with a
    single K+ value, and determine an appropreate Ca response that will drive
    plasticity.
    
    This step looks for the following conditions
    1. All IP3 > thr with K+ < thr. The outcome is increase in Syns 0-N
    2. All IP3 > thr With K+ > thr. The outcome is no activity
    3. Some IP3 > thr, and K+ > thr, The outcome is a decrease in the subset of synapse with IP3 > thr
    """

    syns = _get_syns(params, 'and')
    if syns is None:
        return state

    ip3 = state['ip3'][syns]
    kp = state['kp'][syns]
    ca = state['ca'][syns]

    dca = torch.zeros_like(ca)

    # and_th is a common threshold used for ip3, and k+
    ip3_gt_thr = ip3 > params['and_th']
    kp_gt_thr = kp > params['and_th']

    print(ip3_gt_thr, kp_gt_thr)


    # pre, pre -> no post: increase all weights
    ip3_high_kp_low = torch.logical_and(ip3_gt_thr,
                                        torch.logical_not(kp_gt_thr))
    all_ip3_high_kp_low = torch.logical_and(ip3_high_kp_low,
                                            torch.all(ip3_gt_thr))
    # Ca integrates ip3-s when this condition is met
    dca[all_ip3_high_kp_low] = ip3[all_ip3_high_kp_low]
    if torch.any(all_ip3_high_kp_low):
        print("pre, pre, no-post")

    
    # pre, no pre -> post: decrease synaptic weights
    ip3_high_kp_high = torch.logical_and(ip3_gt_thr, kp_gt_thr)
    some_ip3_high_kp_high = torch.logical_and(ip3_high_kp_high,
                                              torch.logical_not(torch.all(ip3_high_kp_high)))
    if torch.any(some_ip3_high_kp_high):
        print("pre, post, pre")
        
    # if torch.any(some_ip3_high_kp_high):
    #     print("ip3: {}, k+: {}".format(ip3.tolist(), kp.tolist()))

    dca[some_ip3_high_kp_high] = -ip3[some_ip3_high_kp_high]
    # ip3[some_ip3_high_kp_high] = 0.0
    # kp[torch.logical_and(kp_gt_thr, torch.any(some_ip3_high_kp_high))] = 0.0


    # pre, pre -> post: proper AND condition
    all_ip3_high_kp_high = torch.logical_and(ip3_high_kp_high,
                                              torch.all(ip3_high_kp_high))
    ip3[all_ip3_high_kp_high] = 0.0
    kp[all_ip3_high_kp_high] = 0.0
    if torch.any(some_ip3_high_kp_high):
        print("pre, pre, post: reset ip3 and kp")

    ca = ca + dca

    # if dca.abs().sum() > 0:
    #     print("k+: {}, ip3: {}, dca: {}".format(state['kp'][syns].tolist(),
    #                                             state['ip3'][syns].tolist(),
    #                                             dca.tolist()))

    state['ip3'][syns] = ip3
    state['kp'][syns] = kp
    state['ca'][syns] = ca

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
    ca = state['ca']
    dw = ca / params['ca_th']
    dw = torch.clamp(dw, -1.0, 1.0)  # -1.0 to 1.0
    dw = dw * 0.5  # -0.5 to 0.5
    # 1.0 -> 0.5 to 1.5, 0.0 -> -0.5 to 0.5
    dw = u_spike + dw

    wh_zero = torch.where(torch.isclose(u_spike, torch.as_tensor(0.0)))
    wh_u_spike = torch.where(u_spike > 0.5)

    state['ca'][wh_u_spike] = 0.0
    
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
        'ip3': torch.as_tensor(1.0),
        'kp': torch.as_tensor(1.0),
    }

    astro_params = {
        'tau_ip3': 200.0,
        'tau_kp': 200.0,
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
        'ip3': torch.as_tensor(1.0),
        'kp': torch.as_tensor(1.0),
    }

    for i in range(len(exp_values)):
        assert (exp_values[i] - state['ip3']) < 1e-6, "I-PRE: Expected {}, got {}".format(exp_values[i], state['ip3'])
        assert (exp_values[i] - state['kp']) < 1e-6, "I-POST: Expected {}, got {}".format(exp_values[i], state['kp'])

        astro_step_z_pre(0, state, astro_params, 0.001)
        astro_step_z_post(0, state, astro_params, 0.001)


    state = {
        'ip3': torch.as_tensor(-1.0),
        'kp': torch.as_tensor(-1.0),
    }

    for i in range(len(exp_values)):
        assert abs(-exp_values[i] - state['ip3']) < 1e-6, "I-PRE: Expected {}, got {}".format(exp_values[i], state['ip3'])
        assert abs(-exp_values[i] - state['kp']) < 1e-6, "I-POST: Expected {}, got {}".format(exp_values[i], state['kp'])

        astro_step_z_pre(0, state, astro_params, 0.001)
        astro_step_z_post(0, state, astro_params, 0.001)
        
    
def test_astro_step(save_path):
    from matplotlib import pyplot as plt
    from . import encode
    import numpy as np
        
    print("Saving graphs in ", str(save_path))

    dt = 0.001
    astro_params = {
        'tau_ca': 1/1e-1,
        'tau_ip3': 1/1e-4,
        'alpha_pre': 100.0,
        'alpha_post': 1.0,
        'ca_th': 1.0,
    }

    # Simulate for 1000 time steps, constant spiking input
    z = torch.as_tensor(1)
    state = {
        'ca': torch.as_tensor(0.0),
        'ip3': 0.0,
        'kp': 0.0
    }

    timeline = {'ca': [], 'ip3': [], 'z': [], 'u_spike': []}
    
    for i in range(150):
        if i > 2:
            z = 1
        else:
            z = 0

        state = astro_step_decay(state, astro_params, dt)
        state = astro_step_z_pre(z, state, astro_params, dt)
        
        # state, u_spike = astro_step_thr(state, astro_params)
        u_spike = 0

        timeline['ca'].append(state['ca'])
        timeline['ip3'].append(state['ip3'])
        timeline['z'].append(z)
        timeline['u_spike'].append(u_spike)

    fig = plt.Figure(figsize=(6.4, 7))
    
    ax = fig.add_subplot(211)
    ax.set_title("Astrocyte State U and Current Over Time")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Value")
    ax.plot(timeline['ca'], label='Astrocyte State')
    ax.plot(timeline['ip3'], label='pre-state Current')
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
    
