
import torch
import numpy as np

# Function to pre-compute the N kernel
def srm_lif_n_kernel(params, steps, dt):
    kernel = torch.zeros(steps)
    time = torch.arange(0, steps) * dt

    kernel = torch.exp(-time/params['tau'])
    kernel = params['v_reset'] * kernel

    return kernel


# Function to pre-compute the K kernel
def srm_lif_k_kernel(params, steps, dt):
    kernel = torch.zeros(steps)
    time = torch.arange(0, steps) * dt

    kernel = torch.exp(-time/params['tau'])

    return kernel


# Single step of SRM neuron
def srm_step(state, params, n, k, i):
    # Append input i to timeline
    tl = state['timeline']
    tl[1:] = tl[0:-1].clone()
    tl[0] = i

    # Convolve
    u = torch.sum(tl * k)

    # If there is a spike, clear out the timeline, and reset ttls
    if u >= params['thr']:
        u = torch.as_tensor(0.0)
        tl[:] = 0
        state['ttls'] = 0
        state['z'] = 1
    else:
        state['z'] = 0

    # Add in the effect of n (which lasts for some duration after a spike)
    if state['ttls'] < len(n):
        u += n[state['ttls']]

    # Increment for next time
    state['ttls'] = min(state['ttls'] + 1, len(n))
    state['ca'] = u

    return state


################### TESTS ######################

import pytest
from pathlib import Path

@pytest.fixture
def save_path():
    import os
    
    spath = (Path(__file__).parent/".test").absolute()
    if not spath.exists():
        os.makedirs(str(spath))
    return spath


def test_srm_step(save_path):
    from matplotlib import pyplot as plt
        
    print("Saving graphs in ", str(save_path))

    dt = 0.001

    k_params = {
        'tau': 0.001
    }
    k = srm_lif_k_kernel(k_params, 20, dt)

    n_params = {
        'tau': 0.001,
        'v_reset': -0.5,
    }
    n = srm_lif_n_kernel(n_params, 20, dt)

    print("Kernel k: ", k)
    print("Kernel n: ", n)

    srm_lif_params = {
        'thr': 1.5
    }

    state = {
        'timeline': torch.zeros(len(k)),
        'ttls': len(n),
    }
    u = []
    z_in = []
    z_out = []
    
    for i in range(20):
        if i in [2,10,11]:
            z = 1
        else:
            z = 0

        z_in.append(z)
        
        state = srm_step(state, srm_lif_params, n, k, z)

        u.append(state['ca'])
        z_out.append(state['z'])


    fig = plt.Figure()
    
    ax = fig.add_subplot(211)
    ax.set_title("LIF neuron membrane Voltage over time")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage")
    ax.plot(u)

    z_out = np.array(z_out, dtype=np.int)
    z_in = np.array(z_in, dtype=np.int)

    event_z = [
        np.where(z_in > 0)[0].tolist(),
        np.where(z_out > 0)[0].tolist(),
    ]
        

    ax = fig.add_subplot(212)
    ax.set_title("LIF neuron output events")
    ax.eventplot(event_z, lineoffsets=[0, 1], linelengths=[0.5, 0.5])
    ax.set_xlim((0, len(u)))

    fig.savefig(str(save_path/'test_srm_step.jpg'))

    
def test_srm_lif_k_kernel(save_path):
    print("Saving graphs in ", str(save_path))
    from matplotlib import pyplot as plt

    dt = 0.001
    
    params = {
        'tau': 0.0005
    }

    possible_tau = np.logspace(-4, -2, 10)

    fig = plt.Figure()
    ax = fig.add_subplot()
    ax.set_title("K kernel given Tau")

    for tau in possible_tau:
        params['tau'] = tau
        kernel = srm_lif_k_kernel(params, 10, dt)
        ax.plot(kernel.tolist(), label='tau={:2.4f}'.format(tau))

    ax.legend()

    fig.savefig(str(save_path/"test_lif_k_kernel.jpg"))
    
    
def test_srm_lif_n_kernel(save_path):
    print("Saving graphs in ", str(save_path))

    from matplotlib import pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Sweep steps and dt, generate plots
    dt = 0.001
    steps = 10
    possible_tau = np.logspace(-3, -1, 10)

    params = {
        'tau': 0.001,
        'v_reset': -0.5
    }

    fig = plt.Figure()
    
    ax = fig.add_subplot()
    ax.set_title("N kernel with given tau")

    for tau in possible_tau:
        params['tau'] = tau
        kernel = srm_lif_n_kernel(params, steps, dt)
        assert len(kernel) == steps
        ax.plot(kernel.tolist(), label='tau={:2.4f}'.format(tau))

    ax.legend()

    fig.subplots_adjust(left=0.125,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.35)

    fig.savefig(str(save_path/"srm_lif_n_kernel.jpg"))
