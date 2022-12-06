import torch
import numpy as np

from ..module.astrocyte import Astro
from ..module.neuron import LIFNeuron
from ..utils import config, plot
from ..data import spiketrain
from ..experiment import VSweep, ExpStorage

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
import copy

# Astro Sim With pulse pair spikes
def sim_astro_probe(cfg, spikes, db, weight=1.0):
    astro = Astro.from_cfg(cfg['astro_params'], 1, cfg['sim']['dt'])

    n_synapse = 1

    state = None
    timeline = {
        'ca': torch.zeros(len(spikes), n_synapse),
        'eff': torch.zeros(len(spikes), n_synapse),
        'ip3': torch.zeros(len(spikes), n_synapse),
        'kp': torch.zeros(len(spikes), n_synapse),
        'z_pre': torch.zeros(len(spikes), n_synapse),
        'z_post': torch.zeros(len(spikes), n_synapse),
    }

    last_u = torch.as_tensor(0.0)
    
    # Simulate
    for i, (z_pre, z_post) in enumerate(spikes):
        eff, state = astro(state, z_pre=z_pre*weight, z_post=z_post)

        timeline['ca'][i] = state['ca']
        timeline['eff'][i] = eff
        timeline['ip3'][i] = state['ip3']
        timeline['kp'][i] = state['kp']
        timeline['z_pre'][i] = z_pre
        timeline['z_post'][i] = z_post

        if z_post == 1 or z_pre == 1:
            timeline['max_u'] = state['ca']
            timeline['last_u'] = last_u
            timeline['ip3_at_max'] = state['ip3']
            timeline['kp_at_max'] = state['kp']
        last_u = state['ca']

    db.store({'timeline': timeline})

    return db


def graph_dw_dt(db, title="", graph_text="", figsize=(10,6)):
    # Graph
    points = []
    
    for i, (delta_t, by_spike_delta) in enumerate(db.group_by('delta_t').items()):
        weight_change = by_spike_delta[0]['dw']

        points.append(
            (float(delta_t), float(weight_change))
        )

    points = np.array(points)
    spike_deltas = points[:, 0].tolist()

    fig = plt.Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_title("{}: Weight Change vs. Pulse Pair Spike Delta".format(title))
    ax.plot(points[:, 0], points[:, 1])

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')

    ax.set_xticks(
        spike_deltas[::2],
        labels=["{:2.4f}".format(d) for d in spike_deltas[::2]],
        rotation=45)
    ax.set_xlabel("Pulse Pair Delta t")
    ax.xaxis.set_label_coords(0.7, 0.4)

    ax.set_ylabel("Weight Change")
    ax.yaxis.set_label_coords(0.47, 0.67)
    ax.grid(True)

    ax.text(
        -0.05, 0.8,
        graph_text,
        bbox=plot.plt_round_box
    )

    fig.tight_layout()

    return fig, axes


def graph_astro_tls(records, key, prefix=""):
        # Graphing
    fig = plt.Figure(figsize=(6.4, 17))

    num_subplots = len(records)
    sp_i = 1
    for rec in records:
        tl = rec['timeline']
        sim_steps = len(tl['z_pre'])

        # Plot pre and Post Signals over time
        ax = fig.add_subplot(num_subplots, 1, sp_i)
        sp_i += 1

        ax.set_title("{}: Pulse Pair Response with {} = {:4.4f}".format(prefix, key, rec[key]))
        ax.set_xlabel("Timesteps (ms)")
        ax.set_ylabel("Value")
        c1 = ax.plot(tl['ip3'])[0].get_color()
        c2 = ax.plot(tl['kp'])[0].get_color()
        ax.plot(tl['ca'])
        plot.plot_events(
            ax,
            [tl['z_pre'], tl['z_post']],
            colors=(c1, c2))
        ax.legend(['ip3', 'kp', 'ca', 'z_in', 'z_out'])
        ax.set_xlim((0, sim_steps))
        ax.set_xticks(ticks=list(range(sim_steps)))

    fig.tight_layout()

    return fig

        
def sim_classic_stdp(cfg, name="", title="", u_fn='max'):
    spike_delta_range = (-0.05, 0.05)
    spike_deltas = torch.linspace(*spike_delta_range, 101)
    dt = cfg['sim']['dt']
    pulse_pair_spikes = spiketrain.pre_post_pair(spike_deltas, dt)

    db = ExpStorage()


    # Sim
    _astro_sim(astro, spike_deltas, pulse_pair_spikes, db)

    # Graph
    fig = _graph_dw_dt(cfg, db, u_fn=u_fn, title=title)
    fig.savefig("{}_{}.svg".format(name, u_fn))


def sim_spike_pairs(cfg):
    spike_deltas = torch.linspace(-0.02, 0.02, 7)
    pulse_pair_spikes = spiketrain.pre_post_pair(spike_deltas, cfg['sim']['dt'])
    astro = Astro.from_cfg(cfg['astro_params'], 1, cfg['sim']['dt'])
    db = ExpStorage()

    # Simulation
    _astro_sim(astro, spike_deltas, pulse_pair_spikes, db)

    return db
