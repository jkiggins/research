import torch
from torch import nn

import numpy as np

from ..module.astrocyte import Astro
from ..module.neuron import LIFNeuron
from ..utils import config, plot
from ..data import spiketrain
from ..experiment import VSweep, ExpStorage

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from time import time
import copy

class LifNet:
    def __init__(self, cfg, mu=None):
        self.cfg = cfg
        self.dt = cfg['sim']['dt']

        if not (mu is None):
            self.cfg['linear_params']['mu'] = mu

        self.num_synapse = self.cfg['linear_params']['synapse']
        self.neuron = LIFNeuron.from_cfg(cfg['lif_params'], self.dt)

        self.linear = nn.Linear(self.num_synapse, 1, bias=False)
        nn.init.normal_(
            self.linear.weight,
            mean=cfg['linear_params']['mu'],
            std=cfg['linear_params']['sigma'])

        self.neuron_state = None

    def __call__(self, z):
        z = z * 1.0
        if not (type(z) == torch.Tensor):
            z = torch.as_tensor(z)

        z = self.linear(z)
        z_post, self.neuron_state = self.neuron(z, self.neuron_state)

        return z_post, self.neuron_state


        
class LifAstroNet(LifNet):
    def __init__(self, cfg, *args, **kwargs):
        super(LifAstroNet, self).__init__(cfg, *args, **kwargs)
        
        self.astro = Astro.from_cfg(cfg['astro_params'], cfg['linear_params']['synapse'], self.dt)
        self.astro_state = None


    def __call__(self, z, gt=None):
        z = z * 1.0
        if not (type(z) == torch.Tensor):
            z = torch.as_tensor(z)

        z_pre = z
        z = self.linear(z)
        z_post, self.neuron_state = self.neuron(z, self.neuron_state)

        reward = None
        if not (gt is None):
            reward = (((z_post == gt) * 1.0) - 0.5)*2.0

        eff, self.astro_state = self.astro(self.astro_state, z_pre=z_pre, z_post=z_post, reward=reward)

        self.linear.weight[0] = torch.clamp(
            self.linear.weight[0] * eff,
            self.cfg['linear_params']['min'],
            self.cfg['linear_params']['max'])


        return z_post, eff, self.neuron_state, self.astro_state, self.linear


def _store_snn_step(tl, i, spikes, snn, snn_output, s):
    if tl is None:
        tl = {}
        tl['z_pre'] = torch.zeros_like(spikes)
        tl['i_n'] = torch.zeros_like(spikes)
        tl['v_n'] = torch.zeros_like(spikes)
        tl['z_post'] = torch.zeros_like(spikes)
        
        if len(snn_output) == 5:
            tl['u'] = torch.zeros_like(spikes)
            tl['a'] = torch.zeros_like(spikes)
            tl['dw'] = torch.zeros_like(spikes)
            tl['i_pre'] = torch.zeros_like(spikes)
            tl['i_post'] = torch.zeros_like(spikes)
            tl['w'] = torch.zeros_like(spikes)
            tl['w'][0] = snn.linear.weight[:]

    if len(snn_output) == 2:
        z, n_state = snn_output
    elif len(snn_output) == 5:
        z, a, n_state, a_state, linear = snn_output
        weight_update = torch.logical_not(torch.isclose(a, torch.as_tensor(1.0))).float()

        tl['u'][i] = a_state['u']
        tl['a'][i] = weight_update
        tl['dw'][i] = a
        tl['i_pre'][i] = a_state['i_pre']
        tl['i_post'][i] = a_state['i_post']
        tl['w'][i] = linear.weight[:]
        
    tl['z_pre'][i] = s
    tl['z_post'][i] = z
    tl['i_n'][i] = n_state['i']
    tl['v_n'][i] = n_state['v']

    return tl
    

def _sim_snn(snn, spikes):
    tl = None
    for i, s in enumerate(spikes):
        snn_output = snn(s)
        tl = _store_snn_step(tl, i, spikes, snn, snn_output, s)

    return tl

def _sim_reward_snn(snn, spikes, gt):
    tl = None
    for i, s in enumerate(spikes):
        snn_output = snn(s, gt=gt[i])
        tl = _store_snn_step(tl, i, spikes, snn, snn_output, s)

    return tl


def gen_rate_spikes(spec):
    spike_trains = []

    print("spec: ", spec)
    
    torch.manual_seed(12343210938)
    for r, steps in spec:
        spike_trains.append(spiketrain.poisson([r], steps).transpose(1,0))

    return spike_trains


def gen_group_spikes(noise=None, pulse_len=None):
    spike_trains = []

    # Ramp-up spike impulse w/ gap
    spikes = None
    gap_size = 100
    max_imp_len = 15
    for imp_size in range(1, max_imp_len):
        if not (pulse_len is None):
            imp_size = pulse_len
        impulse = torch.as_tensor([1,0]).repeat(imp_size)

        gap = torch.zeros((gap_size))

        if spikes is None:
            spikes = torch.cat((impulse, gap))
        else:
            spikes = torch.cat((spikes, impulse, gap))

    if not (noise is None):
        noise_spikes = spiketrain.uniform_noise(spikes.shape, noise)
        spikes = (spikes + noise_spikes).clip(0,1)
            
    # last dim is 1 for number of synapse
    spike_trains.append(spikes.reshape(-1, 1))

    return spike_trains


def gen_noisy_spikes(duration):
    spikes = spiketrain.poisson(0.1, duration)

    return [spikes.transpose(1,0)]


def _graph_1nNs1a_tl(spikes, tl):
    # Build a figure with the following
    # * Traces for the state values for a single neuron
    # * One plot each for the traces of an astrocyes process, for each synapse
    # * A plot of the spiking events, including weight update events
    # * A single plot with the weight values of all synapses

    spikes = torch.as_tensor(spikes)
    num_synapses = spikes.shape[-1]
    
    # Figure out the gridspec
    nrows = 4
    ncols = num_synapses
    gs = GridSpec(nrows, ncols, height_ratios=[0.6,1,1,1])

    fig = plt.Figure(figsize=(6.4, 10))

    # Neuron plot
    ax = fig.add_subplot(gs[0, 0:ncols])
    ax.set_xlim((0, len(tl['z_pre'])))
    ax.set_title("Neuron Membrane Voltage")
    ax.plot(tl['v_n'].squeeze().tolist(), label='Neuron Membrane Voltage')

    # A set of plots per synapse: Astrocyte signals, spiking activity
    for i in range(num_synapses):
        ax = fig.add_subplot(gs[1, i])
        ax.set_xlim((0, len(tl['z_pre'])))
        ax.set_title("Astrocyte Traces, Synapse: {}".format(i))

        ax.plot(tl['i_pre'][:, i], label='Pre-synaptic Astrocyte Trace')
        ax.plot(tl['i_post'][:, i], label='Post-synaptic Astrocyte Trace')
        ax.plot(tl['u'][:, i], label='Astrocyte State')
        # ax.plot(tl['i_post']-tl['i_pre'], label='Astrocyte Trace Diff')
        ax.legend()

        ax = fig.add_subplot(gs[2, i])
        ax.set_xlim((0, len(tl['z_pre'][:,i])))
        ax.set_title("Astrocyte and Neuron Events")
        plot.plot_events(
            ax,
            [tl['z_pre'][:,i], tl['z_post'][:,i], tl['a'][:,i]],
            colors=['tab:blue', 'tab:orange', 'tab:red'])
        ax.legend(['Pre Spikes', 'Post Spikes', 'Astro dw'])

        ax = fig.add_subplot(gs[3, i])
        ax.set_xlim((0, len(tl['z_pre'][:,i])))
        ax.set_title("Synapse weight")
        ax.plot(tl['w'][:,i], marker='.')

    return fig
    

def sim_lif_astro_reward_net(cfg, spike_trains, name="snn_1n1s1a_reward"):
    db = ExpStorage()
    db.meta['name'] = name

    for spikes in spike_trains:
        gt_snn = LifNet(cfg, mu=0.3)
        tl = _sim_snn(gt_snn, spikes)
        gt_spikes = tl['z_post']

        snn = LifAstroNet(cfg, mu=0.7)
        tl = _sim_reward_snn(snn, spikes, gt_spikes)
        db.store({'spikes': spikes, 'tl': tl})

    return db


def sim_lif_astro_sparse_reward_net(
    cfg,
    spike_trains,
    name="snn_1n1s1a_sparse_reward"
):
    db = ExpStorage()
    db.meta['name'] = name

    for spikes in spike_trains:
        gt_snn = LifNet(cfg, mu=0.3)
        tl = _sim_snn(gt_snn, spikes)
        gt_spikes = tl['z_post']

        snn = LifAstroNet(cfg, mu=0.7)
        tl = _sim_reward_snn(snn, spikes, gt_spikes)
        db.store({'spikes': spikes, 'tl': tl})

    return db


def sim_lif_astro_net(cfg, spike_trains, name="snn_1n1s1a_rate-based-spikes"):

    db = ExpStorage()
    db.meta['name'] = name

    # Sim
    for spikes in spike_trains:
        snn = LifAstroNet(cfg)
        tl = _sim_snn(snn, spikes)
        db.store({'spikes': spikes, 'tl': tl})

    return db


def sim_lif(cfg, spikes, name='lif_sample'):
    db = ExpStorage()
    db.meta['name'] = name

    # Sim
    snn = LifNet(cfg)
    tl = _sim_snn(snn, spikes)
    db.store({'spikes': spikes, 'tl': tl})

    return db


def graph_lif_astro_net(db):
    # Graph
    fig_idx = 0
    name = db.meta['name']
    for spikes, by_spike in db.group_by('spikes').items():
        assert len(by_spike) == 1
        
        d = by_spike[0]
        tl = d['tl']

        fig = _graph_1nNs1a_tl(spikes, tl)

        fig.tight_layout()
        fig.savefig("{}_{}.svg".format(name, fig_idx))
        fig_idx += 1


def graph_lif_astro_reward_net(db):
    # Graph
    fig_idx = 0
    name = db.meta['name']
    for spikes, by_spike in db.group_by('spikes').items():
        assert len(by_spike) == 1
        
        d = by_spike[0]
        tl = d['tl']

        fig = _graph_1nNs1a_tl(spikes, tl)

        fig.tight_layout()
        fig.savefig("{}_{}.svg".format(name, fig_idx))
        fig_idx += 1
