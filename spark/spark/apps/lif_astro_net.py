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


    def __call__(self, z):
        z = z * 1.0
        if not (type(z) == torch.Tensor):
            z = torch.as_tensor(z)

        z_pre = z
        z = self.linear(z)
        z_post, self.neuron_state = self.neuron(z, self.neuron_state)

        eff, self.astro_state = self.astro(self.astro_state, z_pre=z_pre, z_post=z_post)

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


def gen_rate_spikes(spec):
    spike_trains = []

    print("spec: ", spec)
    
    torch.manual_seed(12343210938)
    for r, steps in spec:
        spike_trains.append(spiketrain.poisson([r], steps).transpose(1,0))

    return spike_trains


def gen_impulse_spikes(pulse_len, sim_len=None, num_impulses=None, noise=None):

    if (sim_len is None) == (num_impulses):
        raise ValueError("Either num_impulses or sim_len must be specified")
    
    spike_trains = []
    spikes = None
    gap_size = 100
    impulse_kernel = torch.as_tensor([1,0])

    if num_impulses is None:
        iters = sim_len // (pulse_len*impulse_kernel.numel()+gap_size)
    else:
        iters = num_impulses

    for i in range(iters):
        impulse = impulse_kernel.repeat(pulse_len)
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

    
def gen_ramp_impulse_spikes(noise=None):
    spike_trains = []

    # Ramp-up spike impulse w/ gap
    spikes = None
    gap_size = 100
    max_imp_len = 15            

    for imp_size in range(1, max_imp_len):
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


def gen_1nNs1a_axes(
    num_synapse,
    graphs=[
        'spikes',
        'neuron',
        'astro',
        'weight']
):
    # Validate graphs
    possible_graphs = ['spikes', 'neuron', 'astro', 'weight']
    for g in graphs:
        if not (g in possible_graphs):
            raise ValueError("Graph: {} is not supported, must be one of {}".format(g, possible_graphs))

    # Dynamic height ratios
    possible_height_ratios=dict(zip(possible_graphs, [0.6,1,1,1]))
    height_ratios = [possible_height_ratios[k] for k in graphs]
    
    # Figure out the gridspec
    nrows = len(graphs)
    ncols = num_synapse
    gs = GridSpec(nrows, ncols, height_ratios=height_ratios)

    fig = plt.Figure(figsize=(12, 10))
    axes = []

    graph_to_title = {
        'spikes': "Astrocyte and Neuron Events Synapse {}",
        'neuron': "Neuron Membrane Voltage",
        'astro': "Astrocyte Traces, Synapse {}",
        'weight': "Synapse {} Weight"
    }

    for gs_idx, g in enumerate(graphs):
        if g == "neuron":
            ax = fig.add_subplot(gs[gs_idx, 0:ncols])
            ax.set_title(graph_to_title[g])
        else:
            ax = []
            for i in range(num_synapse):
                a = fig.add_subplot(gs[gs_idx, i])
                ax.append(a)
                a.set_title(graph_to_title[g].format(i))

        axes.append((g, ax))

    return fig, axes
    
    
def graph_1nNs1a(
    tl,
    axes,
    prefix=''
):
    # Build a figure with the following
    # * Traces for the state values for a single neuron
    # * One plot each for the traces of an astrocyes process, for each synapse
    # * A plot of the spiking events, including weight update events
    # * A single plot with the weight values of all synapses

    spikes = tl['z_pre']

    for axis_spec in axes:
        g, ax = axis_spec

        # Neuron plot
        if g == 'neuron':
            ax.plot(tl['v_n'].squeeze().tolist(), label='{}Neuron Membrane Voltage'.format(prefix))

        else:
            # Multiple Axes, one per synapse
            if not (type(ax) == list):
                ax = [ax]

            for i, a in enumerate(ax):
                if g == 'astro':
                    a.plot(tl['i_pre'][:, i], label='{}Pre-synaptic Astrocyte Trace'.format(prefix))
                    a.plot(tl['i_post'][:, i], label='{}Post-synaptic Astrocyte Trace'.format(prefix))
                    a.plot(tl['u'][:, i], label='{}Astrocyte State'.format(prefix))

                elif g == 'spikes':
                    plot.plot_events(
                        a,
                        [tl['z_pre'][:,i], tl['z_post'][:,i], tl['a'][:,i]],
                        colors=['tab:blue', 'tab:orange', 'tab:red'])
                    a.legend(['{}Pre Spikes', '{}Post Spikes', '{}Astro dw'])

                elif g == 'weight':
                    a.plot(tl['w'][:,i], marker='.', label='{}'.format(prefix))

                else:
                    raise ValueError("Unknown graph type: {}".format(g))


def sim_lif_astro_net(cfg, spike_trains, db):
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


def graph_lif_astro_compare(tl, idx, graphs=None, fig=None, axes=None, prefix=''):
    if graphs is None:
        graphs = [
            'weight',
            'neuron',
            'spikes',
            'astro',
        ]


    # Build figure and axes if None
    if (fig is None) and (axes is None):
        fig = plt.Figure(figsize=(12, 10))
                
        nrows = len(graphs)
        ncols = 1
        gs = GridSpec(nrows, ncols)

        graph_to_title = {
            'spikes': "Astrocyte and Neuron Events Synapse {}",
            'neuron': "Neuron Membrane Voltage",
            'astro': "Astrocyte Traces, Synapse {}",
            'weight': "Synapse {} Weight"
        }

        axes = {g: [] for g in graphs}

        for gs_idx, g in enumerate(graphs):
            ax = fig.add_subplot(gs[gs_idx, 0])
            ax.set_title(graph_to_title[g].format(0))
            axes[g].append(ax)

    
    # Graph
    # Gather all the axes associated with idx
    axes_arg = []
    for g in axes:
        if idx < len(axes[g]):
            axes_arg.append((g, axes[g][idx]))
    graph_1nNs1a(tl, axes_arg, prefix=prefix)

    return fig, axes


def graph_lif_astro_net(db, graphs=None, fig=None, axes=None, prefix=''):
    """
    Graph the data from db either
    1. to newly created axes, as specified by graphs
    2. to the axes specified by the axes key word argument

    Also, apply prefix to traces
    """
    
    # Make it a list, if it isn't already
    if graphs is None:
        graphs = [
            'weight',
            'neuron',
            'spikes',
            'astro',
        ]


    for i, (spikes, by_spike) in enumerate(db.group_by('spikes').items()):
        pass

    assert i == 0
    assert len(by_spike) == 1

    num_synapse = torch.as_tensor(spikes).shape[-1]
            
    d = by_spike[0]
    tl = d['tl']

    # Generate axes and the figure
    if fig is None and axes is None:
        fig, axes = gen_1nNs1a_axes(num_synapse, graphs=graphs)
    elif not ((fig is None) == (axes is None)):
        raise ValueError("fig an axes kwargs must be mentioned together, or not at all")

    # Graph
    fig_idx = 0  # Keeping this for legacy naming
    graph_1nNs1a(tl, axes, prefix=prefix)

    fig.tight_layout()

    return fig
