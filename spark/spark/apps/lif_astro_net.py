import torch
from torch import nn

import numpy as np

from ..module.astrocyte import Astro
from ..module.neuron import LIFNeuron
from ..utils import config
from ..utils import plot as uplot
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

    def __call__(self, z, dw=False):
        # Ignore dw, kept for compatibility with LifAstroNet

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


    def __call__(self, z, dw=True):
        z = z * 1.0
        if not (type(z) == torch.Tensor):
            z = torch.as_tensor(z)

        z_pre = z
        z = self.linear(z)
        z_post, self.neuron_state = self.neuron(z, self.neuron_state)

        eff, self.astro_state = self.astro(self.astro_state, z_pre=z_pre, z_post=z_post)

        if dw:
            self.linear.weight[0] = torch.clamp(
                self.linear.weight[0] * eff,
                self.cfg['linear_params']['min'],
                self.cfg['linear_params']['max'])


        return z_post, eff, self.neuron_state, self.astro_state, self.linear


class LifAxis:
    def __init__(self, ax, offset=False):
        self.ax = ax
        self.offset = 0
        self.event_offset = 0

        self.offset_ticks = []
        self.offset_labels = []
        self.enable_offset = offset


    def _step_offset(self, y):
        if self.offset != 0:
            self.offset += max(-y.min(), 0.25)

        y = torch.as_tensor(y)
        y_max = y.max()
        y = y + self.offset

        print("Adding Offset: ", self.offset)

        # Tick locations, and labels for offset graphs
        max_extent = y[(y-self.offset).abs().argmax()]
        self.offset_ticks += [float(self.offset), float(max_extent)]
        self.offset_labels += ["0.0", "{:4.2f}".format(float(max_extent-self.offset))]
        
        # Add the max amount above the y=0 point, so the next graph clears that
        # Minimum offset is 0.25
        self.offset += max(y_max, 0.5)

        return y


    # Pass-throug methods
    def legend(self, *args, **kwargs):
        self.ax.legend(*args, **kwargs)

    def set_title(self, *args, **kwargs):
        self.ax.set_title(*args, **kwargs)

    def set_xlabel(self, *args, **kwargs):
        self.ax.set_xlabel(*args, **kwargs)

    def set_ylabel(self, *args, **kwargs):
        self.ax.set_ylabel(*args, **kwargs)

    def set_xlim(self, *args, **kwargs):
        self.ax.set_xlim(*args, **kwargs)


    def plot(self, *args, **kwargs):
        # Assume args[0] contains the data to plot
        if self.enable_offset:
            args = list(args)
            args[0] = self._step_offset(args[0])
            args = tuple(args)
            self.ax.set_yticks(self.offset_ticks)
            self.ax.set_yticklabels(self.offset_labels)
        self.ax.plot(*args, **kwargs)


    def plot_events(self, events, colors=None):
        event_idxs = []
        max_x = 0
        for z in events:
            if type(z) != np.ndarray:
                z = np.array(z)

            event_idx = np.where(z > 0)[0]
            max_x = max(len(z), max_x)
            event_idxs.append(event_idx.tolist())

        line_offsets = [i + self.event_offset for i in range(len(event_idxs))]

        self.ax.eventplot(event_idxs,
                     lineoffsets=line_offsets,
                     linelengths=0.5,
                     colors=colors)
        self.ax.set_xlim((0, max_x))

        self.event_offset = line_offsets[-1]
        
    
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
    

def _sim_snn_step(snn, tl, spikes, s, i, dw=True):
    snn_output = snn(s, dw=dw)
    tl = _store_snn_step(tl, i, spikes, snn, snn_output, s)

    return tl


def _sim_snn(snn, spikes, dw=True):
    tl = None
    for i, s in enumerate(spikes):
        tl = _sim_snn_step(snn, tl, spikes, s, i, dw=dw)
        # snn_output = snn(s)
        # tl = _store_snn_step(tl, i, sim_len, snn, snn_output, s)

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


def gen_sgnn_axes(
    num_synapse,
    graphs=[
        'spikes',
        'neuron',
        'astro',
        'weight']
):
    # Validate graphs
    possible_graphs = ['spikes', 'neuron', 'astro', 'astro-ca', 'weight']
    for g in graphs:
        if not (g in possible_graphs):
            raise ValueError("Graph: {} is not supported, must be one of {}".format(g, possible_graphs))

    # Dynamic height ratios
    possible_height_ratios=dict(zip(possible_graphs, [0.6,1,1,1,1]))
    height_ratios = [possible_height_ratios[k] for k in graphs]
    
    # Figure out the gridspec
    nrows = len(graphs)
    ncols = num_synapse
    gs = GridSpec(nrows, ncols)

    fig = plt.Figure(figsize=(12, 10))

    graph_to_title = {
        'spikes': "Astrocyte and Neuron Events for Synapse {}",
        'neuron': "Neuron Membrane Voltage",
        'astro': "Astrocyte Traces, Synapse {}",
        'astro-ca': "Astrocyte Calcium Response, Synapse {}",
        'weight': "Synapse {} Weight"
    }

    axes = {g: [] for g in graphs}
    
    for gs_idx, g in enumerate(graphs):
        if g == "neuron":
            ax = fig.add_subplot(gs[gs_idx, 0:ncols])
            ax.set_title(graph_to_title[g])
            ax = LifAxis(ax, offset=True)
        else:
            ax = []
            for i in range(num_synapse):
                a = fig.add_subplot(gs[gs_idx, i])
                a.set_title(graph_to_title[g].format(i))
                a = LifAxis(a, offset=True)
                ax.append(a)

        axes[g].append(ax)

    return fig, axes
    
    
def plot_1nNs1a(
    tl,
    axes,
    idx,
    plot=None,
    prefix=None,
    title=None,
):
    # Build a figure with the following
    # * Traces for the state values for a single neuron
    # * One plot each for the traces of an astrocyes process, for each synapse
    # * A plot of the spiking events, including weight update events
    # * A single plot with the weight values of all synapses

    if prefix is None:
        prefix = ''

    if plot is None:
        plot = ['neuron', 'astro', 'spikes', 'weight']

    spikes = tl['z_pre']

    for g, g_axes in axes.items():
        # if g is not in graphs (None -> match all)
        if not (plot is None) and not (g in plot):
            continue

        # Neuron plot
        if g == 'neuron' and 'neuron' in plot:
            ax = g_axes[idx]
            ax.plot(tl['v_n'].squeeze().tolist(), label='{}Neuron Membrane Voltage'.format(prefix))

            if len(prefix) > 0:
                a.legend()

        else:
            if idx >= len(g_axes):
                continue

            axs = g_axes[idx]

            for i, a in enumerate(axs):

                if g == 'astro' and ('astro' in plot):
                    a.plot(tl['i_pre'][:, i], label='{}Pre-synaptic Astrocyte Trace'.format(prefix))
                    a.plot(tl['i_post'][:, i], label='{}Post-synaptic Astrocyte Trace'.format(prefix))
                    a.plot(tl['u'][:, i], label='{} Astrocyte Ca'.format(prefix))

                elif g == 'astro-ca' and ('astro-ca' in plot):
                    a.plot(tl['u'][:, i], label='{}'.format(prefix))

                elif g == 'spikes' and 'spikes' in plot:
                    a.plot_events(
                        [tl['z_pre'][:,i], tl['z_post'][:,i], tl['a'][:,i]],
                        colors=['tab:blue', 'tab:orange', 'tab:red'])
                    a.legend([l.format(i) for l in ['{}Pre Spikes', '{}Post Spikes', '{}Astro dw']])

                elif g == 'weight' and 'weight' in plot:
                    a.plot(tl['w'][:,i], marker='.', label='{}'.format(prefix))

                else:
                    raise ValueError("Unknown graph type: {}".format(g))

                if len(prefix) > 0:
                    a.legend()


def sim_lif_astro_net(cfg, spike_trains, db, dw=True):
    # Sim
    for spikes in spike_trains:
        snn = LifAstroNet(cfg)
        tl = _sim_snn(snn, spikes, dw=dw)
        
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
            'astro-ca',
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
            'astro-ca': "Astrocyte Calcium Concentration, Synapse {}",
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

    plot_1nNs1a(tl, axes_arg, prefix=prefix)

    return fig, axes


def graph_sgnn(
    db_rec,
    fig, axes,
    idx, plot=None,
    prefix=None, titles=None):
    """
    Graph the data from db either
    1. to newly created axes, as specified by graphs
    2. to the axes specified by the axes key word argument

    Also, apply prefix to traces
    """
    
    # Make it a list, if it isn't already
    if plot is None:
        plot = [
            'weight',
            'neuron',
            'spikes',
            'astro',
        ]

    spikes = db_rec['spikes']
    tl = db_rec['tl']

    # Graph
    plot_1nNs1a(tl, axes, idx, prefix=prefix, plot=plot)

    return fig


def gen_dw_dt_axes(n_plots):
    gs = GridSpec(n_plots, 1)

    fig = plt.Figure(figsize=(16, 8))
    axes = []

    for i in range(n_plots):
        ax = fig.add_subplot(gs[i])
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')

        ax.set_xlabel("Pulse Pair Delta t")
        ax.xaxis.set_label_coords(0.7, 0.4)

        ax.set_ylabel("Weight Change")
        ax.yaxis.set_label_coords(0.47, 0.67)
        ax.grid(True)

        axes.append(ax)

    return fig, axes


def _graph_dw_dt(points, ax, text):
    spike_deltas = points[:, 0].tolist()
    
    ax.plot(points[:, 0], points[:, 1])
    ax.set_xticks(
        spike_deltas[::2],
        labels=["{:2.4f}".format(d) for d in spike_deltas[::2]],
        rotation=45)

    ax.text(
        -0.05, 0.8,
        text,
        bbox=uplot.plt_round_bbox)


def graph_dw_dt(db, title="", graph_text=""):
    # Graph
    points = []

    # Only one plot for now
    fig, axes = gen_dw_dt_axes(1)

    # Gather dw, dt pairs from sim results
    for i, (delta_t, by_spike_delta) in enumerate(db.group_by('delta_t').items()):
        weight_change = by_spike_delta[0]['dw']

        print("dw: ", weight_change)

        points.append(
            (float(delta_t), float(weight_change))
        )

    points = np.array(points)
    _graph_dw_dt(points, axes[0], graph_text)

    axes[0].set_title("{}: Weight Change vs. Pulse Pair Spike Delta".format(title))
    
    fig.tight_layout()

    return fig


def gen_dw_w_axes(n_plots, titles=None, spikes=False, size=(16, 8)):
    gs = GridSpec(n_plots, 1)

    fig = plt.Figure(figsize=size)
    axes = []

    for i in range(n_plots):

        ax = fig.add_subplot(gs[i])
        ax.set_ylabel("Number of Ca Threshold Events")
        ax.set_xlabel("Synaptic Weight")
        ax.grid(True)

        if titles:
            ax.set_title(titles[i])

        axes.append(ax)

    return fig, axes
    

def graph_dw_w(db, fig, axes, prefix=None, title=None, sp=0):
    points = []
    for d in db:
        ca = d['tl']['u']
        points.append([d['w'], d['ca-act'], 0.0, 0.0])

        # Error bars where Ca Threshold is not crossed
        points[-1][2:4] = [ca.min(), ca.max()]

    points = torch.as_tensor(points)

    axes[sp].errorbar(
        points[:, 0],
        points[:, 1],
        marker='.',
        yerr=torch.abs(points[:, 2:4].t()),
        ecolor='tab:orange',
        label=prefix
    )

    if title:
        axes[sp].set_title(title)

    if prefix:
        axes[sp].legend()

    return fig, axes
