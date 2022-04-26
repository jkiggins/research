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


class LifAstroNet:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = cfg['sim']['dt']

        num_synapse = self.cfg['linear_params']['synapse']

        self.modules = {}
        self.needs = {}
        
        self.neuron = LIFNeuron.from_cfg(cfg['lif_params'], self.dt)
        self.astro = Astro.from_cfg(cfg['astro_params'], cfg['linear_params']['synapse'], self.dt)
        self.linear = nn.Linear(num_synapse, 1, bias=False)
        nn.init.normal_(
            self.linear.weight,
            mean=cfg['linear_params']['mu'],
            std=cfg['linear_params']['sigma'])

        self.neuron_state = None
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


def _sim_snn(snn, spikes):
    
    tl = {
        'u': torch.zeros_like(spikes),
        'a': torch.zeros_like(spikes),
        'dw': torch.zeros_like(spikes),
        'i_pre': torch.zeros_like(spikes),
        'i_post': torch.zeros_like(spikes),
        'i_n': torch.zeros_like(spikes),
        'v_n': torch.zeros_like(spikes),
        'z_pre': torch.zeros_like(spikes),
        'z_post': torch.zeros_like(spikes),
        'w': torch.zeros_like(spikes)
    }

    tl['w'][0] = snn.linear.weight[:]

    for i, s in enumerate(spikes):
        z, a, n_state, a_state, linear = snn(s)

        weight_update = torch.logical_not(torch.isclose(a, torch.as_tensor(1.0))).float()

        tl['u'][i] = a_state['u']
        tl['a'][i] = weight_update
        tl['dw'][i] = a
        tl['i_pre'][i] = a_state['i_pre']
        tl['i_post'][i] = a_state['i_post']
        tl['z_pre'][i] = s
        tl['z_post'][i] = z
        
        tl['i_n'][i] = n_state['i']
        tl['v_n'][i] = n_state['v']
        tl['w'][i] = linear.weight[:]

    return tl

def gen_rate_spikes(duration):
    spike_trains = []
    
    torch.manual_seed(12343210938)
    for r in [0.5]:
        spike_trains.append(spiketrain.poisson([r], duration).transpose(1,0))

    return spike_trains

    
def gen_group_spikes():
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
    
    
def sim_spike_trains(snn_fn, cfg, spike_trains, name="snn_1n1s1a_rate-based-spikes"):

    db = ExpStorage()

    # Sim
    for spikes in spike_trains:
        snn = snn_fn()
        tl = _sim_snn(snn, spikes)
        db.store({'spikes': spikes, 'tl': tl})

    # Graph
    fig_idx = 0
    for spikes, by_spike in db.group_by('spikes').items():
        assert len(by_spike) == 1
        
        d = by_spike[0]
        tl = d['tl']

        fig = _graph_1nNs1a_tl(spikes, tl)

        fig.tight_layout()
        fig.savefig("{}_{}.svg".format(name, fig_idx))
        fig_idx += 1


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-e', '--exp', type=str, nargs='+')

    return parser.parse_args()


def _exp_rate_learning(args):
    with torch.no_grad():
        # Sim w/ out ltd/ltp thresholds
        cfg = config.Config(args.config)
        cfg['astro_params'] = cfg['anti_stdp']
        cfg['astro_params']['u_step_params']['ltd'] = 0.0
        cfg['astro_params']['u_step_params']['ltp'] = 0.0

        spikes = gen_rate_spikes(cfg['sim']['steps'])
        sim_spike_trains(lambda: LifAstroNet(cfg), cfg, spikes, name="snn_1n1s1a_rp_no-band")

        # Sim w/ thresholds
        cfg = config.Config(args.config)
        cfg['astro_params'] = cfg['anti_stdp']
        cfg['astro_params']['u_step_params']['ltd'] = -1.5
        cfg['astro_params']['u_step_params']['ltp'] = 1.5

        spikes = gen_rate_spikes(cfg['sim']['steps'])
        sim_spike_trains(lambda: LifAstroNet(cfg), cfg, spikes, name="snn_1n1s1a_rp_band")


def _exp_average_pulse_pair(args):
    """
    Simulate an snn in the 1n1s1a configuration, and demonstrate how plasticity
    triggered by a threshold on u can average multiple pulse pairs, and allow
    for confident steps for plasticity, based on multiple events.
    """
    
    with torch.no_grad():
        cfg = config.Config(args.config)
        cfg['astro_params'] = cfg['classic_stdp']

        spikes = gen_group_spikes()

        # Sim w/ baseline
        sim_spike_trains(lambda: LifAstroNet(cfg), cfg, spikes, name="snn_1n1s1a_tp_pulse")

        exit(0)

        # Set u_thr, show that when an input is driving the firing of a downstream spike, it tends to increase the weight
        cfg['astro_params']['u_th'] = 2.5
        # sim_spike_trains(
        #     lambda: LifAstroNet(cfg),
        #     cfg,
        #     spikes,
        #     name="snn_1n1s1a_stdp_u_thr={}".format(cfg['astro_params']['u_th'])
        # )

        
        # Use a different synapse to drive firing, and sprinkle in random spikes on the first

        noise_spikes = gen_noisy_spikes(spikes[0].shape[0])

        assert len(noise_spikes) == len(spikes)

        spikes = [torch.cat((spikes[i], noise_spikes[i]), axis=-1) for i in range(len(spikes))]

        sim_spike_trains(
            lambda: LifAstroNet(cfg),
            cfg,
            spikes,
            name="snn_1n1s1a_stdp_u_thr={}".format(cfg['astro_params']['u_th'])
        )


def _main(args):
    torch.manual_seed(12343210938)
    if 'rate' in args.exp:
        _exp_rate_learning(args)

    if 'tp-pulse' in args.exp:
        _exp_average_pulse_pair(args)
    
if __name__ == '__main__':
    args = _parse_args()

    _main(args)
