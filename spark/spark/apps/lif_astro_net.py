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
from time import time
import copy


class LifAstroNet:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = cfg['sim']['dt']

        self.modules = {}
        self.needs = {}
        
        self.neuron = LIFNeuron.from_cfg(cfg['lif_params'], self.dt)
        self.astro = Astro.from_cfg(cfg['astro_params'], self.dt)
        self.linear = nn.Linear(1, 1, bias=False)
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
        'u': torch.as_tensor([0.0]),
        'a': torch.as_tensor([0.0]),
        'dw': torch.as_tensor([0.0]),
        'i_pre': torch.as_tensor([0.0]),
        'i_post': torch.as_tensor([0.0]),
        'i_n': torch.as_tensor([0.0]),
        'v_n': torch.as_tensor([0.0]),
        'z_pre': torch.as_tensor([0.0]),
        'z_post': torch.as_tensor([0.0]),
        'w': snn.linear.weight[0],
    }

    for s in spikes[0]:
        s = s.reshape(1)
        z, a, n_state, a_state, linear = snn(s)

        weight_update = ((a < 1.0 or a > 1.0) and torch.logical_not(torch.isclose(a, torch.as_tensor(1.0)))).float()

        tl['u'] = torch.cat((tl['u'], a_state['u'].reshape(1)))
        tl['a'] = torch.cat((tl['a'], weight_update.reshape(1)))
        tl['dw'] = torch.cat((tl['dw'], a.reshape(1)))
        tl['i_pre'] = torch.cat((tl['i_pre'], a_state['i_pre'].reshape(1)))
        tl['i_post'] = torch.cat((tl['i_post'], a_state['i_post'].reshape(1)))
        tl['z_pre'] = torch.cat((tl['z_pre'], s.reshape(1)))
        tl['z_post'] = torch.cat((tl['z_post'], z.reshape(1)))
        
        tl['i_n'] = torch.cat((tl['i_n'], n_state['i'].reshape(1)))
        tl['v_n'] = torch.cat((tl['v_n'], n_state['v'].reshape(1)))

        tl['w'] = torch.cat((tl['w'], linear.weight[0]))

    return tl


def sim_rate_spikes(snn_fn, cfg, name="snn_1n1s1a_rate-based-spikes"):
    spike_trains = []
    impulse_spikes = spiketrain.impulse(0, 10, 1000)

    db = ExpStorage()
    
    # spike_trains.append(impulse_spikes)
    for r in [0.5, 0.7]:
        spike_trains.append(spiketrain.poisson([r], cfg['sim']['steps']))

    # Sim
    for spikes in spike_trains:
        snn = snn_fn()
        tl = _sim_snn(snn, spikes)
        db.store({'spikes': spikes, 'tl': tl})

    i = 0
    for spikes, by_spike in db.group_by('spikes').items():
        assert len(by_spike) == 1
        
        d = by_spike[0]
        tl = d['tl']

        fig = plt.Figure(figsize=(6.4, 10))
        ax = fig.add_subplot(411)
        ax.set_xlim((0, len(tl['z_pre'])))
        ax.set_title("Neuron Traces")
        ax.plot(tl['i_n'], label='Neuron Current')
        ax.plot(tl['v_n'], label='Neuron Membrane Voltage')
        ax.legend()

        ax = fig.add_subplot(412)
        ax.set_xlim((0, len(tl['z_pre'])))
        ax.set_title("Astrocyte Traces")
        ax.plot(tl['i_pre'], label='Pre-synaptic Astrocyte Trace')
        ax.plot(tl['i_post'], label='Post-synaptic Astrocyte Trace')
        ax.plot(tl['u'], label='Astrocyte State')
        ax.plot(tl['i_post']-tl['i_pre'], label='Astrocyte Trace Diff')
        ax.legend()

        ax = fig.add_subplot(413)
        ax.set_xlim((0, len(tl['z_pre'])))
        ax.set_title("Astrocyte and Neuron Events")
        plot.plot_events(
            ax,
            [tl['z_pre'], tl['z_post'], tl['a']],
            colors=['tab:blue', 'tab:orange', 'tab:red'])
        ax.legend(['Pre Spikes', 'Post Spikes', 'Astro dw'])

        ax = fig.add_subplot(414)
        ax.set_xlim((0, len(tl['z_pre'])))
        ax.set_title("Synapse weight")
        ax.plot(tl['w'], marker='.')

        fig.tight_layout()
        fig.savefig("{}_{}.svg".format(name, i))
        i += 1

        # Some sanity checking
        for i in range(1, len(tl['w'])-1):
            if tl['w'][i] != tl['w'][i-1]:
                assert tl['a'][i] == 1
            if tl['a'][i] == 1:
                assert tl['w'][i] != tl['w'][i-1] or tl['w'][i] == cfg['linear_params']['max'], "w[i-1]: {}, w[i]: {}".format(tl['w'][i], tl['w'][i-1])

            

def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str)

    return parser.parse_args()



def _main(args):
    with torch.no_grad():
        cfg = config.Config(args.config)
        cfg['astro_params'] = cfg['classic_stdp']
        sim_rate_spikes(lambda: LifAstroNet(cfg), cfg, name="snn_1n1s1a_stdp_rate-input")

        bcfg = config.Config(args.config)
        cfg['astro_params'] = cfg['anti_stdp']
        sim_rate_spikes(lambda: LifAstroNet(cfg), cfg, name="snn_1n1s1a_anti-stdp_rate-input")

        

if __name__ == '__main__':
    args = _parse_args()

    _main(args)
