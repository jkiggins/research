import torch

from ..module.astrocyte import Astro
from ..module.neuron import LIFNeuron
from ..utils import config, plot
from ..data import spiketrain
from ..experiment import VSweep, ExpStorage

import argparse
import matplotlib.pyplot as plt
    

def sim_sweep_u(cfg, tau_i_pre, alpha_pre):
    tau_u = torch.logspace(-1, 4, 5)
    db = ExpStorage()
    
    spike_trains = []
    spike_trains.append(list(spiketrain.impulse(0, 10, 100)))
    for r in [0.1, 0.5, 0.7]:
        spike_trains.append(list(spiketrain.poisson(r, 100)))

    param_sweep = VSweep(tau_u)
    param_sweep = param_sweep.foreach(spike_trains)

    cfg('astro_params.tau_i_pre', tau_i_pre)
    cfg('astro_params.alpha_pre', alpha_pre)

    for tau_u, spikes in param_sweep.head.run():
        cfg('astro_params.tau_u', tau_u)
        astro = Astro.from_cfg(cfg['astro_params'], cfg['sim']['dt'])
        state = None

        timeline = {'i_pre': [], 'z_in': [], 'u': []}

        for z in spikes:
            eff, state = astro(state, z_pre=z)
            timeline['i_pre'].append(state['i_pre'])
            timeline['z_in'].append(z)
            timeline['u'].append(state['u'])

        db.store({
            'tau_u': tau_u,
            'spikes': spikes,
            'timeline': timeline
        })

    # One figure per spike-train
    records_by_spikes = db.group_by('spikes')
    for i, (_, by_spike) in enumerate(records_by_spikes.items()):
        fig = plt.Figure(figsize=(6.4, 7.5))
        ax = fig.add_subplot(3, 1, 1)
        ax.set_title("Astrocyte State U and Current Over Time")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Value")

        # Plot i_pre and u for each tau on a single plot
        for d in by_spike:
            tl = d['timeline']
            spikes = d['spikes']
            i_pre = tl['i_pre']
            ax.plot(tl['u'], label='tau_u={}'.format(d['tau_u']))
            ax.set_xlim((0, len(tl['z_in'])))
        ax.legend()

        ax = fig.add_subplot(3,1,2)
        ax.set_title("Astrocyte Pre Current Over time")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Value")
        ax.plot(i_pre)

        ax = fig.add_subplot(3,1,3)
        plot.plot_events(ax, [spikes])
        ax.set_title("Spikes over time")

        fig.tight_layout()
        fig.savefig("astro_u_{}.jpg".format(i))
    

def sim_sweep_pre_i(cfg):
    alpha_i_pre_vals = torch.linspace(0.1, 2.0, 5)
    tau_i_pre_vals = torch.logspace(1, 4, 10)
    spike_trains = []
    spike_trains.append(list(spiketrain.impulse(0, 10, 100)))
    for r in [0.1, 0.5, 0.7]:
        spike_trains.append(list(spiketrain.poisson(r, 100)))
        
    param_sweep = VSweep(tau_i_pre_vals)
    param_sweep = param_sweep.foreach(alpha_i_pre_vals)
    param_sweep = param_sweep.foreach(spike_trains)
    db = ExpStorage()

    for tau_i, alpha, spikes in param_sweep.head.run():
        cfg('astro_params.tau_i_pre', tau_i)
        cfg('astro_params.alpha_pre', alpha)
        astro = Astro.from_cfg(cfg['astro_params'], cfg['sim']['dt'])

        timeline = {'i_pre': [], 'z_in': []}
        
        state = None
        for z in spikes:
            eff, state = astro(state, z_pre=z)
            timeline['i_pre'].append(state['i_pre'])
            timeline['z_in'].append(z)

        db.store({
            'tau_i': float(tau_i),
            'alpha': float(alpha),
            'spikes': spikes,
            'timeline': timeline})

    # Get the timelines by spike train
    records_by_spike = db.group_by('spikes')
    for i, (_, by_spike) in enumerate(records_by_spike.items()):
        # One figure per spike train
        fig = plt.Figure(figsize=(6.4, 17))

        # Then split up by alpha
        records_by_alpha = by_spike.group_by('alpha')

        # One subplot per alpha value
        num_subplots = len(records_by_alpha) + 1
        for j, (alpha, by_alpha) in enumerate(records_by_alpha.items()):
            ax = fig.add_subplot(num_subplots, 1, j+1)
            ax.set_title("Astrocyte State U and Current Over Time for alpha={}".format(alpha))
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Value")

            # Plot i_pre and u for each tau on a single plot
            for d in by_alpha:
                tl = d['timeline']
                spikes = d['spikes']
                ax.plot(tl['i_pre'], label='tau={}'.format(d['tau_i']))
                ax.set_xlim((0, len(tl['z_in'])))
            ax.legend()

        # Last subplot has spike train
        ax = fig.add_subplot(num_subplots, 1, num_subplots)
        plot.plot_events(ax, [spikes])
        ax.set_title("Spikes over time")
        ax.legend(['Z In'])
        
        fig.tight_layout()
        fig.savefig('sweep_astro{}.jpg'.format(i))


def _parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--config', type=str)

    return parser.parse_args()


def _main():
    args = _parse_args()

    cfg = config.Config(args.config)

    sim_sweep_pre_i(cfg)
    sim_sweep_u(cfg, 250, 0.5)


if __name__ == '__main__':
    _main()
