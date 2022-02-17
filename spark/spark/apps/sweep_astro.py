import torch

from ..module.astrocyte import Astro
from ..module.neuron import LIFNeuron
from ..utils import config, plot
from ..data import spiketrain
from ..experiment import VSweep, ExpStorage

import argparse
import matplotlib.pyplot as plt
    
    
def sim(cfg):
    timelines = []
    taus = []

    tau_i_pre_vals = torch.logspace(1, 4, 10)
    spike_trains = []
    spike_trains.append(list(spiketrain.impulse(0, 10, 100)))
    for r in [0.1, 0.5, 0.7]:
        spike_trains.append(list(spiketrain.poisson(r, 100)))
        
    param_sweep = VSweep(tau_i_pre_vals)
    param_sweep = param_sweep.foreach(spike_trains)
    db = ExpStorage()


    for tau_i, spikes in param_sweep.head.run():
        cfg('astro_params.tau_i_pre', tau_i)
        astro = Astro.from_cfg(cfg['astro_params'], cfg['sim']['dt'])

        timeline = {'i_pre': [], 'z_in': []}
        
        state = None
        for z in spikes:
            eff, state = astro(state, z_pre=z)
            timeline['i_pre'].append(state['i_pre'])
            timeline['z_in'].append(z)

        db.store((tau_i, spikes, timeline))

    # Get the timelines by spike train
    records = db.unique(1)

    for i, (_, entires) in enumerate(records.items()):
        fig = plt.Figure(figsize=(6.4, 7))
        ax = fig.add_subplot(211)
        ax.set_title("Astrocyte State U and Current Over Time")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Value")
        
        for tau_i, spikes, tl in entires:
            ax.plot(tl['i_pre'], label='tau={}'.format(tau_i))
            ax.set_xlim((0, len(tl['z_in'])))
        ax.legend()

        ax = fig.add_subplot(212)
        plot.plot_events(ax, [spikes])
        ax.set_title("Spikes over time")
        ax.legend(['Z In'])
        
        fig.tight_layout()
        fig.savefig('sweep_pre_tau_{}.jpg'.format(i))


def _parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--config', type=str)

    return parser.parse_args()


def _main():
    args = _parse_args()

    cfg = config.Config(args.config)

    sim(cfg)


if __name__ == '__main__':
    _main()
