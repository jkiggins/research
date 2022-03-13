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
def _astro_sim(astro, spike_deltas, pulse_pair_spikes, db, db_iter=None):
    param_sweep = VSweep(spike_deltas).zip(pulse_pair_spikes)

    # Simulate
    for delta_t, spikes in tqdm(param_sweep):
        state = None
        timeline = {
            'u': torch.as_tensor([0.0]),
            'i_pre': torch.as_tensor([0.0]),
            'i_post': torch.as_tensor([0.0]),
            'z_pre': torch.as_tensor([0.0]),
            'z_post': torch.as_tensor([0.0]),
        }
        
        for z_pre, z_post in spikes:
            eff, state = astro(state, z_pre=z_pre, z_post=z_post)
            timeline['u'] = torch.cat((timeline['u'], state['u'].reshape(1)))
            timeline['i_pre'] = torch.cat((timeline['i_pre'], state['i_pre'].reshape(1)))
            timeline['i_post'] = torch.cat((timeline['i_post'], state['i_post'].reshape(1)))
            timeline['z_pre'] = torch.cat((timeline['z_pre'], z_pre.reshape(1)))
            timeline['z_post'] = torch.cat((timeline['z_post'], z_post.reshape(1)))

            if (delta_t > 0 and z_post == 1) or (delta_t < 0 and z_pre == 1) or (z_post == 1 and z_pre == 1):
                timeline['max_u'] = state['u']
                timeline['last_u'] = last_u
                timeline['i_pre_at_max'] = state['i_pre']
                timeline['i_post_at_max'] = state['i_post']                

            last_u = state['u']
                

        if not (db_iter is None):
            d = {'timeline': timeline, 'delta_t': delta_t}
            d = db_iter(d)
            db.store(d)
        else:
            db.store({'timeline': timeline, 'delta_t': delta_t})


def _graph_dw_dt(cfg, db, u_fn='max', title=""):
    # Graph
    points = []
    for i, (delta_t, by_spike_delta) in enumerate(db.group_by('delta_t').items()):
        tl = by_spike_delta[0]['timeline']

        if u_fn == 'max':
            weight_change = float(tl['u'][tl['u'].abs().argmax()])
        if u_fn == 'max_pre_post':
            weight_change = tl['max_u']            
        elif u_fn == 'sum':
            weight_change = sum(tl['u'])

        points.append(
            (
                float(delta_t), weight_change
            )
        )

    points = np.array(points)
    spike_deltas = points[:, 0].tolist()
    
    fig = plt.Figure(figsize=(16, 8))
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
    ax.set_ylabel("Weight Change")
    ax.grid(True)

    ax.text(
        -0.05, 0.8,
        plot.astro_params_text(cfg['astro_params']),
        bbox=plot.plt_round_bbox)

    fig.tight_layout()

    return fig
    

def _graph_astro_tl_by_key(db, key, title=""):
        # Graphing
    fig = plt.Figure(figsize=(6.4, 17))

    records_by_key = db.group_by(key)
    num_subplots = len(records_by_key)
    sp_i = 1
    for val, by_key in records_by_key.items():
        assert len(by_key) == 1, "len is > 1: {}".format(len(by_key))
        
        tl = by_key[0]['timeline']
        sim_steps = len(tl['z_pre'])

        # Plot pre and Post Signals over time
        ax = fig.add_subplot(num_subplots, 1, sp_i)
        sp_i += 1
        ax.set_title("{}: Pulse Pair Response with {} = {:4.4f}".format(title, key, val))
        ax.set_xlabel("Timesteps (ms)")
        ax.set_ylabel("Value")
        c1 = ax.plot(tl['i_pre'])[0].get_color()
        c2 = ax.plot(tl['i_post'])[0].get_color()
        ax.plot(tl['u'])
        plot.plot_events(
            ax,
            [tl['z_pre'], tl['z_post']],
            colors=(c1, c2))
        ax.legend(['i_pre', 'i_post', 'u', 'z_in', 'z_out'])
        ax.set_xlim((0, sim_steps))
        ax.set_xticks(ticks=list(range(sim_steps)))

    fig.tight_layout()

    return fig


def _get_sim_name(cfg, name, sim_name):

    name = name.replace(" ", "_")
    sim_name = sim_name.replace(" ", "_")

    return "stdp_{}_{}_{}".format(
        cfg['astro_params']['mode'],
        name,
        sim_name
    )
    

def sim_shifted_stdp(cfg, tau_pre=300.0, tau_u=1000.0, alpha=1.0, vary=None):
    spike_delta_range = (-0.05, 0.05)
    spike_deltas = torch.linspace(*spike_delta_range, 101)
    pulse_pair_spikes = spiketrain.pre_post_pair(spike_deltas, cfg['sim']['dt'])
    dt = cfg['sim']['dt']

    if vary is None:
        return
    
    key, mods = vary

    cfg('astro_params.tau_u', tau_u)
    cfg('astro_params.tau_i_pre', tau_pre)
    cfg('astro_params.tau_i_post', tau_pre)
    cfg('astro_params.alpha_pre', alpha)
    cfg('astro_params.alpha_post', alpha)
    cfg('astro_params.u_step_params.mode', 'stdp_ordered')

    orig_val = cfg['astro_params'][key]

    db = ExpStorage()

    # Simulate the same spikes across different variants on the same astrocyte config
    for m in mods:
        # Init astro with modified param
        cfg['astro_params'][key] = orig_val * m
        astro = Astro.from_cfg(cfg['astro_params'], dt)
        
        _astro_sim(
            astro,
            spike_deltas,
            pulse_pair_spikes,
            db,
            db_iter=lambda x: dict(x, mod=(vary, m, cfg['astro_params'][key])))

    db.group_by('mod')

    # Graph
    fig = plt.Figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    ax.set_title("Astrocyte Plasticity: Weight Change vs. Pulse Pair delta-t, varying: {}".format(vary[0]))
    # ax.grid(True)
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')

    ax.set_xticks(
        spike_deltas.tolist()[::2],
        labels=["{:2.4f}".format(d) for d in spike_deltas.tolist()[::2]],
        rotation=45)
    ax.set_xlabel("Pulse Pair Delta t")
    ax.set_ylabel("Weight Change")

    ax.text(
        -0.05, 0.8,
        plot.astro_params_text(cfg['astro_params'], exclude=vary),
        bbox=plot.plt_round_bbox)


    for val, by_val in db.group_by('mod').items():
        points = []
        
        for r in by_val:
            points.append((r['delta_t'], r['timeline']['max_u']))

        points = np.array(points)
        ax.plot(points[:, 0], points[:, 1], '.-', label='{}={}'.format(key, val[2]))
    ax.legend()

    fig.tight_layout()
    fig.savefig("shifted_stdp_var_{}.svg".format(vary[0]))
    
        
def sim_classic_stdp(cfg, name="", title="", u_fn='max'):
    spike_delta_range = (-0.05, 0.05)
    spike_deltas = torch.linspace(*spike_delta_range, 101)
    dt = cfg['sim']['dt']
    pulse_pair_spikes = spiketrain.pre_post_pair(spike_deltas, dt)


    db = ExpStorage()
    astro = Astro.from_cfg(cfg['astro_params'], dt)

    # Sim
    _astro_sim(astro, spike_deltas, pulse_pair_spikes, db)

    # Graph
    fig = _graph_dw_dt(cfg, db, u_fn=u_fn, title=title)
    fig.savefig("{}_{}.svg".format(name, u_fn))
        


def sim_spike_pairs(cfg, name="", title=""):
    spike_deltas = torch.linspace(-0.02, 0.02, 7)
    pulse_pair_spikes = spiketrain.pre_post_pair(spike_deltas, cfg['sim']['dt'])
    astro = Astro.from_cfg(cfg['astro_params'], cfg['sim']['dt'])
    db = ExpStorage()

    # Simulation
    _astro_sim(astro, spike_deltas, pulse_pair_spikes, db)

    # Graph
    fig = _graph_astro_tl_by_key(db, "delta_t", title=title)
    fig.savefig("{}.svg".format(name))



def _get_configs(cfg_path):
    cfg_variants = [
        ('Classic STDP', 'classic_stdp'),
        ('Anti STDP', 'anti_stdp'),
        ('LTP Bias', 'ltp_bias'),
        ('LTD Bias', 'ltd_bias'),
        ('LTD dt Shift', 'ltd_dt_shift'),
        ('LTP dt Shift', 'ltp_dt_shift'),
    ]


    for cfg_vary in cfg_variants:
        cfg = config.Config(cfg_path)
        cfg['astro_params'] = cfg[cfg_vary[1]]

        yield cfg_vary[0], cfg

    
def _parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--config', type=str)

    return parser.parse_args()


def _main():
    args = _parse_args()

    with torch.no_grad():
        # cfg = config.Config(args.config)

        for cfg_name, cfg in _get_configs(args.config):
            cfg = copy.deepcopy(cfg)
            name = _get_sim_name(cfg, cfg_name, "dt_dw")
            title = cfg_name
            sim_classic_stdp(cfg, name=name, title=title)
            
            name = _get_sim_name(cfg, cfg_name, "tl")
            sim_spike_pairs(cfg, name=name, title=title)

        for cfg_name, cfg in _get_configs(args.config):
            cfg = copy.deepcopy(cfg)
            name = _get_sim_name(cfg, cfg_name, "dt_dw")
            title = cfg_name
            sim_classic_stdp(cfg, name=name, title=title, u_fn='sum')


        # cfg = config.Config(args.config)
        # mods = torch.linspace(0.5, 1.5, 5)
        # sim_shifted_stdp(cfg, tau_pre=100.0, tau_u=1000.0, alpha=1.0, vary=('tau_i_pre', mods))
        # cfg = config.Config(args.config)
        # sim_shifted_stdp(cfg, tau_pre=100.0, tau_u=1000.0, alpha=1.0, vary=('tau_i_post', mods))
        # cfg = config.Config(args.config)
        # sim_shifted_stdp(cfg, tau_pre=100.0, tau_u=1000.0, alpha=1.0, vary=('alpha_pre', mods))
        # cfg = config.Config(args.config)
        # sim_shifted_stdp(cfg, tau_pre=100.0, tau_u=1000.0, alpha=1.0, vary=('alpha_post', mods))        


if __name__ == '__main__':
    _main()
