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

# Astro STDP Sim
def _astro_stdp_sim(astro, spike_deltas, pulse_pair_spikes, db, db_iter=None):
    param_sweep = VSweep(spike_deltas).zip(pulse_pair_spikes)

    # Simulate
    for delta_t, spikes in tqdm(param_sweep):
        state = None
        timeline = {
            'u': torch.as_tensor([0.0]),
            'i_pre': torch.as_tensor([0.0]),
            'i_post': torch.as_tensor([0.0]),
        }
        
        for z_pre, z_post in spikes:
            eff, state = astro(state, z_pre=z_pre, z_post=z_post)
            timeline['u'] = torch.cat((timeline['u'], state['u'].reshape(1)))
            timeline['i_pre'] = torch.cat((timeline['i_pre'], state['i_pre'].reshape(1)))
            timeline['i_post'] = torch.cat((timeline['i_post'], state['i_post'].reshape(1)))

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
    cfg('astro_params.u_update', 'stdp_ordered')

    orig_val = cfg['astro_params'][key]

    db = ExpStorage()

    # Simulate the same spikes across different variants on the same astrocyte config
    for m in mods:
        # Init astro with modified param
        cfg['astro_params'][key] = orig_val * m
        astro = Astro.from_cfg(cfg['astro_params'], dt)
        
        _astro_stdp_sim(
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
    
        
def sim_classic_stdp(cfg, tau_pre=300.0, tau_u=1000.0, alpha=1.0):
    spike_delta_range = (-0.05, 0.05)
    spike_deltas = torch.linspace(*spike_delta_range, 101)
    pulse_pair_spikes = spiketrain.pre_post_pair(spike_deltas, cfg['sim']['dt'])

    db = ExpStorage()

    dt = cfg['sim']['dt']
    cfg('astro_params.tau_u', tau_u)
    cfg('astro_params.tau_i_pre', tau_pre)
    cfg('astro_params.tau_i_post', tau_pre)
    cfg('astro_params.alpha_pre', alpha)
    cfg('astro_params.alpha_post', alpha)
    cfg('astro_params.u_update', 'stdp_ordered')
    astro = Astro.from_cfg(cfg['astro_params'], dt)

    # Sim
    _astro_stdp_sim(astro, spike_deltas, pulse_pair_spikes, db)

    # Graph
    points = []
    for i, (delta_t, by_spike_delta) in enumerate(db.group_by('delta_t').items()):
        tl = by_spike_delta[0]['timeline']
        points.append(
            (
                float(delta_t), float(tl['max_u'])
            )
        )

        # print("{}\t{}\t{}\t{}\t{}".format(delta_t, tl['i_pre_at_max'], tl['i_post_at_max'], tl['last_u'], tl['max_u']))

    points = np.array(points)
    
    fig = plt.Figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    ax.set_title("Astrocyte Plasticity: Weight Change vs. Pulse Pair Spike Delta")
    ax.plot(points[:, 0], points[:, 1])
    ax.set_xticks(
        spike_deltas.tolist()[::2],
        labels=["{:2.4f}".format(d) for d in spike_deltas.tolist()[::2]],
        rotation=45)
    ax.set_xlabel("Pulse Pair Delta t")
    ax.set_ylabel("Weight Change")
    ax.grid(True)
    ax.text(
        -0.05, 0.8,
        plot.astro_params_text(cfg['astro_params']),
        bbox=plot.plt_round_bbox)

    fig.tight_layout()
    fig.savefig("astro_classic_stdp.svg")
        

def sim_sweep_io_freq(cfg, spike_rate_range):
    spike_rates = torch.linspace(*spike_rate_range, 5)
    all_pre_spikes = spiketrain.poisson(spike_rates, 100)
    all_post_spikes = spiketrain.poisson(spike_rates, 100)

    astro = Astro.from_cfg(cfg['astro_params'], cfg['sim']['dt'])

    assert all_pre_spikes.shape[0] == all_post_spikes.shape[0]

    db = ExpStorage()

    # Running simulation
    for i, spike_rate in enumerate(tqdm(spike_rates)):
        pre_spikes = all_pre_spikes[i]
        post_spikes = all_post_spikes[i]

        timeline = {
            'i_pre': [],
            'i_post': [],
            'z_pre': [],
            'z_post': [],
            'u': []
        }

        state = None
        for z_pre, z_post in zip(pre_spikes, post_spikes):
            eff, state = astro(state, z_pre=z_pre, z_post=z_post)

            timeline['i_pre'].append(float(state['i_pre']))
            timeline['i_post'].append(float(state['i_post']))
            timeline['z_pre'].append(int(z_pre))
            timeline['z_post'].append(int(z_post))
            timeline['u'].append(float(state['u']))

        db.store({
            'spike_rate': spike_rate,
            'timeline': timeline})


    # Graphs
    fig = plt.Figure(figsize=(6.4, 17))
    records_by_spike_rate = db.group_by('spike_rate')
    num_subplots = len(records_by_spike_rate)
    sp_i = 1
    for spike_rate, by_spike_rate in records_by_spike_rate.items():
        tl = by_spike_rate[0]['timeline']
                
        ax = fig.add_subplot(num_subplots, 1, sp_i)
        sp_i += 1
        ax.set_title("Astrocyte Learning response for a spike rate of: {:4.4f}".format(spike_rate))
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Value")
        c1 = ax.plot(tl['i_pre'])[0].get_color()
        c2 = ax.plot(tl['i_post'])[0].get_color()
        ax.plot(tl['u'])
        
        plot.plot_events(
            ax,
            [tl['z_pre'], tl['z_post']],
            colors=(c1, c2),
            offset=-2)

        ax.legend(['i_pre', 'i_post', 'u', 'z_pre', 'z_post'])

    fig.tight_layout()
    fig.savefig("sweep_pre_post_freq.svg")


def sim_sweep_spike_pairs(cfg, spike_delta_range):
    spike_deltas = torch.linspace(*spike_delta_range, 8)
    pulse_pair_spikes = spiketrain.pre_post_pair(spike_deltas, cfg['sim']['dt'])
    astro = Astro.from_cfg(cfg['astro_params'], cfg['sim']['dt'])
    db = ExpStorage()

    # Simulation
    total_iters = len(spike_deltas)
    for spikes, delta_t in tqdm(zip(pulse_pair_spikes, spike_deltas), total=total_iters):
        print("Simulating {} spikes".format(len(spikes)))

        state = None
        timeline = {
            'z_pre': [],
            'z_post': [],
            'i_pre': [],
            'i_post': [],
            'u': [],
            'spikes': spikes,
        }
        
        for z_pre, z_post in spikes:
            eff, state = astro(state, z_pre=z_pre, z_post=z_post)
            timeline['z_pre'].append(z_pre)
            timeline['z_post'].append(z_post)
            timeline['i_pre'].append(state['i_pre'])
            timeline['i_post'].append(state['i_post'])
            timeline['u'].append(state['u'])

        db.store({'delta_t': delta_t, 'timeline': timeline})


    # Graphing
    fig = plt.Figure(figsize=(6.4, 17))

    records_by_delta_t = db.group_by('delta_t')
    num_subplots = len(records_by_delta_t)
    sp_i = 1
    for delta_t, by_delta_t in records_by_delta_t.items():
        assert len(by_delta_t) == 1, "len is > 1: {}".format(len(by_delta_t))
        tl = by_delta_t[0]['timeline']
        sim_steps = len(tl['spikes'])

        # Plot pre and Post Signals over time
        ax = fig.add_subplot(num_subplots, 1, sp_i)
        sp_i += 1
        ax.set_title("Pulse Pair Response with \delta_t = {:4.4f}".format(delta_t))
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
    fig.savefig("astro_stdp.svg")


def sim_heatmap_alpha_update_rate(cfg):
    spike_rate_range = torch.linspace(0.05, 0.8, 20)
    alpha_range = torch.linspace(0.1, 2.0, 20)

    param_sweep = VSweep(values=spike_rate_range)
    param_sweep = param_sweep.foreach(alpha_range)

    dt = cfg['sim']['dt']

    db = ExpStorage()

    # Simulate
    for spike_rate, alpha in tqdm(param_sweep):
        cfg('astro_params.alpha_pre', alpha)
        cfg('astro_params.alpha_post', alpha)
        
        astro = Astro.from_cfg(cfg['astro_params'], cfg['sim']['dt'])
        pre_spikes = spiketrain.poisson(spike_rate, 1000)
        post_spikes = spiketrain.poisson(spike_rate, 1000)
        state = None
        timeline = {
            'eff': [],
            'u': []
        }

        for z_pre, z_post in zip(pre_spikes[0], post_spikes[0]):
            eff, state = astro(state, z_pre=z_pre, z_post=z_post)
            timeline['eff'].append(eff)
            timeline['u'].append(state['u'])

        # print(
        #     "alpha: {}, spike rate: {}, any_effect: {}, max(u): {}".format(
        #         alpha,
        #         spike_rate,
        #         any(timeline['eff']),
        #         max(timeline['u']),
        #     ))

        db.store({
            'spike_rate': spike_rate,
            'alpha': alpha,
            'timeline': timeline})

    # Graph
    fig = plt.Figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    ax.set_title("Heatmap of average time to weight update given Poisson spike rate vs. alpha")
    ax.set_yticks(
        list(range(len(alpha_range))),
        labels=["{:2.4f}".format(float(a)) for a in alpha_range],
        rotation=45)
    ax.set_ylabel('Pre and Post Alpha')

    ax.set_xticks(
        list(range(len(spike_rate_range))),
        labels=["{:2.4f}".format(float(a)) for a in spike_rate_range])
    ax.set_xlabel('Spike Rate')
    
    heat_img = torch.zeros((len(alpha_range), len(spike_rate_range)))
    for i, (spike_rate, alpha_db) in enumerate(db.group_by('spike_rate', sort=True).items()):
        for j, (alpha, elem_db) in enumerate(alpha_db.group_by('alpha', sort=True).items()):
            tl = elem_db[0]['timeline']
            eff_sum = sum(tl['eff'])
            heat_img[j, i] = eff_sum
            ax.text(
                i, j,
                "{}".format(float(eff_sum)),
                ha="center", va="center", color="w")

    ax.imshow(heat_img)
    fig.tight_layout()
    fig.savefig("astro_alpha_spike_freq_heat.svg")
            
    
def sim_heatmap_dt_tau(cfg):
    tau_range = torch.logspace(0.5, 3, 50)
    delta_t_range = torch.linspace(0, 20e-3, 21)

    param_sweep = VSweep(values=tau_range)
    param_sweep = param_sweep.foreach(delta_t_range)

    db = ExpStorage()

    # Simulate
    for tau, delta_t in tqdm(param_sweep):

        # Create astro with modified config
        cfg('astro_params.tau_i_pre', tau)
        cfg('astro_params.tau_i_post', tau)
        
        astro = Astro.from_cfg(cfg['astro_params'], cfg['sim']['dt'])
        state = None
        
        pulse_pair_spikes = spiketrain.pre_post_pair(float(delta_t), cfg['sim']['dt'])
        delta_u = 0

        for z_pre, z_post in pulse_pair_spikes[0]:
            eff, state = astro(state, z_pre=z_pre, z_post=z_post)

            if int(z_post) == 1:
                delta_u = state['u']
                break

        db.store({
            'tau': tau,
            'delta_t': delta_t,
            'delta_u': delta_u
        })


    # Graph
    heat_img = torch.zeros((len(tau_range), len(delta_t_range)))
    for d in db:
        tau_idx = tau_range.tolist().index(d['tau'])
        delta_idx = delta_t_range.tolist().index(d['delta_t'])
        heat_img[tau_idx, delta_idx] = d['delta_u']

    fig = plt.Figure(figsize=(14,40))
    ax = fig.add_subplot(111)
    img = ax.imshow(heat_img)

    # Add annotation
    for i in range(heat_img.shape[0]):
        for j in range(heat_img.shape[1]):
            ax.text(
                j, i,
                "{:2.2f}".format(float(heat_img[i, j])),
                ha="center", va="center", color="w")
    
    ax.set_xticks(
        list(range(len(delta_t_range))),
        labels=["{:2.4f}".format(float(a)) for a in delta_t_range],
        rotation=45)
    ax.set_xlabel("Delta T")
    
    ax.set_yticks(
        list(range(len(tau_range))),
        labels=["{:2.4f}".format(float(a)) for a in tau_range])
    ax.set_ylabel("I Pre and Post Tau")

    fig.savefig('astro_stdp_tau_dt_heat.svg')


def _parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--config', type=str)

    return parser.parse_args()


def _main():
    args = _parse_args()

    with torch.no_grad():
        # cfg = config.Config(args.config)
        # sim_sweep_spike_pairs(cfg, (-8e-3, 8e-3))

        # cfg = config.Config(args.config)
        # sim_heatmap_dt_tau(cfg)

        # cfg = config.Config(args.config)
        # sim_heatmap_alpha_update_rate(cfg)

        # cfg = config.Config(args.config)
        # sim_sweep_io_freq(cfg, (0.1, 0.6))

        cfg = config.Config(args.config)
        sim_classic_stdp(cfg, tau_pre=100, tau_u=1000.0, alpha=1.0)

        cfg = config.Config(args.config)
        mods = torch.linspace(0.5, 1.5, 5)
        sim_shifted_stdp(cfg, tau_pre=100.0, tau_u=1000.0, alpha=1.0, vary=('tau_i_pre', mods))
        cfg = config.Config(args.config)
        sim_shifted_stdp(cfg, tau_pre=100.0, tau_u=1000.0, alpha=1.0, vary=('tau_i_post', mods))
        cfg = config.Config(args.config)
        sim_shifted_stdp(cfg, tau_pre=100.0, tau_u=1000.0, alpha=1.0, vary=('alpha_pre', mods))
        cfg = config.Config(args.config)
        sim_shifted_stdp(cfg, tau_pre=100.0, tau_u=1000.0, alpha=1.0, vary=('alpha_post', mods))
        



if __name__ == '__main__':
    _main()
