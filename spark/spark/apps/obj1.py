import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

from ..module.astrocyte import Astro
from ..utils import config, plot
from ..data import spiketrain
from .lif_astro_net import gen_rate_spikes, gen_group_spikes, sim_lif_astro_net, graph_lif_astro_net
from .astro_spike_pair import sim_astro_probe, graph_dw_dt, graph_astro_tls
from ..experiment import ExpStorage, VSweep, seed_many, load_or_run

# Experiments
######## Plot dw vs. spike dt ########
def _sim_dw_dt_sweep(cfg):
    spike_deltas = torch.linspace(-0.05, 0.05, 50)
    pulse_pair_spikes = spiketrain.pre_post_pair(spike_deltas, cfg['sim']['dt'])
    
    db = ExpStorage()

    for spike_delta, spikes in zip(spike_deltas, pulse_pair_spikes):
        sim_astro_probe(cfg, spikes, db)
        db.last()['delta_t'] = spike_delta

    return (cfg, db)


def _exp_dw_dt_sweep(cfg_path):
    def _vary_cfg():
        astro_params_options = [
            ('Classic STDP', 'classic_stdp'),
            ('Anti STDP', 'anti_stdp'),
            ('LTP Bias', 'ltp_bias'),
            ('LTD Bias', 'ltd_bias'),
            ('LTD dt Shift', 'ltd_dt_shift'),
            ('LTP dt Shift', 'ltp_dt_shift'),
        ]

        # Go through config variants for u_ordered_prod mode
        for descr, key in astro_params_options:
            cfg = config.Config(cfg_path)
            cfg['astro_params'] = cfg[key]
            cfg['astro_params']['u_step_params']['mode'] = 'u_ordered_prod'

            name = "{}_rp".format(key)

            yield descr, name, cfg
            
            cfg['astro_params']['u_step_params']['mode'] = 'stdp'
            name = "{}_tp".format(key)
            
            yield descr, name, cfg


    exp_records = []
    for descr, name, cfg in tqdm(_vary_cfg()):
        cfg, db = _sim_dw_dt_sweep(cfg)
        db.meta['descr'] = descr
        db.meta['name'] = name

        exp_records.append((cfg, db))
    
    return exp_records


def _graph_dw_dt_sweep(exp_records, prefix=''):
    for cfg, db in exp_records:
        fig = graph_dw_dt(cfg, db, title=db.meta['descr'])
        fig.savefig("{}-{}.svg".format(prefix, db.meta['name']))

        # Select subset (5) of db entries to graph timeline
        step = len(db) // 5
        tls = []
        for i in range(0, len(db), step):
            tls.append(db[i])
        fig = graph_astro_tls(tls, 'delta_t', prefix=db.meta['descr'])

        fig.savefig("{}-{}-tl.svg".format(prefix, db.meta['name']))

######## Sweep tau_u ########
def _exp_sweep_tau_u(cfg_path):
    cfg = config.Config(cfg_path)
    cfg['astro_params'] = cfg['classic_stdp']
    
    # Sweep tau_u
    db = ExpStorage()
    all_tau_u = torch.linspace(50, 500, 20)

    # Spikes for sim
    spike_trains = []
    impulse_spikes = spiketrain.impulse(0, 10, 100).repeat((2,1)).transpose(1,0)
    spike_trains.append(impulse_spikes)
    for r in [0.1, 0.5, 0.7]:
        spike_trains.append(
            spiketrain.poisson([r, r], 100).transpose(1,0)
        )
    
    suffix = ['impulse', 'p0.1', 'p0.5', 'p0.7']

    for i, spikes in enumerate(spike_trains):
        for tau_u in all_tau_u:
            cfg('astro_params.tau_u', tau_u)
            sim_astro_probe(cfg, spikes, db)
            db.last()['tau_u'] = tau_u
            db.last()['spikes'] = spikes
            db.last()['suffix'] = suffix[i]

    return db


def _graph_sweep_tau_u(db):
    records_by_spikes = db.group_by('spikes')
    for i, (_, by_spike) in enumerate(records_by_spikes.items()):
        fig = plt.Figure(figsize=(6.4, 7.5))
        fig.suptitle("Astrocyte Response to Different Values of U Tau")

        ax = fig.add_subplot(3, 1, 1)
        ax.set_title("Astrocyte U")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Value")

        # Plot u for each tau on a single plot
        for d in by_spike:
            tl = d['timeline']
            spikes = d['spikes']
            ax.plot(tl['u'], label='tau_u={}'.format(d['tau_u']))
            ax.set_xlim((0, len(tl['z_pre'])))
            # ax.legend()

        # i_pre, i_post, and spikes are the same across varying u
        suffix = by_spike[0]['suffix']
        tl = by_spike[0]['timeline']
        i_pre = tl['i_pre']
        i_post = tl['i_post']

        ax = fig.add_subplot(3,1,2)
        ax.set_title("Astrocyte Input Traces")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Value")
        c1 = ax.plot(i_pre, label='ip3')[0].get_color()
        c2 = ax.plot(i_post, label='k+')[0].get_color()
        ax.legend()

        ax = fig.add_subplot(3,1,3)
        ax.set_ylabel("Spikes")
        ax.set_xlabel("Time (ms)")
        
        plot.plot_events(ax, spikes.transpose(1,0).tolist(), colors=(c1,c2))
        ax.set_title("Pre and Post-synapic Spikes")

        fig.tight_layout()
        fig.savefig("sweep_u_tau_{}.svg".format(suffix))

######## Sweep i_pre ########
def _sweep_pre_alpha_tau_vals(all_tau_ip3, all_alpha_ip3, spike_trains):
    for tau_ip3 in all_tau_ip3:
        for alpha_ip3 in all_alpha_ip3:
            for spikes in spike_trains:
                yield tau_ip3, alpha_ip3, spikes
    
    
def _exp_sweep_pre_alpha_tau(cfg_path):
    cfg = config.Config(cfg_path)
    cfg['astro_params'] = cfg['classic_stdp']
    
    alpha_i_pre_vals = torch.linspace(0.1, 1.0, 3)
    tau_i_pre_vals = torch.logspace(1, 3, 5)

    # Generate spikes
    spike_trains = []
    spikes = spiketrain.impulse(0, 10, 100).transpose(1,0)
    spikes = torch.cat((spikes, torch.zeros(100, 1)), axis=1)
    spike_trains.append(spikes)
    for r in [0.1, 0.5, 0.7]:
        spikes = spiketrain.poisson(r, 100).transpose(1,0)
        spikes = torch.cat((spikes, torch.zeros(100, 1)), axis=1)
        spike_trains.append(spikes)

    db = ExpStorage()

    # Simulation
    for tau_ip3, alpha_ip3, spikes in _sweep_pre_alpha_tau_vals(tau_i_pre_vals, alpha_i_pre_vals, spike_trains):
        cfg('astro_params.tau_i_pre', tau_ip3)
        cfg('astro_params.alpha_pre', alpha_ip3)

        sim_astro_probe(cfg, spikes, db)
        
        db.last()['tau_ip3'] = float(tau_ip3)
        db.last()['alpha'] = float(alpha_ip3)
        db.last()['spikes'] = spikes

    return db


def _graph_sweep_pre_alpha_tau(db):
    records_by_spike = db.group_by('spikes')
    
    for i, (_, by_spike) in enumerate(records_by_spike.items()):
        # One figure per spike train
        fig = plt.Figure(figsize=(6.4, 13))
        fig.suptitle("Astrocyte ip3 Response to Input Spikes With Various tau and alpha")

        # Then split up by alpha
        records_by_alpha = by_spike.group_by('alpha')

        # One subplot per alpha value
        num_subplots = len(records_by_alpha) + 1
        for j, (alpha, by_alpha) in enumerate(records_by_alpha.items()):
            ax = fig.add_subplot(num_subplots, 1, j+1)
            ax.set_title("Alpha={:2.2f}".format(alpha))
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Value")

            # Plot i_pre and u for each tau on a single plot
            for d in by_alpha:
                tl = d['timeline']
                spikes = d['spikes']
                ax.plot(tl['i_pre'].tolist(), label='tau_ip3={}'.format(d['tau_ip3']))
                ax.set_xlim((0, len(tl['z_pre'])))
            ax.legend()

        # Last subplot has spike train
        ax = fig.add_subplot(num_subplots, 1, num_subplots)

        plot.plot_events(ax, [spikes[:, 0].squeeze()])
        ax.set_title("Spikes over time")
        ax.legend(['Z In'])

        fig.tight_layout()
        fig.savefig(
            'sweep-alpha_tau-ip3-spike{:1.3f}.svg'.format(
                torch.mean(spikes)
            )
        )


######## Heatmap: tau_u vs spike rate -> effect waiting time ########
def _exp_heatmap_tau_u_thr_events(cfg_path):
    cfg = config.Config(cfg_path)
    cfg['astro_params'] = cfg['classic_stdp']
    cfg['astro_params']['u_th'] = 10.0
    
    spike_rate_range = torch.linspace(0.05, 0.8, 20)
    tau_range = torch.linspace(50, 500, 20)
    dt = cfg['sim']['dt']

    db = ExpStorage()
    db.meta['tau_range'] = tau_range
    db.meta['spike_rate_range'] = spike_rate_range


    # Simulate
    for spike_rate in spike_rate_range:
        for tau_u in tau_range:
            cfg('astro_params.tau_u', tau_u)

            spikes = spiketrain.poisson([spike_rate]*2, 1000).transpose(1,0)

            sim_astro_probe(cfg, spikes, db)

            db.last()['spike_rate'] = spike_rate
            db.last()['tau_u'] = tau_u

    return db


def _graph_heatmap_tau_u_thr_events(db):
    tau_range = db.meta['tau_range']
    spike_rate_range = db.meta['spike_rate_range']

    fig = plt.Figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    ax.set_title("Mean U Threshold Event Waiting Time Given Poisson Spike Rate vs. U Tau")
    ax.set_yticks(
        list(range(len(tau_range))),
        labels=["{:2.4f}".format(float(a)) for a in tau_range],
        rotation=45)
    ax.set_ylabel('U Tau')

    ax.set_xticks(
        list(range(len(spike_rate_range))),
        labels=["{:2.4f}".format(float(a)) for a in spike_rate_range])
    ax.set_xlabel('Spike Rate')

    heat_img = torch.zeros((len(tau_range), len(spike_rate_range)))

    for i, (spike_rate, tau_u_db) in enumerate(db.group_by('spike_rate', sort=True).items()):
        for j, (alpha, elem_db) in enumerate(tau_u_db.group_by('tau_u', sort=True).items()):
            tl = elem_db[0]['timeline']
            eff = torch.as_tensor(tl['eff'])
            eff = torch.abs(eff - 1.0) * 20.0

            # print("eff: ", eff.mean())
            
            heat_img[j, i] = eff.mean()
            ax.text(
                i, j,
                "{:1.2f}".format(float(heat_img[j, i])),
                ha="center", va="center", color="w")

    ax.imshow(heat_img)
    fig.tight_layout()
    fig.savefig("heatmap_spike-rate_tau_u.svg")


def _exp_heatmap_alpha_thr_events(cfg_path):
    cfg = config.Config(cfg_path)
    cfg['astro_params'] = cfg['classic_stdp']
    cfg['astro_params']['u_th'] = 10.0
    
    spike_rate_range = torch.linspace(0.05, 0.8, 20)
    alpha_range = torch.linspace(0.1, 2.0, 20)
    dt = cfg['sim']['dt']

    db = ExpStorage()
    db.meta['alpha_range'] = alpha_range
    db.meta['spike_rate_range'] = spike_rate_range

    # Simulate
    for spike_rate in spike_rate_range:
        for alpha in alpha_range:
            cfg('astro_params.alpha_pre', alpha)
            cfg('astro_params.alpha_post', alpha)

            spikes = spiketrain.poisson([spike_rate]*2, 1000).transpose(1,0)

            sim_astro_probe(cfg, spikes, db)

            db.last()['spike_rate'] = spike_rate
            db.last()['alpha'] = alpha

    return db


def _graph_heatmap_alpha_thr_events(db):
    alpha_range = db.meta['alpha_range']
    spike_rate_range = db.meta['spike_rate_range']

    fig = plt.Figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    ax.set_title("Mean U Threshold Event Waiting Time Given Poisson Spike Rate vs. Pathway Alphas")
    ax.set_yticks(
        list(range(len(alpha_range))),
        labels=["{:2.4f}".format(float(a)) for a in alpha_range],
        rotation=45)
    ax.set_ylabel('Ip3 and K+ Alpha')

    ax.set_xticks(
        list(range(len(spike_rate_range))),
        labels=["{:2.4f}".format(float(a)) for a in spike_rate_range])
    ax.set_xlabel('Spike Rate')

    heat_img = torch.zeros((len(alpha_range), len(spike_rate_range)))

    for i, (spike_rate, alpha_db) in enumerate(db.group_by('spike_rate', sort=True).items()):
        for j, (alpha, elem_db) in enumerate(alpha_db.group_by('alpha', sort=True).items()):
            tl = elem_db[0]['timeline']
            eff = torch.as_tensor(tl['eff'])
            eff = torch.abs(eff - 1.0) * 20.0

            # print("eff: ", eff.mean())
            
            heat_img[j, i] = eff.mean()
            ax.text(
                i, j,
                "{:1.2f}".format(float(heat_img[j, i])),
                ha="center", va="center", color="w")

    ax.imshow(heat_img)
    fig.tight_layout()
    fig.savefig("heatmap_spike-rate_alpha.svg")


######## Astro-LIF Networks #########
def _exp_rate_learning(cfg_path):
    dbs = []
    with torch.no_grad():
        # Sim w/ out ltd/ltp thresholds
        cfg = config.Config(cfg_path)
        cfg['astro_params'] = cfg['anti_stdp']
        cfg['astro_params']['u_step_params']['ltd'] = 0.0
        cfg['astro_params']['u_step_params']['ltp'] = 0.0

        spikes = gen_rate_spikes(cfg['sim']['steps'])
        db = sim_lif_astro_net(cfg, spikes, name="snn_1n1s1a_rp_no-band")
        dbs.append(db)

        # Sim w/ thresholds
        cfg = config.Config(cfg_path)
        cfg['astro_params'] = cfg['anti_stdp']
        cfg['astro_params']['u_step_params']['ltd'] = -1.5
        cfg['astro_params']['u_step_params']['ltp'] = 1.5

        spikes = gen_rate_spikes(cfg['sim']['steps'])
        db = sim_lif_astro_net(cfg, spikes, name="snn_1n1s1a_rp_band")
        dbs.append(db)

    return dbs


def _exp_average_pulse_pair(cfg_path):
    """
    Simulate an snn in the 1n1s1a configuration, and demonstrate how plasticity
    triggered by a threshold on u can average multiple pulse pairs, and allow
    for confident steps for plasticity, based on multiple events.
    """

    dbs = []
    
    with torch.no_grad():
        cfg = config.Config(cfg_path)
        cfg['astro_params'] = cfg['classic_stdp']

        spikes = gen_group_spikes()

        # Sim w/ baseline
        db = sim_lif_astro_net(cfg, spikes, name="snn_1n1s1a_tp_pulse")
        dbs.append(db)

        # Set u_thr, show that when an input is driving the firing of a downstream spike, it tends to increase the weight
        cfg['astro_params']['u_th'] = 2.5
        db = sim_lif_astro_net(
            cfg,
            spikes,
            name="snn_1n1s1a_tp_stdp_u_thr-{}".format(cfg['astro_params']['u_th'])
        )
        dbs.append(db)

    return dbs


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--astro-spike-pairs', action='store_true')
    parser.add_argument('--astro-param-sweep', action='store_true')
    parser.add_argument('--astro-lif-net', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--sim', action='store_true')

    return parser.parse_args()


def _main(args):
    os.makedirs('./obj1', exist_ok=True)
    os.chdir('./obj1')

    if args.astro_param_sweep or args.all:
        seed_many()
        db = load_or_run(
            lambda: _exp_sweep_tau_u('../../config/astro_stdp.yaml'),
            'exp_sweep_tau_u.db',
            args.sim,
        )
        _graph_sweep_tau_u(db)

        db = load_or_run(
            lambda: _exp_sweep_pre_alpha_tau('../../config/astro_stdp.yaml'),
            'exp_sweep_pre_alpha_tau.db',
            args.sim,
        )
        _graph_sweep_pre_alpha_tau(db)

        db = load_or_run(
            lambda: _exp_heatmap_tau_u_thr_events('../../config/astro_stdp.yaml'),
            'exp_heatmap_tau_u_thr_events.db',
            args.sim
        )
        _graph_heatmap_tau_u_thr_events(db)

        db = load_or_run(
            lambda: _exp_heatmap_alpha_thr_events('../../config/astro_stdp.yaml'),
            'exp_heatmap_alpha_thr_events.db',
            args.sim
        )
        _graph_heatmap_alpha_thr_events(db)


    if args.astro_spike_pairs or args.all:
        seed_many()
        dbs = _exp_dw_dt_sweep('../../config/astro_stdp.yaml')
        _graph_dw_dt_sweep(dbs, prefix='astro_probe_dwdt')


    if args.astro_lif_net or args.all:
        seed_many()
        
        dbs = _exp_average_pulse_pair('../../config/1n1s1a_temporal-learn.yaml')
        for db in dbs:
            graph_lif_astro_net(db)

        dbs =_exp_rate_learning('../../config/1n1s1a_rate-learn.yaml')
        for db in dbs:
            graph_lif_astro_net(db)


if __name__ == '__main__':
    args = _parse_args()

    _main(args)
