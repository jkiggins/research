import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import copy

from pathlib import Path

import torch

from ..module.astrocyte import Astro
from ..utils import config, plot
from ..data import spiketrain

from .lif_astro_net import (
    gen_rate_spikes,
    gen_ramp_impulse_spikes,
    gen_sgnn_axes,
    graph_sgnn,
    sim_lif,
    gen_impulse_spikes,
    sim_lif_astro_net,
    graph_lif_astro_compare,
    gen_dw_w_axes,
    graph_dw_w,
    astro_dwdt_text
)

from .astro_spike_pair import sim_astro_probe, graph_astro_tls
from .lif_astro_net import graph_dw_dt
from ..experiment import ExpStorage, VSweep, seed_many, load_or_run, lif_astro_name, try_load_dbs

# Experiments
######## Plot dw vs. spike dt ########


def _print_sim(name):
    print("\n##### Simulation: {} #####".format(name))


def _sim_dw_dt_sweep(cfg, db):
    spike_deltas = torch.linspace(-0.05, 0.05, 50)
    pulse_pair_spikes = spiketrain.pre_post_pair(spike_deltas, cfg['sim']['dt'])
    
    for spike_delta, spikes in zip(spike_deltas, pulse_pair_spikes):
        spikes = spikes.unsqueeze(-1)
        db = sim_astro_probe(cfg, spikes, db)
        tl = db.last()['timeline']
        eff = tl['eff']
        wh_not_1 = torch.where(eff != 1.0)

        db.last()['delta_t'] = spike_delta
        # db.last()['dw'] = eff[wh_not_1][0] - 1.0
        db.last()['dw'] = tl['max_u']

    return (cfg, db)


def _exp_classic_stdp(cfg_path, sim=False):
    db_name = '_exp_classic_stdp_{}.db'
    
    dbs = []
    if not sim:
        dbs = try_load_dbs(db_name, many=True)
    if len(dbs) > 0:
        return dbs

    cfg = config.Config(cfg_path)
    tau_ltp = cfg['astro_plasticity']['tau_ip3']
    alpha_ltp = cfg['astro_plasticity']['alpha_ip3']
    tau_ltd = cfg['astro_plasticity']['tau_kp']
    alpha_ltd = cfg['astro_plasticity']['alpha_kp']

    spike_deltas = torch.linspace(-0.05, 0.05, 50)

    dw = torch.zeros_like(spike_deltas)
    wh_gt_0 = torch.where(spike_deltas > 0.0)
    wh_lt_0 = torch.where(spike_deltas < 0.0)

    dw[wh_gt_0] = alpha_ltp * torch.exp(-spike_deltas[wh_gt_0]*tau_ltp)
    dw[wh_lt_0] = -alpha_ltd * torch.exp(spike_deltas[wh_lt_0]*tau_ltd)

    db = ExpStorage()
    db.meta['descr'] = 'closed_form_stdp'
    db.meta['name'] = "Closed Form of STDP"
    for dw_i, dt_i in zip(dw, spike_deltas):
        db.store({'delta_t': dt_i, 'dw': dw_i})
    dbs.append(db)

    for i, db in enumerate(dbs):
        db.save(db_name.format(i))

    return dbs
        

def _exp_dw_dt_sweep(cfg_path, sim=False):
    astro_param_sets = [
        ('Classic STDP', 'astro_plasticity'),
        ('Anti STDP', 'anti_stdp'),
        ('LTP Bias', 'ltp_bias'),
        ('LTD Bias', 'ltd_bias'),
        ('LTD dt Shift', 'ltd_dt_shift'),
        ('LTP dt Shift', 'ltp_dt_shift'),
    ]

    db_name = '_exp_dw_dt_sweep_{}.db'
    dbs = []
    if not sim:
        dbs = try_load_dbs(db_name, many=True)
    if len(dbs) > 0:
        return dbs

    for p_name, p_key in astro_param_sets:
        cfg = config.Config(cfg_path)
        # Set params to only explore ca response, and not react to the value
        cfg['astro_params'] = cfg[p_key]
        cfg['astro_params']['ca_th'] = 100000.0
        cfg['astro_params']['local']['ca_thr'] = [0]
        cfg['astro_params']['local']['ordered_prod'] = [0]
        cfg['astro_params']['local']['stdp'] = None
        cfg['astro_params']['dw'] = 'dw_mult'

        # Sims with rate-based response
        db = ExpStorage()
        db.meta['descr'] = "1n1s1a_rp_{}_dwdt_{}".format(lif_astro_name(cfg['astro_params']), p_key)
        db.meta['name'] = p_name
        db.meta['graph_text'] = astro_dwdt_text(cfg, ordered_prod=True)
        _sim_dw_dt_sweep(cfg, db)
        dbs.append(db)

        # Sims with temporal response
        db = ExpStorage()
        cfg['astro_params']['ca_th'] = 100000.0
        cfg['astro_params']['local']['ordered_prod'] = None
        cfg['astro_params']['local']['stdp'] = [0]
        db.meta['descr'] = "1n1s1a_tp_{}_dwdt_{}".format(lif_astro_name(cfg['astro_params']), p_key)
        db.meta['name'] = p_name
        db.meta['graph_text'] = astro_dwdt_text(cfg, stdp=True)

        _sim_dw_dt_sweep(cfg, db)
        dbs.append(db)

    for i, db in enumerate(dbs):
        db.save(db_name.format(i))

    return dbs


######## Sweep tau_ca ########
def _exp_sweep_tau_ca(cfg_path, sim=False):
    db_name = '_exp_sweep_tau_ca.db'
    
    cfg = config.Config(cfg_path)
    cfg['astro_params'] = cfg['astro_plasticity']
    cfg['astro_params']['local']['stdp'] = [0]
    
    # Sweep tau_ca
    if not sim:
        db = try_load_dbs(db_name)
        if db:
            return db
    db = ExpStorage()

    all_tau_ca = torch.linspace(50, 500, 20)

    # Spikes for sim
    spike_trains = []
    impulse_spikes = spiketrain.impulse(0, 10, 100).repeat((2,1)).transpose(1,0)
    impulse_spikes = impulse_spikes.unsqueeze(-1)

    spike_trains.append(impulse_spikes)
    for r in [0.1, 0.5, 0.7]:
        spike_trains.append(
            spiketrain.poisson([r, r], 100).transpose(1,0).unsqueeze(-1)
        )
    
    suffix = ['impulse', 'p0.1', 'p0.5', 'p0.7']

    for i, spikes in enumerate(spike_trains):
        for tau_ca in tqdm(all_tau_ca, desc='tau_ca'):
            cfg('astro_params.tau_ca', tau_ca)
            sim_astro_probe(cfg, spikes, db)
            db.last()['tau_ca'] = tau_ca
            db.last()['spikes'] = spikes
            db.last()['suffix'] = suffix[i]

    db.save(db_name)
    
    return db


def _graph_sweep_tau_ca(db):
    records_by_spikes = db.group_by('spikes')
    for i, (_, by_spike) in enumerate(records_by_spikes.items()):
        fig = plt.Figure(figsize=(6.4, 7.5))
        gs = plot.gs(2,1)
        fig, axes = plot.gen_axes(
            ('astro', gs[0]),
            ('spikes', gs[1]),
            figsize=(15,10),
        )
        fig.suptitle("Astrocyte Response to Different Values of $\tau_{{ca}}$")

        suffix = by_spike[0]['suffix']

        # Plot Ca for each tau on a single plot
        first = True
        for d in by_spike:
            tl = d['timeline']
            spikes = d['spikes']

            if first:
                plot.plot_astro(
                    axes, ('astro',),
                    tl['ip3'],tl['kp'],tl['ca'],
                    None,None
                )
                first = False
            else:
                plot.plot_astro(
                    axes, ('astro',),
                    None, None, tl['ca'],
                    None,None,
                    no_legend=True
                )

        plot.plot_spikes(
            axes, ('spikes',),
            tl['z_pre'], tl['z_post']
        )

        fig_path = "sweep_u_tau_{}.svg".format(suffix)
        print("Saving: ", fig_path)
        fig.tight_layout()
        fig.savefig(fig_path)

######## Sweep ip3 ########
def _sweep_pre_alpha_tau_vals(all_tau_ip3, all_alpha_ip3, spike_trains):
    for tau_ip3 in all_tau_ip3:
        for alpha_ip3 in all_alpha_ip3:
            for spikes in spike_trains:
                yield tau_ip3, alpha_ip3, spikes
    
    
def _exp_sweep_pre_alpha_tau(cfg_path, sim=False):
    db_name = '_exp_sweep_pre_alpha_tau.db'
    if not sim:
        db = try_load_dbs(db_name)
        if db:
            return db
    db = ExpStorage()
    
    cfg = config.Config(cfg_path)
    cfg['astro_params'] = cfg['astro_plasticity']
    cfg['astro_params']['local']['stdp'] = [0]
    
    alpha_ip3_vals = torch.linspace(0.1, 1.0, 3)
    tau_ip3_vals = torch.logspace(1, 3, 5)

    # Generate spikes
    spike_trains = []
    spikes = spiketrain.impulse(0, 10, 100).transpose(1,0)
    spikes = torch.cat((spikes, torch.zeros(100, 1)), axis=1)
    spikes = spikes.unsqueeze(-1)

    spike_trains.append(spikes)
    for r in [0.1, 0.5, 0.7]:
        spikes = spiketrain.poisson(r, 100).transpose(1,0)
        spikes = spikes.unsqueeze(-1)
        
        spikes = torch.cat((spikes, torch.zeros(100, 1, 1)), axis=1)
        spike_trains.append(spikes)


    # Simulation
    sweep_params = _sweep_pre_alpha_tau_vals(tau_ip3_vals, alpha_ip3_vals, spike_trains)
    for tau_ip3, alpha_ip3, spikes in tqdm(sweep_params, desc='ip3-alpha/tau'):
        cfg('astro_params.tau_ip3', tau_ip3)
        cfg('astro_params.alpha_ip3', alpha_ip3)

        sim_astro_probe(cfg, spikes, db)
        
        db.last()['tau_ip3'] = float(tau_ip3)
        db.last()['alpha'] = float(alpha_ip3)
        db.last()['spikes'] = spikes

    db.save(db_name)
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
            ax.set_ylabel("[IP3],[K+]")

            # Plot ip3 and u for each tau on a single plot
            for d in by_alpha:
                tl = d['timeline']
                spikes = d['spikes']
                ax.plot(tl['ip3'].tolist(), label='tau_ip3={}'.format(d['tau_ip3']))
                ax.set_xlim((0, len(tl['z_pre'])))
            ax.legend()

        # Last subplot has spike train
        ax = fig.add_subplot(num_subplots, 1, num_subplots)

        plot.plot_events(ax, [spikes[:, 0].squeeze()])
        ax.set_title("Spikes over time")
        ax.legend(['Z In'])

        fig_path = 'sweep-alpha_tau-ip3-spike{:1.3f}.svg'.format(torch.mean(spikes))
        print("Saving: ", fig_path)
        fig.tight_layout()
        fig.savefig(fig_path)  


######## Heatmap: tau_ca vs spike rate -> effect waiting time ########
def _exp_heatmap_tau_ca_thr_events(cfg_path, sim=False):
    db_name = '_exp_heatmap_tau_ca_thr_events.db'
    if not sim:
        db = try_load_dbs(db_name)
        if db:
            return db
    db = ExpStorage()

    cfg = config.Config(cfg_path)
    cfg['astro_params'] = cfg['astro_plasticity']
    cfg['astro_params']['ca_th'] = 2.5
    cfg['astro_params']['local']['stdp'] = [0]
    cfg['astro_params']['local']['ca_thr'] = [0]
    cfg['astro_params']['dw'] = 'dw_mult'
    
    spike_rate_range = torch.linspace(0.05, 0.8, 20)
    tau_range = torch.linspace(50, 500, 20)
    dt = cfg['sim']['dt']

    db = ExpStorage()
    db.meta['tau_range'] = tau_range
    db.meta['spike_rate_range'] = spike_rate_range


    # Simulate
    for spike_rate in tqdm(spike_rate_range, desc='spike-rate/tau'):
        for tau_ca in tau_range:
            cfg('astro_params.tau_ca', tau_ca)

            spikes = spiketrain.poisson([spike_rate]*2, 1000).transpose(1,0)
            spikes = spikes.unsqueeze(-1)

            sim_astro_probe(cfg, spikes, db)

            db.last()['spike_rate'] = spike_rate
            db.last()['tau_ca'] = tau_ca

    db.save(db_name)
    return db


def _graph_heatmap_tau_ca_thr_events(db):
    tau_range = db.meta['tau_range']
    spike_rate_range = db.meta['spike_rate_range']

    fig = plt.Figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    ax.set_title("Mean time for $Ca^{2+} > thr_{{ca}}$ Given Poisson Spike Rate vs. $\tau_{ca}$")
    ax.set_yticks(
        list(range(len(tau_range))),
        labels=["{:2.4f}".format(float(a)) for a in tau_range],
        rotation=45)
    ax.set_ylabel('$\tau_{ca}$')

    ax.set_xticks(
        list(range(len(spike_rate_range))),
        labels=["{:2.4f}".format(float(a)) for a in spike_rate_range])
    ax.set_xlabel('Spike Rate')

    heat_img = torch.zeros((len(tau_range), len(spike_rate_range)))

    for i, (spike_rate, tau_ca_db) in enumerate(db.group_by('spike_rate', sort=True).items()):
        for j, (alpha, elem_db) in enumerate(tau_ca_db.group_by('tau_ca', sort=True).items()):
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

    fig_path = "heatmap_spike-rate_tau_ca.svg"
    print("Saving: ", fig_path)
    fig.tight_layout()
    fig.savefig(fig_path)


def _exp_heatmap_alpha_thr_events(cfg_path, sim=False):
    db_name = '_exp_heatmap_alpha_thr_events.db'
    if not sim:
        db = try_load_dbs(db_name)
        if db:
            return db
    db = ExpStorage()

    cfg = config.Config(cfg_path)
    cfg['astro_params'] = cfg['astro_plasticity']
    cfg['astro_params']['local']['ca_thr'] = [0]
    cfg['astro_params']['local']['stdp'] = [0]
    cfg['astro_params']['ca_th'] = 10.0
    cfg['astro_params']['dw'] = 'dw_mult'
    
    spike_rate_range = torch.linspace(0.05, 0.8, 20)
    alpha_range = torch.linspace(0.1, 2.0, 20)
    dt = cfg['sim']['dt']

    db = ExpStorage()
    db.meta['alpha_range'] = alpha_range
    db.meta['spike_rate_range'] = spike_rate_range

    # Simulate
    for spike_rate in tqdm(spike_rate_range, desc='spike-rate/alpha'):
        for alpha in alpha_range:
            cfg('astro_params.alpha_ip3', alpha)
            cfg('astro_params.alpha_kp', alpha)

            spikes = spiketrain.poisson([spike_rate]*2, 1000).transpose(1,0)
            spikes = spikes.unsqueeze(-1)

            sim_astro_probe(cfg, spikes, db)

            db.last()['spike_rate'] = spike_rate
            db.last()['alpha'] = alpha

    db.save(db_name)
    return db


def _graph_heatmap_alpha_thr_events(db):
    alpha_range = db.meta['alpha_range']
    spike_rate_range = db.meta['spike_rate_range']

    fig = plt.Figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    ax.set_title("Mean time for $Ca^{2+} > thr_{ca}$ Given Poisson Spike Rate and $alpha_{ip3/k+}$")
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

    fig_path = "heatmap_spike-rate_alpha.svg"
    print("Saving: ", fig_path)
    fig.tight_layout()
    fig.savefig(fig_path)


def _graph_exp_tp_w_sweep(db):
    records_by_ca_thr = db.group_by('ca_th')
    if len(records_by_ca_thr) == 0:
        raise ValueError("No records found when grouping by ca_th")

    fig, axes = gen_dw_w_axes(len(records_by_ca_thr), size=(10, 10))

    for i, (ca_thr, by_ca_thr) in enumerate(records_by_ca_thr.items()):
        fig, axes = graph_dw_w(
            by_ca_thr, fig, axes,
            title='Astrocyte $Ca^{{2+}}$ Response with $thr_{{ca}}$ = {:4.2f}'.format(ca_thr),
            sp=i
        )
   
    fig_path = "{}_sweep.svg".format(db.meta['descr'])
    print("Saving: ", fig_path)
    fig.tight_layout()
    fig.savefig(fig_path)


def _graph_exp_rp_w_sweep(db):
    records_by_band = db.group_by('band')
    if len(records_by_band) == 0:
        raise ValueError("No records found when grouping by ltp/ltd band")

    fig, axes = gen_dw_w_axes(len(records_by_band), size=(10, 10))

    for i, (band, by_band) in enumerate(records_by_band.items()):
        fig, axes = graph_dw_w(
            by_band, fig, axes,
            title='Simulation With $thr_{{ltp/ltd}}={:4.2f}$'.format(band),
            sp=i)

    fig_path = "{}_sweep.svg".format(db.meta['descr'])
    print("Saving: ", fig_path)

    fig.tight_layout()
    fig.savefig(fig_path)


def _graph_exp_w_tl(dbs, xlim=None, ip3_kp=False):
    """
    For each db graph on different subplot, the calcium timeline,
    with synaptic weight value in the title
    """

    fig, axes = None, None

    plot = ['spikes']
    if ip3_kp:
        graphs=['astro']*len(dbs)
        plot.append('astro')
    else:
        graphs=['astro-ca']*len(dbs)
        plot.append('astro-ca')
        
    graphs.append('spikes')

    fig, axes = gen_sgnn_axes(1, graphs, offset=True)

    axes['spikes'][0].set_title("Pre and Post-Synaptic Spikes")

    # Each db gets its own subplot on the graph
    for i, db in enumerate(dbs):

        # Set title
        axes[plot[1]][i].set_title(db.meta['title'])
        
        # Each db contains multiple runs with different weight values
        # Graph the timelines offset, on the same axis
        records_by_w = db.filter(tl_graph=True).group_by('w')

        if len(records_by_w) == 0:
            raise ValueError("No records found with tl_graph=True")

        for j, (w, by_w) in enumerate(records_by_w.items()):
            assert len(by_w) == 1

            if hasattr(w, '__len__'):
                prefix = ['{:4.2f}'.format(w_i) for w_i in w]
                prefix = "w=" + ",".join(prefix)
            else:
                prefix = 'w={:4.2f}'.format(w)

            if i == 0:
                graph_sgnn(
                    by_w[0],
                    fig,
                    axes,
                    i,
                    plot=plot,
                    prefix=prefix)
            else:
                graph_sgnn(
                    by_w[0],
                    fig,
                    axes,
                    i,
                    plot=['astro-ca'],
                    prefix=prefix)

        if not (xlim is None):
            if 'astro-ca' in axes:
                axes['astro-ca'][i].set_xlim(*xlim)
            if 'astro' in axes:
                axes['astro'][i].set_xlim(*xlim)

    fig_path = "{}_tl.svg".format(db.meta['descr'])
    if not (xlim is None):
        axes['spikes'][0].set_xlim(*xlim)
        fig_path = "{}_tl.svg".format(db.meta['descr'] + '_xlim')
        
    print("Saving: ", fig_path)
    fig.savefig(fig_path)


######## Astro-W Networks #########
def _sim_rate_w_impulse(cfg, spikes, db, tl_graph_idx, all_w, desc=""):
    for i, w in enumerate(tqdm(all_w, desc=desc)):
        cfg['linear_params']['mu'] = w

        if i in tl_graph_idx:
            db.prefix({'tl_graph': True})
        else:
            db.prefix({'tl_graph': False})
        
        sim_lif_astro_net(cfg, spikes, db, dw=False)

        db.last()['ca-act'] = (db.last()['tl']['ca']).sum()


def _exp_rate_w_impulse(cfg_path, sim=False):

    db_name = "_exp_rate_w_impulse_{}.db"

    dbs = []
    if not sim:
        dbs = try_load_dbs(db_name, many=True)
        if len(dbs) > 0:
            return dbs

    with torch.no_grad():
        # Sim w/ out ltd/ltp thresholds
        cfg = config.Config(cfg_path)
        cfg['astro_params'] = cfg['anti_stdp']
        cfg['astro_params']['ca_th'] = 2.5
        cfg['astro_params']['local']['ca_thr'] = [0]
        cfg['astro_params']['dw'] = 'dw_mult'
        cfg['astro_params']['local']['ordered_prod'] = [0]

        cfg['astro_params']['ordered_prod']['ltd'] = 0.0
        cfg['astro_params']['ordered_prod']['ltp'] = 0.0

        spikes = gen_rate_spikes([
            (0.3, cfg['sim']['steps'])
        ])

        all_w = torch.linspace(0.7, 15, 100)
        tl_graphs = torch.as_tensor([2, 4, 10])
        tl_graph_idx = torch.argmin(torch.abs(all_w.reshape(-1, 1) - tl_graphs), axis=0).tolist()

        all_band = torch.linspace(1.5, 2.0, 3)

        db = ExpStorage()
        db.meta['descr'] = "astro_rp_many-w"
        db.meta['title'] = "Astrocyte Response given $w$ with $thr_{{ltp/ltd}}=0$"

        _sim_rate_w_impulse(cfg, spikes, db, tl_graph_idx, all_w, 'Astro Sweep W (no tol)')
        print("db tl_graph len: ", len(db.filter(tl_graph=True)))
        dbs.insert(0, db)

        # Sim w/ thresholds
        cfg['astro_params']['ordered_prod']['ltd'] = -1.75
        cfg['astro_params']['ordered_prod']['ltp'] = 1.75

        db = ExpStorage()
        db.meta['descr'] = "astro_rp_many-w"
        db.meta['title'] = \
            "Astrocyte Response given $w$ with $thr_{{ltp/ltd}}={}$".format(
                cfg['astro_params']['ordered_prod']['ltp'])

        for i, w in enumerate(tqdm(all_w, desc='Astro w/band sweep')):
            cfg['linear_params']['mu'] = w
            for j, band in enumerate(all_band):
                cfg['astro_params']['ordered_prod']['ltd'] = -band
                cfg['astro_params']['ordered_prod']['ltp'] = band

                if i in tl_graph_idx and band == 1.75:
                    db.prefix({'band': band, 'tl_graph': True})
                else:
                    db.prefix({'band': band})

                sim_lif_astro_net(cfg, spikes, db, dw=False)
                db.last()['ca-act'] = db.last()['tl']['ca'].sum()

        print("db tl_graph len: ", len(db.filter(tl_graph=True)))
        dbs.insert(0, db)

    for i, db in enumerate(dbs):
        db_path = Path(db_name.format(i))
        db.save(db_path)

    return dbs


def _exp_astro_impulse(cfg_path):
    with torch.no_grad():
        cfg = config.Config(cfg_path)
        cfg['astro_params'] = cfg['astro_plasticity']
        cfg['astro_params']['local']['stdp'] = [0]
        cfg['astro_params']['local']['ca_thr'] = [0]
        cfg['astro_params']['dw'] = 'dw_mult'

        spikes = gen_ramp_impulse_spikes()

        db = ExpStorage()
        db.meta['descr'] = "astro_tp_impulse"
        db.meta['title'] = "Astrocyte Response to Ramping Impulse Input"

        sim_lif_astro_net(cfg, spikes, db, dw=False)

    return db


def _exp_astro_poisson(cfg_path):
    with torch.no_grad():
        cfg = config.Config(cfg_path)
        cfg['astro_params'] = cfg['astro_plasticity']
        cfg['astro_params']['local']['stdp'] = [0]
        cfg['astro_params']['local']['ca_thr'] = [0]
        cfg['astro_params']['dw'] = 'dw_mult'

        spikes = [spiketrain.poisson(0.75, 20).transpose(1,0)]

        db = ExpStorage()
        db.meta['descr'] = "astro_tp_poisson"
        db.meta['title'] = "Astrocyte Response to Random Poisson Input"

        sim_lif_astro_net(cfg, spikes, db, dw=False)

    return db


def _sim_pulse_pair_w_impulse(cfg, db, spikes, all_w, all_ca_thr, tl_graph_idx, tl_ca):
    for i, w in enumerate(tqdm(all_w, desc='Astro Sweep W (thr)')):
        for j, ca_thr in enumerate(all_ca_thr):
            if (i in tl_graph_idx) and ca_thr == tl_ca:
                db.prefix({'ca_th': ca_thr, 'tl_graph': True})
            else:
                db.prefix({'ca_th': ca_thr})

            cfg['linear_params']['mu'] = w
            cfg['astro_params']['ca_th'] = ca_thr
            sim_lif_astro_net(cfg, spikes, db, dw=False)

            db.last()['ca-act'] = (db.last()['tl']['eff'] - 1.0).sum()
            

def _exp_pulse_pair_w_impulse(
    cfg_path,
    sim=False,
    poisson_impulse=False,
    tl_w = [0.8112, 1.0, 1.2],
    tl_ca_thr = 2.5
):
    """
    Simulate an snn in the 1n1s1a configuration, and demonstrate how plasticity
    triggered by a threshold on u can average multiple pulse pairs, and allow
    for confident steps for plasticity, based on multiple events.
    """
    
    db_path = "_exp_pulse_pair_w"
    if poisson_impulse:
        db_path += "_poisson"
    db_path += "_impulse"
    
    db_path = Path(db_path + ".db")

    if not sim:
        db = try_load_dbs(db_path)
        if db:
            return db

    with torch.no_grad():
        cfg = config.Config(cfg_path)
        cfg['astro_params'] = cfg['astro_plasticity']
        cfg['astro_params']['local']['stdp'] = [0]
        cfg['astro_params']['local']['ca_thr'] = [0]
        cfg['astro_params']['dw'] = 'dw_mult'

        db = ExpStorage()
        db.meta['descr'] = "astro_tp_many-w"

        db.meta['title'] = "Astrocyte Response to Impulse input given $w$ with $thr_{{ca}}={:4.2f}$".format(tl_ca_thr)

        if poisson_impulse:
            db.meta['descr'] += '_poisson'

        spikes = gen_impulse_spikes(10, num_impulses=15, poisson=poisson_impulse)
        all_w = torch.linspace(0.75, 2.0, 100)
        tl_graphs = torch.as_tensor(tl_w)

        all_ca_thr = torch.linspace(2.0, 3.0, 3)
        # all_ca_thr = torch.linspace(100.0, 100.0, 1)

        w_graph_idx = torch.argmin(torch.abs(all_w.reshape(-1, 1) - tl_graphs), axis=0).tolist()

        _sim_pulse_pair_w_impulse(
            cfg,
            db,
            spikes,
            all_w, all_ca_thr,
            w_graph_idx, tl_ca_thr
        )

    print("Saving: ", db_path)
    db.save(db_path)

    return db


def _exp_lif_sample(cfg_path, sim=False):

    dbs = []
    
    base_name = '_exp_lif_sample_{}.db'
    if not sim:
        dbs = try_load_dbs(base_name, many=True)

    if len(dbs) > 0:
        return dbs
    
    cfg = config.Config(cfg_path)
    cfg['linear_params']['mu'] = 1.0
    spikes = gen_rate_spikes([
        (0.05, cfg['sim']['steps'])
    ])

    mem_tau_range = [cfg['lif_params']['tau_mem']]
    mem_tau_range.append(mem_tau_range[0] * 0.5)
    mem_tau_range.append(700)

    syn_tau_range = [cfg['lif_params']['tau_syn']]
    syn_tau_range.append(syn_tau_range[0] * 0.5)

    params = itertools.product(mem_tau_range, syn_tau_range)


    with torch.no_grad():
        for tau_mem, tau_syn in params:
            cfg['lif_params']['tau_mem'] = tau_mem
            cfg['lif_params']['tau_syn'] = tau_syn

            db = sim_lif(
                cfg,
                spikes[0],
                name='lif_sample_mem-{}_syn-{}'.format(tau_mem, tau_syn),
            )

            dbs.append(db)

    for i, db in enumerate(dbs):
        db.save(base_name.format(i))

    return dbs


def _graph_lif_sample(db, xlim=None):
    name = db.meta['name']

    assert len(db) == 1

    # Generate and arrage Figure
    gs = plot.gs(2,1)
    fig, axes = plot.gen_axes(
        ('lif', gs[0]),
        ('spikes', gs[1]),
    )
    fig.suptitle("LIF Neuron Response")
    
    # Traces
    tl = db[0]['tl']
    v_psp = tl['i_n']
    v_mem = tl['v_n']
    z_pre = tl['z_pre']
    z_post = tl['z_post']

    plot.plot_spikes(axes, ('spikes',), z_pre, z_post)
    plot.plot_lif(axes, ('lif',), v_psp, v_mem)

    if not (xlim is None):
        plot.xlim(axes, xlim)

    fig_path = "{}.svg".format(name)
    print("Saving: ", fig_path)
    fig.tight_layout()
    fig.savefig(fig_path)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lif', action='store_true')
    parser.add_argument('--stdp', action='store_true')
    parser.add_argument('--astro-spike-pairs', action='store_true')
    parser.add_argument('--astro-param-sweep', action='store_true')
    parser.add_argument('--astro-weight-sweep', action='store_true')
    parser.add_argument('--astro-impulse', action='store_true')
    parser.add_argument('--astro-poisson', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--sim', action='store_true')

    return parser.parse_args()


def _main(args):
    os.makedirs('./obj1', exist_ok=True)
    os.chdir('./obj1')

    cfg_path = '../../config/1n1s1a_obj1.yaml'

    if args.lif or args.all:
        seed_many()
        dbs = _exp_lif_sample(cfg_path, sim=args.sim)
        for db in dbs:
            _graph_lif_sample(db, xlim=(0,60))

    if args.stdp or args.all:
        seed_many()
        dbs = _exp_classic_stdp(cfg_path, sim=args.sim)
        
        for db in dbs:
            fig, axes = graph_dw_dt(db, title=db.meta['name'])

            fig_path = "{}.svg".format(db.meta['descr'])
            print("Saving: ", fig_path)
            fig.tight_layout()
            fig.savefig(fig_path)

    if args.astro_param_sweep or args.all:
        _print_sim("Astro Param Sweep")
        seed_many()
        db = _exp_sweep_tau_ca(cfg_path, sim=args.sim)
        _graph_sweep_tau_ca(db)

        db = _exp_sweep_pre_alpha_tau(cfg_path, sim=args.sim)
        _graph_sweep_pre_alpha_tau(db)

        db = _exp_heatmap_tau_ca_thr_events(cfg_path, sim=args.sim)
        _graph_heatmap_tau_ca_thr_events(db)

        db = _exp_heatmap_alpha_thr_events(cfg_path, sim=args.sim)
        _graph_heatmap_alpha_thr_events(db)


    if args.astro_spike_pairs or args.all:
        _print_sim("Astro Spike Pairs")
        seed_many()
        dbs = _exp_dw_dt_sweep(cfg_path, sim=args.sim)

        fig = None
        for db in dbs:
            fig, axes = graph_dw_dt(
                db,
                title=db.meta['name'],
                graph_text=db.meta['graph_text'],
            )

            fig_path = "{}.svg".format(db.meta['descr'])
            print("Saving: ", fig_path)
            fig.tight_layout()
            fig.savefig(fig_path)


        fig = None
        axes = None
        for db in dbs:
            fig, axes = graph_dw_dt(
                db,
                title=db.meta['name'],
                graph_text=db.meta['graph_text'],
                fig=fig, axes=axes
            )

        fig_path = "astro_dw_dt_all.svg"
        print("Saving: ", fig_path)
        fig.tight_layout()
        fig.savefig(fig_path)


    if args.astro_weight_sweep or args.all:
        _print_sim("LIF/Astro Weight Sweep")
        seed_many()
        
        # dbs = _exp_rate_w_impulse(cfg_path, sim=args.sim)
        # _graph_exp_w_tl(dbs)
        # _graph_exp_rp_w_sweep(dbs[0])

        # db = _exp_pulse_pair_w_impulse(cfg_path, sim=args.sim)

        db = _exp_pulse_pair_w_impulse(
            cfg_path,
            tl_w = [1.0, 1.2, 1.8],
            tl_ca_thr = 2.5,
            sim=args.sim)
        _graph_exp_w_tl([db])
        _graph_exp_w_tl([db], xlim=(580, 640))
        _graph_exp_tp_w_sweep(db)

        # db = _exp_pulse_pair_w_impulse(
        #     cfg_path,
        #     tl_w = [1.0, 1.2, 1.8],
        #     tl_ca_thr = 2.5,
        #     sim=args.sim,
        #     poisson_impulse=True)
        # _graph_exp_w_tl([db])
        # _graph_exp_w_tl([db], xlim=(655, 670))
        # _graph_exp_tp_w_sweep(db)

        # db = _exp_pulse_pair_w_impulse(cfg_path, sim=args.sim)
        # _graph_exp_tp_w_sweep(db)


    if args.astro_impulse or args.all:
        _print_sim("LIF/Astro Impulse, No Dw")
        seed_many()

        db = _exp_astro_impulse(cfg_path)

        assert len(db) == 1

        fig, axes = gen_sgnn_axes(1)
        graph_sgnn(db[0], fig, axes, 0)
        fig_path = "{}.svg".format(db.meta['descr'])
        print("Saving: ", fig_path)
        fig.tight_layout()
        fig.savefig(fig_path)


    if args.astro_poisson or args.all:
        _print_sim("LIF/Astro Poisson, No Dw")
        seed_many()

        db = _exp_astro_poisson(cfg_path)

        assert len(db) == 1

        fig, axes = gen_sgnn_axes(1, graphs=['spikes','astro'])
        graph_sgnn(db[0], fig, axes, 0, plot=['spikes','astro'])
        fig_path = "{}.svg".format(db.meta['descr'])
        print("Saving: ", fig_path)
        fig.tight_layout()
        fig.savefig(fig_path)


if __name__ == '__main__':
    args = _parse_args()

    with torch.no_grad():
        _main(args)
