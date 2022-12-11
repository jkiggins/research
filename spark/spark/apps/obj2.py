import argparse
import os
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import torch

from ..module.astrocyte import Astro
from ..utils import config, plot
from ..data import spiketrain
from .lif_astro_net import (
    gen_rate_spikes,
    gen_ramp_impulse_spikes,
    gen_impulse_spikes,
    sim_lif_astro_net,
    gen_sgnn_axes,
    graph_sgnn
)

from .astro_spike_pair import sim_astro_probe, graph_dw_dt, graph_astro_tls
from ..experiment import (
    ExpStorage,
    VSweep,
    seed_many,
    try_load_dbs,
    try_load_dbs_pairs
)

def _print_sim(name):
    print("##### Simulation: {} #####".format(name))


# def _sim_stdp_and_astro(cfg, spikes, db_astro, db_stdp, descr):
#     db_stdp = _sim_stdp(cfg, spikes, descr)
#     db_astro = _sim_astro(cfg, spikes, descr)

#     return [db_stdp, db_astro]


def _config_astro(cfg, ca_th=2.5):
    cfg['astro_params']['local']['stdp'] = [0]
    cfg['astro_params']['local']['ca_thr'] = [0]

    cfg['astro_params']['ca_th'] = ca_th

    
def _config_stdp(cfg):
    cfg['astro_params']['local']['stdp'] = [0]
    cfg['astro_params']['local']['ca_thr'] = [0]
    
    cfg['astro_params']['ca_th'] = 0.0


def _config_rp(cfg, band):
    cfg['astro_params'] = cfg['anti_stdp']
    cfg['astro_params']['local']['ordered_prod'] = [0]
    cfg['astro_params']['local']['ca_thr'] = [0]
    cfg['astro_params']['ca_th'] = 2.5
    cfg['astro_params']['dw'] = 'dw_mult'
    cfg['astro_params']['dw_mult']['prop_ca'] = False
    cfg['astro_params']['ordered_prod']['ltd'] = -band
    cfg['astro_params']['ordered_prod']['ltp'] = band

    return cfg


def _exp_average_pulse_pair_sweep(cfg_path, sim=False):
    """
    Simulate an snn in the 1n1s1a configuration, and demonstrate how plasticity
    triggered by a threshold on u can average multiple pulse pairs, and allow
    for confident steps for plasticity, based on multiple events.

    See how sweeping some different parameters effect things
    * mu for synaptic weight initialization
    """

    db_name = '_exp_average_pulse_pair_sweep_{}_{}.db'
    dbs = []
    all_dbs_types = ['mu', 'assoc', 'tau', 'dw_factor', 'v_th', 'ca_th']
                     
    if not sim:
        dbs = try_load_dbs_pairs(db_name, all_dbs_types, ['astro', 'stdp'])

    if len(dbs) > 0:
        return dbs


    with torch.no_grad():
        cfg = config.Config(cfg_path)

        ## Simulate for various values of mu
        _print_sim("Initial Average Weight")
        # spikes = gen_ramp_impulse_spikes()
        spikes = gen_impulse_spikes(10, num_impulses=15)
        all_mu = torch.linspace(0.2, 1.0, 7)

        # Create db objects
        db_stdp = ExpStorage()
        db_stdp.meta['descr'] = 'snn_1n1s1a_tp_mu_sweep'
        db_astro = ExpStorage()
        db_astro.meta['descr'] = 'snn_1n1s1a_tp_mu_sweep'

        for mu in tqdm(all_mu, desc="mu: "):
            cfg['linear_params']['mu'] = mu

            cfg['astro_params'] = cfg['astro_plasticity']
            _config_astro(cfg)
            db_astro.prefix({'mu': mu})
            sim_lif_astro_net(cfg, spikes, db_astro)

            cfg['astro_params'] = cfg['astro_plasticity']
            _config_stdp(cfg)
            db_stdp.prefix({'mu': mu})
            sim_lif_astro_net(cfg, spikes, db_stdp)

        dbs.append(('mu', db_astro, db_stdp))

        ## Simulate different spike associations with just STDP
        _print_sim("Spike Associations")
        cfg = config.Config(cfg_path)

        db_stdp = ExpStorage()
        db_stdp.meta['descr'] = 'snn_1n1s1a_tp_sweep_association'
        db_astro = ExpStorage()
        db_astro.meta['descr'] = 'snn_1n1s1a_tp_sweep_association'
        cfg['astro_params'] = cfg['astro_plasticity']

        # Nearest Neighbor
        cfg['astro_plasticity']['pre_reset_on_spike'] = True
        cfg['astro_plasticity']['post_reset_on_spike'] = True
        db_stdp.prefix({'assoc': 'nn'})
        db_astro.prefix({'assoc': 'nn'})
        _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

        # Nearest Pre
        cfg['astro_plasticity']['pre_reset_on_spike'] = True
        cfg['astro_plasticity']['post_reset_on_spike'] = False
        db_stdp.prefix({'assoc': 'nn-pre'})
        db_astro.prefix({'assoc': 'nn-pre'})
        _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

        # Nearest Post
        cfg['astro_plasticity']['pre_reset_on_spike'] = False
        cfg['astro_plasticity']['post_reset_on_spike'] = True

        db_stdp.prefix({'assoc': 'nn-post'})
        db_astro.prefix({'assoc': 'nn-post'})
        _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

        # Any
        cfg['astro_plasticity']['pre_reset_on_spike'] = False
        cfg['astro_plasticity']['post_reset_on_spike'] = False
        db_stdp.prefix({'assoc': 'many-many'})
        db_astro.prefix({'assoc': 'many-many'})
        _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

        dbs.append(('assoc', db_astro, db_stdp))


        ## Simulate with different values for tau_ip3 and tau_k+
        _print_sim("Tau ip3 and Tau K+")
        cfg = config.Config(cfg_path)
        cfg['astro_params'] = cfg['astro_plasticity']
        def_tau = cfg['astro_params']['tau_ip3']
        all_tau = torch.linspace(10, 800, 10)

        # Different descr only needed when each db will result in a separate graph
        db_stdp = ExpStorage()
        db_stdp.meta['descr'] = 'snn_1n1s1a_tp_sweep_tau_classic'
        db_astro = ExpStorage()
        db_astro.meta['descr'] = 'snn_1n1s1a_tp_sweep_tau_astro'

        # Same both tau together
        for tau in tqdm(all_tau, desc="tau"):
            # Sim with ip3 and k+ time constants = tau
            cfg['astro_plasticity']['tau_ip3'] = tau
            cfg['astro_plasticity']['tau_kp'] = tau
            
            db_stdp.prefix({'tau': tau, 'sweep': 'both'})
            db_astro.prefix({'tau': tau, 'sweep': 'both'})
            _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

            # Sim with ip3 time constant = tau
            cfg['astro_plasticity']['tau_ip3'] = tau
            cfg['astro_plasticity']['tau_kp'] = def_tau

            db_stdp.prefix({'tau': tau, 'sweep': 'ip3'})
            db_astro.prefix({'tau': tau, 'sweep': 'ip3'})
            _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

            # Sim with k+ time constant = tau
            cfg['astro_plasticity']['tau_ip3'] = def_tau
            cfg['astro_plasticity']['tau_kp'] = tau
            db_stdp.prefix({'tau': tau, 'sweep': 'k+'})
            db_astro.prefix({'tau': tau, 'sweep': 'k+'})
            _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

        dbs.append(('tau', db_astro, db_stdp))


        ## Simulate with different values for alpha_ip3 and alpha_k+
        _print_sim("Alpha ip3 and Alpha K+")
        cfg = config.Config(cfg_path)
        cfg['astro_params'] = cfg['astro_plasticity']
        def_dw_ltd = cfg['astro_params']['dw_mult']['dw_ltd']
        def_dw_ltp = cfg['astro_params']['dw_mult']['dw_ltp']
        all_dw = torch.linspace(0.1, 0.5, 5)

        db_stdp = ExpStorage()
        db_stdp.meta['descr'] = 'snn_1n1s1a_tp_sweep_dw_factor_classic'
        db_astro = ExpStorage()
        db_astro.meta['descr'] = 'snn_1n1s1a_tp_sweep_dw_factor_astro'
        

        # Same both tau together
        for dw in tqdm(all_dw, desc="dw"):
            # Sim with ltp/ltd factors changed together
            cfg['astro_plasticity']['dw_mult']['dw_ltd'] = 1.0 - dw
            cfg['astro_plasticity']['dw_mult']['dw_ltp'] = 1.0 + dw
            db_stdp.prefix({'dw': dw, 'sweep': 'ltp/ltd'})
            db_astro.prefix({'dw': dw, 'sweep': 'ltp/ltd'})
            _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

            # Sim with ltd = dw
            cfg['astro_params']['dw_mult']['dw_ltd'] = 1.0 - dw
            cfg['astro_params']['dw_mult']['dw_ltp'] = def_dw_ltp
            db_stdp.prefix({'dw': dw, 'sweep': 'ltd'})
            db_astro.prefix({'dw': dw, 'sweep': 'ltp/ltd'})
            _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

            # Sim with ltp = dw
            cfg['astro_params']['dw_mult']['dw_ltd'] = def_dw_ltd
            cfg['astro_params']['dw_mult']['dw_ltp'] = 1.0 + dw
            db_stdp.prefix({'dw': dw, 'sweep': 'ltp'})
            db_astro.prefix({'dw': dw, 'sweep': 'ltp/ltd'})
            _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

        dbs.append(('dw_factor', db_astro, db_stdp))


        ## Simulate with different values for lif neuron threshold
        cfg = config.Config(cfg_path)
        all_thr = torch.linspace(0.01, 0.4, 10)

        db_stdp = ExpStorage()
        db_stdp.meta['descr'] = 'snn_1n1s1a_tp_sweep_lif_v_th'
        db_astro = ExpStorage()
        db_astro.meta['descr'] = 'snn_1n1s1a_tp_sweep_lif_v_th'

        for thr in tqdm(all_thr, desc="v_th"):
            cfg['lif_params']['v_th'] = thr
            db_stdp.prefix({'v_th': thr})
            db_astro.prefix({'v_th': thr})
            _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

        dbs.append(('v_th', db_astro, db_stdp))


        ## Simulate, sweeping ca threshold
        cfg = config.Config(cfg_path)
        all_thr = torch.linspace(0.1, 3.0, 10)

        db_stdp = ExpStorage()
        db_stdp.meta['descr'] = 'snn_1n1s1a_tp_sweep_ca_th'
        db_astro = ExpStorage()
        db_astro.meta['descr'] = 'snn_1n1s1a_tp_sweep_ca_th'

        for thr in tqdm(all_thr, desc="ca_th"):
            cfg['astro_plasticity']['ca_th'] = thr
            db_stdp.prefix({'ca_th': thr})
            db_astro.prefix({'ca_th': thr})
            _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

        dbs.append(('ca_th', db_astro, db_stdp))
        

        for dbt in dbs:
            path = Path(db_name.format(dbt[0], 'astro'))
            dbt[1].save(path)

            path = Path(db_name.format(dbt[0], 'stdp'))
            dbt[2].save(path)

        return dbs


def _sim_stdp_and_astro_v2(
    cfg,
    spikes,
    db_stdp,
    db_astro
):

    dbs = []

    if not (db_stdp is None):
        cfg['astro_params'] = cfg['astro_plasticity']
        _config_stdp(cfg)
        sim_lif_astro_net(
            cfg,
            spikes,
            db_stdp
        )

        dbs.append(db_stdp)

    if not (db_astro is None):
        cfg['astro_params'] = cfg['astro_plasticity']
        _config_astro(cfg)
        sim_lif_astro_net(
            cfg,
            spikes,
            db_astro
        )

        dbs.append(db_astro)


    return dbs


def _sim_stdp_and_astro(
    cfg,
    spikes,
    descr,
    astro_only=False, stdp_only=False,
    graph_only_weight=False,
    astro_only_ca=True,
    ca_th=2.5,
):
    dbs = []

    if not astro_only:
        db = ExpStorage()
        db.meta['descr'] = "{}_classic".format(descr)
        cfg['astro_params'] = cfg['astro_plasticity']
        _config_stdp(cfg)
        sim_lif_astro_net(
            cfg,
            spikes,
            db
        )

        dbs.append(db)

    if not stdp_only:
        db = ExpStorage()
        db.meta['descr'] = "{}_astro".format(descr)
        cfg['astro_params'] = cfg['astro_plasticity']
        _config_astro(cfg, ca_th=ca_th)
        sim_lif_astro_net(
            cfg,
            spikes,
            db
        )

        dbs.append(db)


    for db in dbs:
        db.meta['graphs'] = ['spikes', 'weight']
        
        if astro_only_ca:
            db.meta['graphs'].append('astro-ca')
        else:
            db.meta['graphs'].append('astro')

        if graph_only_weight:
            db.meta['graphs'] = ['weight']

    return dbs


def _exp_rate_plasticity(cfg_path, sim=False):
    db_name = '_exp_rate_plasticity_{}.db'

    dbs = []
    if not sim:
        dbs = try_load_dbs(db_name, many=True)
        if len(dbs) > 0:
            return dbs

    with torch.no_grad():
        cfg = config.Config(cfg_path)

        cfg = _config_rp(cfg, 1.5)

        spikes = gen_rate_spikes([
            (0.3, cfg['sim']['steps'])
        ])
        
        db = ExpStorage()
        db.meta['descr'] = 'snn_1n1s1a_rp'
        sim_lif_astro_net(
            cfg,
            spikes,
            db
        )
        dbs.append(db)

        cfg = _config_rp(cfg, 0.8)
        db = ExpStorage()
        db.meta['descr'] = 'snn_1n1s1a_rp_thr'
        sim_lif_astro_net(
            cfg,
            spikes,
            db
        )
        dbs.append(db)


    for i, db in enumerate(dbs):
        db.save(db_name.format(i))

    return dbs


def _exp_average_pulse_pair_baseline(cfg_path, sim=False):
    """
    Simulate an snn in the 1n1s1a configuration, and demonstrate how plasticity
    triggered by a threshold on Ca can average multiple pulse pairs, and allow
    for confident steps for plasticity, based on multiple events.
    """

    db_name = '_exp_average_pulse_pair_baseline_{}.db'

    dbs = []
    if not sim:
        dbs = try_load_dbs(db_name, many=True)

        if len(dbs) > 0:
            return dbs
    

    with torch.no_grad():
        cfg = config.Config(cfg_path)

        # Classic pulsing input
        spikes = gen_ramp_impulse_spikes()
        dbs_sim = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse')
        dbs = dbs + dbs_sim

        # Repeat, but with a fixed length for each impulse
        spikes = gen_impulse_spikes(10, num_impulses=15)
        db_stdp, db_astro = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_const')
        db_astro.meta['xlim'] = (100, 150)
        db_stdp.meta['xlim'] = (100, 150)
        dbs = dbs + [db_astro, db_stdp]

        # Repeat, but with ltd shift
        spikes = gen_impulse_spikes(10, num_impulses=15)
        
        cfg['astro_plasticity']['stdp']['ltp'] = 0.5
        cfg['astro_plasticity']['stdp']['ltd'] = 1000.0

        db_stdp, db_astro = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_const_ltd-shift')
        db_astro.meta['xlim'] = (600, 800)
        db_stdp.meta['xlim'] = (600, 800)
        dbs = dbs + [db_astro, db_stdp]

        cfg['astro_plasticity']['stdp']['ltp'] = 1000.0
        cfg['astro_plasticity']['stdp']['ltd'] = 1000.0

        # Repeat with noisy input
        seed_many(123123098)
        spikes = gen_impulse_spikes(10, num_impulses=20, noise=0.02)        
        db_stdp, db_astro = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_const_noise', ca_th=3.75)
        db_astro.meta['xlim'] = (0, 500)
        db_stdp.meta['xlim'] = (750, 1250)
        dbs = dbs + [db_astro, db_stdp]
        
        dbs = dbs + dbs_sim

        # Repeat, but with a fixed length for each impulse, and poisson pattern
        seed_many(123123098)
        spikes = gen_impulse_spikes(20, num_impulses=15, poisson=True, rate=0.75)
        cfg['linear_params']['mu'] = 1.0
        db_stdp, db_astro = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_poisson_const')
        db_astro.meta['xlim'] = (100, 150)
        dbs = dbs + [db_astro, db_stdp]

        cfg['linear_params']['mu'] = 0.7

        # Sim just STDP, with longer spikes to show divergence
        spikes = gen_impulse_spikes(10, sim_len=20000)
        dbs_sim = _sim_stdp_and_astro(
            cfg, spikes,
            'snn_1n1s1a_tp_pulse_const_diverge',
            stdp_only=True, graph_only_weight=True)
        dbs_sim[0].meta['graph_weight_only'] = True
        dbs = dbs + dbs_sim

        # Again with LTD bias
        spikes = gen_ramp_impulse_spikes()
        cfg['astro_plasticity']['tau_kp'] = 30
        dbs_sim = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_ltd_bias')
        dbs = dbs + dbs_sim

        # Again with LTP bias
        cfg = config.Config(cfg_path)
        spikes = gen_ramp_impulse_spikes()
        cfg['astro_plasticity']['tau_ip3'] = 30
        dbs_sim = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_ltp_bias')
        dbs = dbs + dbs_sim

        # Trigger Plasticity at the end of a group of spikes
        # Weight updates are proportial to Ca
        if False:
            cfg = config.Config(cfg_path)
            spikes = gen_ramp_impulse_spikes()
            cfg['astro_plasticity']['weight_update'] = 'ip3_k+_fall'
            cfg['astro_plasticity']['tau_ca'] = 10.0
            dbs_sim = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_fall', astro_only=True)
            dbs = dbs + dbs_sim

            # Same as above, but with a fixed length impulse
            cfg = config.Config(cfg_path)
            spikes = gen_impulse_spikes(10, num_impulses=15)
            cfg['astro_plasticity']['weight_update'] = 'ip3_k+_fall'
            cfg['astro_plasticity']['tau_ca'] = 10.0
            dbs_sim = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_const_fall', astro_only=True)
            dbs = dbs + dbs_sim


    for i, db in enumerate(dbs):
        db.save(db_name.format(i))

    return dbs


def _graph_sweep_param(v, suffix, prefix_key):
    db_astro, db_stdp = v

    descr = db_astro.meta['descr']

    graphs=['weight']*2
    graphs.append('pre-spikes')

    fig, axes = gen_sgnn_axes(1, graphs=graphs, offset=False)
    
    axes['weight'][0].set_title("Astrocyte Plasticity Response to Various {}".format(suffix), fontsize=25)
    axes['weight'][1].set_title("Classic STDP Response to Spike Impulses for Various {}".format(suffix), fontsize=25)

    for i, d in enumerate(db_astro):
        prefix = d[prefix_key]
        graph_sgnn(d, fig, axes, 0, prefix=prefix, plot=graphs)

    for d in db_stdp:
        prefix = d[prefix_key]
        graph_sgnn(d, fig, axes, 1, prefix=prefix, plot=graphs)

    return ("{}_0.svg".format(descr), fig, axes)


def _graph_sweep_param_sp(db, db_by_key, prefix_key, title):
    """
    Generate a figure for a single db
    """
    num_subplots = len(db_by_key)
    graphs = ['weight']*num_subplots
    graphs.append('pre-spikes')
    
    fig, axes = gen_sgnn_axes(1, graphs=graphs, offset=False)
    descr = db.meta['descr']

    for i, (key, by_key) in enumerate(db_by_key.items()):
        # graph each timeline
        for d in by_key:
            prefix = d[prefix_key]
            graph_sgnn(d, fig, axes, i, prefix=prefix, plot=graphs)
            
        # for each plot, set the title
        axes['weight'][i].set_title(title)

    return ("{}_0.svg".format(descr), fig, axes)


def _graph_sweep_v_th(v):
    db_stdp, db_astro = v[0], v[1]

    descr = db_stdp.meta['descr']

    graphs = ['weight']*2
    graphs.append('spikes')

    fig, axes = None, None
    for d in db_stdp:
        tl = d['tl']
        prefix = d['v_th']

        if fig is None and axes is None:
            fig, axes = graph_lif_astro_compare(tl, 0, graphs=graphs, prefix=prefix)
            axes['weight'][0].set_title("Classic STDP Response to Sweeping LIF Neuron Threshold")
            axes['weight'][1].set_title("Astrocyte Response to Sweeping LIF Neuron Threshold")
        else:
            fig, axes = graph_lif_astro_compare(tl, 0, fig=fig, axes=axes, prefix=prefix)

    for d in db_astro:
        tl = d['tl']
        prefix = d['v_th']
        fig, axes = graph_lif_astro_compare(tl, 1, fig=fig, axes=axes, prefix=prefix)

    return ("{}_0.svg".format(descr), fig, axes)


def _graph_sweep_ca_th(v):
    db_stdp, db_astro = v[0], v[1]

    descr = db_stdp.meta['descr']

    graphs = ['weight']*2
    graphs.append('spikes')

    fig, axes = None, None
    for d in db_stdp:
        tl = d['tl']
        prefix = d['ca_th']

        if fig is None and axes is None:
            fig, axes = graph_lif_astro_compare(tl, 0, graphs=graphs, prefix=prefix)
            axes['weight'][0].set_title("Classic STDP Response to Sweeping Weight Alpha")
            axes['weight'][1].set_title("Astrocyte Response to Sweeping Ca Threshold")
        else:
            fig, axes = graph_lif_astro_compare(tl, 0, fig=fig, axes=axes, prefix=prefix)

    for d in db_astro:
        tl = d['tl']
        prefix = d['ca_th']
        fig, axes = graph_lif_astro_compare(tl, 1, fig=fig, axes=axes, prefix=prefix)

    return ("{}_0.svg".format(descr), fig, axes)


def _graph_average_pulse_pair(dbs):
    for db in dbs:
        assert len(db) == 1

        graphs = None
        if 'graphs' in db.meta:
            graphs = db.meta['graphs']

        fig, axes = gen_sgnn_axes(1, graphs=graphs, figsize=(8,10))

        db_rec = db[0]
        graph_sgnn(db_rec, fig, axes, 0, plot=graphs)

        if 'xlim' in db.meta:
            print("xlim: ", db.meta['xlim'])
            for k, v in axes.items():
                for ax in v:
                    ax.set_xlim(db.meta['xlim'])


        fig_path = "{}_0.svg".format(db.meta['descr'])
        print("Saving: ", fig_path)
        fig.tight_layout()
        fig.savefig(fig_path)


def _graph_average_pulse_pair_sweep(sim_results):
    figures = []

    for entry in sim_results:
        k = entry[0]
        v = entry[1:]

        print("graphing: ", k)

        # If Sweeping mu
        if k in ['mu', 'assoc', 'v_th', 'ca_th']:
            suffix = k
            prefix_key = k

            if k == 'assoc':
                suffix = "Spike Association"

            figures.append(_graph_sweep_param(v, suffix, prefix_key))

        elif k in ['tau', 'dw_factor']:
            db_astro, db_stdp = v[0:2]

            if k == 'dw_factor':
                k = 'dw'
            
            figures.append(_graph_sweep_param_sp(db_stdp, db_stdp.group_by('sweep'), k, "Classic STDP"))
            figures.append(_graph_sweep_param_sp(db_astro, db_astro.group_by('sweep'), k, "Astrocyte"))


    for fig_path, fig, axes in figures:
        print("saving: ", fig_path)

        fig.savefig(fig_path)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--astro-impulse', action='store_true')
    parser.add_argument('--astro-impulse-sweep', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--sim', action='store_true')

    return parser.parse_args()


def _main(args):
    os.makedirs('./obj2', exist_ok=True)
    os.chdir('./obj2')

    cfg_path = '../../config/1n1s1a_obj2.yaml'

    plot.rc_config({'font.size': 14})

    if args.astro_impulse or args.all:
        dbs = _exp_rate_plasticity(cfg_path, sim=args.sim)
        _graph_average_pulse_pair(dbs)

        dbs = _exp_average_pulse_pair_baseline(cfg_path, sim=args.sim)
        _graph_average_pulse_pair(dbs)

    if args.astro_impulse_sweep or args.all:
        dbs = _exp_average_pulse_pair_sweep(cfg_path, sim=args.sim)
        _graph_average_pulse_pair_sweep(dbs)


if __name__ == '__main__':
    args = _parse_args()

    _main(args)
