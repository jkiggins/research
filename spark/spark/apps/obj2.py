import argparse
import os
from tqdm import tqdm
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
    graph_lif_astro_net,
    graph_lif_astro_compare,
)
from .astro_spike_pair import sim_astro_probe, graph_dw_dt, graph_astro_tls
from ..experiment import ExpStorage, VSweep, seed_many, load_or_run

def _sim_stdp_and_astro(cfg, spikes, db_astro, db_stdp, descr):
    db_stdp = _sim_stdp(cfg, spikes, descr)
    db_astro = _sim_astro(cfg, spikes, descr)

    return [db_stdp, db_astro]


def _exp_average_pulse_pair_sweep(cfg_path):
    """
    Simulate an snn in the 1n1s1a configuration, and demonstrate how plasticity
    triggered by a threshold on u can average multiple pulse pairs, and allow
    for confident steps for plasticity, based on multiple events.

    See how sweeping some different parameters effect things
    * mu for synaptic weight initialization
    """

    dbs = []

    with torch.no_grad():
        cfg = config.Config(cfg_path)

        ## Simulate for various values of mu
        spikes = gen_ramp_impulse_spikes()
        all_mu = torch.linspace(0.2, 1.0, 7)

        # Create db objects
        db_stdp = ExpStorage()
        db_stdp.meta['descr'] = 'snn_1n1a1s_tp_mu_sweep'
        db_astro = ExpStorage()
        db_astro.meta['descr'] = 'snn_1n1a1s_tp_mu_sweep'
        
        for mu in all_mu:
            cfg['linear_params']['mu'] = mu

            cfg['astro_params'] = cfg['astro_plasticity']
            db_astro.prefix({'mu': mu})
            sim_lif_astro_net(cfg, spikes, db_astro)

            cfg['astro_params'] = cfg['classic_stdp']
            db_stdp.prefix({'mu': mu})
            sim_lif_astro_net(cfg, spikes, db_stdp)

        dbs.append(('mu', db_astro, db_stdp))

        ## Simulate different spike associations with just STDP
        cfg = config.Config(cfg_path)
        cfg['astro_params'] = cfg['classic_stdp']
        
        db_stdp = ExpStorage()
        db_stdp.meta['descr'] = 'snn_1n1a1s_tp_sweep_association'
        cfg['astro_params'] = cfg['classic_stdp']

        # Nearest Neighbor
        cfg['astro_params']['pre_reset_on_spike'] = True
        cfg['astro_params']['post_reset_on_spike'] = True
        db_stdp.prefix({'assoc': 'nn'})
        sim_lif_astro_net(cfg, spikes, db_stdp)

        # Nearest Pre
        cfg['astro_params']['pre_reset_on_spike'] = True
        cfg['astro_params']['post_reset_on_spike'] = False
        db_stdp.prefix({'assoc': 'nn-pre'})
        sim_lif_astro_net(cfg, spikes, db_stdp)

        # Nearest Post
        cfg['astro_params']['pre_reset_on_spike'] = False
        cfg['astro_params']['post_reset_on_spike'] = True
        db_stdp.prefix({'assoc': 'nn-post'})
        sim_lif_astro_net(cfg, spikes, db_stdp)

        # Any
        cfg['astro_params']['pre_reset_on_spike'] = False
        cfg['astro_params']['post_reset_on_spike'] = False
        db_stdp.prefix({'assoc': 'many-many'})
        sim_lif_astro_net(cfg, spikes, db_stdp)

        dbs.append(('assoc', db_stdp))


        ## Simulate with different values for tau_ip3 and tau_k+
        cfg = config.Config(cfg_path)
        cfg['astro_params'] = cfg['classic_stdp']
        def_tau = cfg['astro_params']['tau_i_pre']
        all_tau = torch.linspace(10, 800, 10)
        
        db_stdp = ExpStorage()
        db_stdp.meta['descr'] = 'snn_1n1a1s_tp_sweep_tau'
        
        # Same both tau together
        for tau in all_tau:
            # Sim with ip3 and k+ time constants = tau
            cfg['astro_params']['tau_i_pre'] = tau
            cfg['astro_params']['tau_i_post'] = tau
            db_stdp.prefix({'tau': tau, 'sweep': 'both'})
            sim_lif_astro_net(cfg, spikes, db_stdp)

            # Sim with ip3 time constant = tau
            cfg['astro_params']['tau_i_pre'] = tau
            cfg['astro_params']['tau_i_post'] = def_tau
            db_stdp.prefix({'tau': tau, 'sweep': 'ip3'})
            sim_lif_astro_net(cfg, spikes, db_stdp)

            # Sim with k+ time constant = tau
            cfg['astro_params']['tau_i_pre'] = def_tau
            cfg['astro_params']['tau_i_post'] = tau
            db_stdp.prefix({'tau': tau, 'sweep': 'k+'})
            sim_lif_astro_net(cfg, spikes, db_stdp)

        dbs.append(('tau', db_stdp))

        return dbs


def _sim_stdp_and_astro(
    cfg,
    spikes,
    descr,
    astro_only=False, stdp_only=False,
    graph_only_weight=False
):
    dbs = []
    
    if not astro_only:
        db = ExpStorage()
        db.meta['descr'] = "{}_classic".format(descr)
        cfg['astro_params'] = cfg['classic_stdp']
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
        sim_lif_astro_net(
            cfg,
            spikes,
            db
        )

        dbs.append(db)

    if graph_only_weight:
        for db in dbs:
            db.meta['graphs'] = ['weight']

    return dbs


def _exp_average_pulse_pair_baseline(cfg_path):
    """
    Simulate an snn in the 1n1s1a configuration, and demonstrate how plasticity
    triggered by a threshold on u can average multiple pulse pairs, and allow
    for confident steps for plasticity, based on multiple events.
    """

    dbs = []

    with torch.no_grad():
        cfg = config.Config(cfg_path)

        # Classic pulsing input
        spikes = gen_ramp_impulse_spikes()
        dbs_sim = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse')
        dbs = dbs + dbs_sim

        # Repeat with noisy input
        spikes = gen_ramp_impulse_spikes(noise=0.02)
        dbs_sim = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_noise')
        dbs = dbs + dbs_sim

        # Repeat, but with a fixed length for each impulse
        spikes = gen_impulse_spikes(10, num_impulses=15)
        dbs_sim = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_const')
        dbs = dbs + dbs_sim

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
        cfg['classic_stdp']['tau_i_post'] = 30
        cfg['astro_plasticity']['tau_i_post'] = 30
        dbs_sim = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_ltd_bias')
        dbs = dbs + dbs_sim

        # Again with LTP bias
        cfg = config.Config(cfg_path)
        spikes = gen_ramp_impulse_spikes()
        cfg['classic_stdp']['tau_i_pre'] = 30
        cfg['astro_plasticity']['tau_i_pre'] = 30
        dbs_sim = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_ltp_bias')
        dbs = dbs + dbs_sim

        # Trigger Plasticity at the end of a group of spikes
        # Weight updates are proportial to Ca
        cfg = config.Config(cfg_path)
        spikes = gen_ramp_impulse_spikes()
        cfg['astro_plasticity']['weight_update'] = 'ip3_k+_fall'
        cfg['astro_plasticity']['tau_u'] = 10.0
        dbs_sim = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_fall', astro_only=True)
        dbs = dbs + dbs_sim

        # Same as above, but with a fixed length impulse
        cfg = config.Config(cfg_path)
        spikes = gen_impulse_spikes(10, num_impulses=15)
        cfg['astro_plasticity']['weight_update'] = 'ip3_k+_fall'
        cfg['astro_plasticity']['tau_u'] = 10.0
        dbs_sim = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_const_fall', astro_only=True)
        dbs = dbs + dbs_sim


    return dbs


def _graph_average_pulse_pair_sweep(dbs):
    # Each entry contains a pair of db's each representing a sweep of some parameter for astrocyte and classic STDP cases

    for entry in dbs:
        k = entry[0]
        v = entry[1:]
    
        # One graph per set of db items
        fig, axes = None, None

        # If Sweeping mu
        if k == 'mu':
            db_astro, db_stdp = v

            descr = db_astro.meta['descr']

            for i, d in enumerate(db_astro):
                tl = d['tl']
                prefix = str(float(d['mu']))
                if i == 0:
                    fig, axes = graph_lif_astro_compare(tl, 0, graphs=['weight']*2, prefix=prefix)
                    axes['weight'][0].set_title("Astrocyte Plasticity Response to Various mu")
                    axes['weight'][1].set_title("Classic STDP Response to Spike Impulses for Various mu")
                else:
                    fig, axes = graph_lif_astro_compare(tl, 0, fig=fig, axes=axes, prefix=prefix)

            for d in db_stdp:
                prefix = str(float(d['mu']))
                tl = d['tl']
                fig, axes = graph_lif_astro_compare(tl, 1, fig=fig, axes=axes, prefix=prefix)
                print("stdp mu: ", d['mu'])

        # If sweeping different spike associations
        elif k == 'assoc':
            db_stdp = v[0]
            descr = db_stdp.meta['descr']

            for i, d in enumerate(db_stdp):
                prefix = d['assoc']
                tl = d['tl']

                if i == 0:
                    fig, axes = graph_lif_astro_compare(tl, 0, graphs=['weight'], prefix=prefix)
                    axes['weight'][0].set_title("Classic STDP Response to Various Spike Associations")
                else:
                    fig, axes = graph_lif_astro_compare(tl, 0, fig=fig, axes=axes, prefix=prefix)

        elif k == 'tau':
            db_stdp = v[0]
            descr = db_stdp.meta['descr']

            # Generate one weight subplot plot per sweep type, which is one of [ip3, k+, both]
            db_by_sweep = db_stdp.group_by('sweep')
            num_subplots = len(db_by_sweep)

            for i, (sweep, by_sweep) in enumerate(db_by_sweep.items()):
                # graph each timeline
                for d in by_sweep:
                    prefix = d['tau']
                    tl = d['tl']

                    if fig is None or axes is None:
                        fig, axes = graph_lif_astro_compare(tl, i, graphs=['weight']*num_subplots, prefix=prefix)
                    else:
                        fig, axes = graph_lif_astro_compare(tl, i, fig=fig, axes=axes, prefix=prefix)

                # for each plot, set the title
                axes['weight'][i].set_title("Classic STDP Response to Sweeping {} Time Constant (tau)".format(sweep))
                

        # draw legend for all axes
        for k, axs in axes.items():
            for ax in axs:
                ax.legend()

        print("{}: {}".format(descr, axes))
        fig.savefig("{}_0.svg".format(descr))


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

    if args.astro_impulse or args.all:
        dbs = _exp_average_pulse_pair_baseline('../../config/1n1s1a_obj2.yaml')
        for db in dbs:
            if 'graphs' in db.meta:
                fig = graph_lif_astro_net(db, graphs=db.meta['graphs'])
            else:
                fig = graph_lif_astro_net(db)
            fig.savefig("{}_0.svg".format(db.meta['descr']))

    if args.astro_impulse_sweep or args.all:
        dbs = _exp_average_pulse_pair_sweep('../../config/1n1s1a_obj2.yaml')
        _graph_average_pulse_pair_sweep(dbs)
            

if __name__ == '__main__':
    args = _parse_args()

    _main(args)
