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
)
from .astro_spike_pair import sim_astro_probe, graph_dw_dt, graph_astro_tls
from ..experiment import ExpStorage, VSweep, seed_many, load_or_run

def _sim_astro(cfg, spikes, descr):
    # Sim w/ Averaged STDP
    cfg['astro_params'] = cfg['astro_plasticity']
    db_astro = sim_lif_astro_net(
        cfg,
        spikes,
    )

    descr = "{}_astro_plasticity".format(descr)
    db_astro.meta['descr'] = descr
    
    return db_astro


def _sim_stdp(cfg, spikes, descr):
    # Sim w/ Averaged STDP
    cfg['astro_params'] = cfg['classic_stdp']
    db_classic = sim_lif_astro_net(
        cfg,
        spikes
    )

    descr = "{}_classic".format(descr)
    db_classic.meta['descr'] = descr

    return db_classic


def _sim_stdp_and_astro(cfg, spikes, descr):
    db_classic = _sim_stdp(cfg, spikes, descr)
    db_astro = _sim_astro(cfg, spikes, descr)

    return [db_classic, db_astro]


def _exp_average_pulse_pair_sweep(cfg_path):
    """
    Simulate an snn in the 1n1s1a configuration, and demonstrate how plasticity
    triggered by a threshold on u can average multiple pulse pairs, and allow
    for confident steps for plasticity, based on multiple events.
    """

    dbs = []

    with torch.no_grad():
        cfg = config.Config(cfg_path)

        # Show weight updates vs. Inital weight \mu
        spikes = gen_ramp_impulse_spikes()
        all_mu = torch.linspace(0.5, 1.0, 5)
        dbs = []

        for mu in all_mu:
            cfg['linear_params']['mu'] = mu
            dbs_sim = _sim_stdp_and_astro(cfg, spikes, 'snn_1n1s1a_tp_sweep_mu')

            for db in dbs_sim:
                db.meta['prefix'] = 'mu={:2.2f}'.format(float(mu))
            
            dbs.append(dbs_sim)

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
        dbs_sim = [_sim_stdp(cfg, spikes, 'snn_1n1s1a_tp_pulse_const_diverge')]
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
        dbs_sim = [_sim_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_fall')]
        dbs = dbs + dbs_sim

        # Same as above, but with a fixed length impulse
        cfg = config.Config(cfg_path)
        spikes = gen_impulse_spikes(10, num_impulses=15)
        cfg['astro_plasticity']['weight_update'] = 'ip3_k+_fall'
        cfg['astro_plasticity']['tau_u'] = 10.0
        dbs_sim = [_sim_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_const_fall')]
        dbs = dbs + dbs_sim


    return dbs


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
            fig = graph_lif_astro_net(db, db.meta['descr'])

    if args.astro_impulse_sweep or args.all:
        dbs = _exp_average_pulse_pair_sweep('../../config/1n1s1a_obj2.yaml')

        # Unzip into astro and classic dbs
        astro_dbs, classic_dbs = zip(*dbs)

        fig = graph_lif_astro_net(dbs, 'lif_astro_net_sweep_mu')
            


if __name__ == '__main__':
    args = _parse_args()

    _main(args)
