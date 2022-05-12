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
    gen_group_spikes,
    sim_lif_astro_net,
    sim_lif_astro_reward_net,
    graph_lif_astro_net,
    graph_lif_astro_reward_net,
)
from .astro_spike_pair import sim_astro_probe, graph_dw_dt, graph_astro_tls
from ..experiment import ExpStorage, VSweep, seed_many, load_or_run

def _sim_pulse_pair_classic_and_astro(cfg, spikes, prefix):
    # Sim w/ classic STDP
    cfg['astro_params'] = cfg['classic_stdp']
    db_classic = sim_lif_astro_net(
        cfg,
        spikes,
        name="{}_classic".format(prefix)
    )

    # Sim w/ Averaged STDP
    cfg['astro_params'] = cfg['astro_plasticity']
    db_astro = sim_lif_astro_net(
        cfg,
        spikes,
        name="{}_astro_plasticity".format(prefix)
    )

    return [db_classic, db_astro]
    
    
def _exp_average_pulse_pair(cfg_path):
    """
    Simulate an snn in the 1n1s1a configuration, and demonstrate how plasticity
    triggered by a threshold on u can average multiple pulse pairs, and allow
    for confident steps for plasticity, based on multiple events.
    """

    dbs = []
    
    with torch.no_grad():
        cfg = config.Config(cfg_path)

        # Classisc pulsing input
        spikes = gen_group_spikes()
        dbs_sim = _sim_pulse_pair_classic_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse')
        dbs = dbs + dbs_sim

        # Repeat with noisy input
        spikes = gen_group_spikes(noise=0.02)
        dbs_sim = _sim_pulse_pair_classic_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_noise')
        dbs = dbs + dbs_sim

        # Again with LTD bias
        spikes = gen_group_spikes()
        cfg['classic_stdp']['tau_i_post'] = 30
        cfg['astro_plasticity']['tau_i_post'] = 30
        dbs_sim = _sim_pulse_pair_classic_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_ltd_bias')
        dbs = dbs + dbs_sim

        # Again with LTP bias
        cfg = config.Config(cfg_path)
        spikes = gen_group_spikes()
        cfg['classic_stdp']['tau_i_pre'] = 30
        cfg['astro_plasticity']['tau_i_pre'] = 30
        dbs_sim = _sim_pulse_pair_classic_and_astro(cfg, spikes, 'snn_1n1s1a_tp_pulse_ltp_bias')
        dbs = dbs + dbs_sim


    return dbs


def _exp_reward_plasticity(cfg_path):
    dbs = []

    cfg = config.Config(cfg_path)

    spikes = gen_rate_spikes([(0.1, 1000)])

    with torch.no_grad():
        cfg['astro_params'] = cfg['classic_stdp']
        db1 = sim_lif_astro_reward_net(cfg, spikes, name='snn_1n1s1a_reward')

        cfg['astro_params'] = cfg['astro_plasticity']
        db2 = sim_lif_astro_reward_net(cfg, spikes, name='snn_1n1s1a_reward_plastic')


    return [db1, db2]


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--astro-impulse', action='store_true')
    parser.add_argument('--astro-reward', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--sim', action='store_true')

    return parser.parse_args()


def _main(args):
    os.makedirs('./obj2', exist_ok=True)
    os.chdir('./obj2')

    if args.astro_impulse or args.all:
        dbs = _exp_average_pulse_pair('../../config/1n1s1a_obj2.yaml')
        for db in dbs:
            graph_lif_astro_net(db)

    if args.astro_reward or args.all:
        dbs = _exp_reward_plasticity('../../config/1n1s1a_obj2.yaml')
        for db in dbs:
            graph_lif_astro_reward_net(db)


if __name__ == '__main__':
    args = _parse_args()

    _main(args)
