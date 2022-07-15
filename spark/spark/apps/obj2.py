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

def _print_sim(name):
    print("##### Simulation: {} #####".format(name))


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
        _print_sim("Initial Average Weight")
        # spikes = gen_ramp_impulse_spikes()
        spikes = gen_impulse_spikes(10, num_impulses=15)
        all_mu = torch.linspace(0.2, 1.0, 7)

        # Create db objects
        db_stdp = ExpStorage()
        db_stdp.meta['descr'] = 'snn_1n1a1s_tp_mu_sweep'
        db_astro = ExpStorage()
        db_astro.meta['descr'] = 'snn_1n1a1s_tp_mu_sweep'

        for mu in tqdm(all_mu, desc="mu: "):
            cfg['linear_params']['mu'] = mu

            cfg['astro_params'] = cfg['astro_plasticity']
            db_astro.prefix({'mu': mu})
            sim_lif_astro_net(cfg, spikes, db_astro)

            cfg['astro_params'] = cfg['classic_stdp']
            db_stdp.prefix({'mu': mu})
            sim_lif_astro_net(cfg, spikes, db_stdp)

        dbs.append(('mu', db_astro, db_stdp))

        ## Simulate different spike associations with just STDP
        _print_sim("Spike Associations")
        cfg = config.Config(cfg_path)

        db_stdp = ExpStorage()
        db_stdp.meta['descr'] = 'snn_1n1a1s_tp_sweep_association'
        db_astro = ExpStorage()
        db_astro.meta['descr'] = 'snn_1n1a1s_tp_sweep_association'
        cfg['astro_params'] = cfg['classic_stdp']

        # Nearest Neighbor
        cfg['astro_plasticity']['pre_reset_on_spike'] = True
        cfg['astro_plasticity']['post_reset_on_spike'] = True
        cfg['classic_stdp']['pre_reset_on_spike'] = True
        cfg['classic_stdp']['post_reset_on_spike'] = True
        db_stdp.prefix({'assoc': 'nn'})
        db_astro.prefix({'assoc': 'nn'})
        _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

        # Nearest Pre
        cfg['astro_plasticity']['pre_reset_on_spike'] = True
        cfg['astro_plasticity']['post_reset_on_spike'] = False
        cfg['classic_stdp']['pre_reset_on_spike'] = True
        cfg['classic_stdp']['post_reset_on_spike'] = False
        db_stdp.prefix({'assoc': 'nn-pre'})
        db_astro.prefix({'assoc': 'nn-pre'})
        _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

        # Nearest Post
        cfg['astro_plasticity']['pre_reset_on_spike'] = False
        cfg['astro_plasticity']['post_reset_on_spike'] = True
        cfg['classic_stdp']['pre_reset_on_spike'] = False
        cfg['classic_stdp']['post_reset_on_spike'] = True

        db_stdp.prefix({'assoc': 'nn-post'})
        db_astro.prefix({'assoc': 'nn-post'})
        _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

        # Any
        cfg['astro_plasticity']['pre_reset_on_spike'] = False
        cfg['astro_plasticity']['post_reset_on_spike'] = False
        cfg['classic_stdp']['pre_reset_on_spike'] = False
        cfg['classic_stdp']['post_reset_on_spike'] = False
        db_stdp.prefix({'assoc': 'many-many'})
        db_astro.prefix({'assoc': 'many-many'})
        _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

        dbs.append(('assoc', db_stdp, db_astro))


        ## Simulate with different values for tau_ip3 and tau_k+
        _print_sim("Tau ip3 and Tau K+")
        cfg = config.Config(cfg_path)
        cfg['astro_params'] = cfg['classic_stdp']
        def_tau = cfg['astro_params']['tau_i_pre']
        all_tau = torch.linspace(10, 800, 10)

        # Different descr only needed when each db will result in a separate graph
        db_stdp = ExpStorage()
        db_stdp.meta['descr'] = 'snn_1n1a1s_tp_sweep_tau_classic'
        db_astro = ExpStorage()
        db_astro.meta['descr'] = 'snn_1n1a1s_tp_sweep_tau_astro'

        # Same both tau together
        for tau in tqdm(all_tau, desc="tau"):
            # Sim with ip3 and k+ time constants = tau
            cfg['astro_plasticity']['tau_i_pre'] = tau
            cfg['astro_plasticity']['tau_i_post'] = tau
            cfg['classic_stdp']['tau_i_pre'] = tau
            cfg['classic_stdp']['tau_i_post'] = tau
            
            db_stdp.prefix({'tau': tau, 'sweep': 'both'})
            db_astro.prefix({'tau': tau, 'sweep': 'both'})
            _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

            # Sim with ip3 time constant = tau
            cfg['astro_plasticity']['tau_i_pre'] = tau
            cfg['astro_plasticity']['tau_i_post'] = def_tau
            cfg['classic_stdp']['tau_i_pre'] = tau
            cfg['classic_stdp']['tau_i_post'] = def_tau

            db_stdp.prefix({'tau': tau, 'sweep': 'ip3'})
            db_astro.prefix({'tau': tau, 'sweep': 'ip3'})
            _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

            # Sim with k+ time constant = tau
            cfg['astro_plasticity']['tau_i_pre'] = def_tau
            cfg['astro_plasticity']['tau_i_post'] = tau
            cfg['classic_stdp']['tau_i_pre'] = def_tau
            cfg['classic_stdp']['tau_i_post'] = tau
            db_stdp.prefix({'tau': tau, 'sweep': 'k+'})
            db_astro.prefix({'tau': tau, 'sweep': 'k+'})
            _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

        dbs.append(('tau', db_stdp, db_astro))


        ## Simulate with different values for alpha_ip3 and alpha_k+
        _print_sim("Alpha ip3 and Alpha K+")
        cfg = config.Config(cfg_path)
        cfg['astro_params'] = cfg['classic_stdp']
        def_dw_ltd = cfg['astro_params']['u_step_params']['dw_ltd']
        def_dw_ltp = cfg['astro_params']['u_step_params']['dw_ltp']
        all_dw = torch.linspace(0.1, 0.5, 5)

        db_stdp = ExpStorage()
        db_stdp.meta['descr'] = 'snn_1n1a1s_tp_sweep_dw_factor_classic'
        db_astro = ExpStorage()
        db_astro.meta['descr'] = 'snn_1n1a1s_tp_sweep_dw_factor_astro'
        

        # Same both tau together
        for dw in tqdm(all_dw, desc="dw"):
            # Sim with ltp/ltd factors changed together
            cfg['classic_stdp']['u_step_params']['dw_ltd'] = 1.0 - dw
            cfg['classic_stdp']['u_step_params']['dw_ltp'] = 1.0 + dw
            cfg['astro_plasticity']['u_step_params']['dw_ltd'] = 1.0 - dw
            cfg['astro_plasticity']['u_step_params']['dw_ltp'] = 1.0 + dw
            db_stdp.prefix({'dw': dw, 'sweep': 'ltp/ltd'})
            db_astro.prefix({'dw': dw, 'sweep': 'ltp/ltd'})
            _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

            # Sim with ltd = dw
            cfg['astro_params']['u_step_params']['dw_ltd'] = 1.0 - dw
            cfg['astro_params']['u_step_params']['dw_ltp'] = def_dw_ltp
            db_stdp.prefix({'dw': dw, 'sweep': 'ltd'})
            db_astro.prefix({'dw': dw, 'sweep': 'ltp/ltd'})
            _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

            # Sim with ltp = dw
            cfg['astro_params']['u_step_params']['dw_ltd'] = def_dw_ltd
            cfg['astro_params']['u_step_params']['dw_ltp'] = 1.0 + dw
            db_stdp.prefix({'dw': dw, 'sweep': 'ltp'})
            db_astro.prefix({'dw': dw, 'sweep': 'ltp/ltd'})
            _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

        dbs.append(('dw_factor', db_stdp, db_astro))


        ## Simulate with different values for lif neuron threshold
        cfg = config.Config(cfg_path)
        all_thr = torch.linspace(0.01, 0.4, 10)

        db_stdp = ExpStorage()
        db_stdp.meta['descr'] = 'snn_1n1a1s_tp_sweep_lif_v_th'
        db_astro = ExpStorage()
        db_astro.meta['descr'] = 'snn_1n1a1s_tp_sweep_lif_v_th'

        for thr in tqdm(all_thr, desc="v_th"):
            cfg['lif_params']['v_th'] = thr
            db_stdp.prefix({'v_th': thr})
            db_astro.prefix({'v_th': thr})
            _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

        dbs.append(('v_th', db_stdp, db_astro))


        ## Simulate, sweeping ca threshold
        cfg = config.Config(cfg_path)
        all_thr = torch.linspace(0.1, 3.0, 10)

        db_stdp = ExpStorage()
        db_stdp.meta['descr'] = 'snn_1n1a1s_tp_sweep_ca_th'
        db_astro = ExpStorage()
        db_astro.meta['descr'] = 'snn_1n1a1s_tp_sweep_ca_th'

        for thr in tqdm(all_thr, desc="ca_th"):
            cfg['classic_stdp']['u_th'] = thr
            cfg['astro_plasticity']['u_th'] = thr
            db_stdp.prefix({'ca_th': thr})
            db_astro.prefix({'ca_th': thr})
            _sim_stdp_and_astro_v2(cfg, spikes, db_stdp, db_astro)

        dbs.append(('ca_th', db_stdp, db_astro))
        

        return dbs


def _sim_stdp_and_astro_v2(
    cfg,
    spikes,
    db_stdp,
    db_astro
):

    dbs = []

    if not (db_stdp is None):
        cfg['astro_params'] = cfg['classic_stdp']
        sim_lif_astro_net(
            cfg,
            spikes,
            db_stdp
        )

        dbs.append(db_stdp)

    if not (db_astro is None):
        cfg['astro_params'] = cfg['astro_plasticity']
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
        import code
        code.interact(local=dict(globals(), **locals()))
        exit(1)
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


def _graph_sweep_mu(v):
    db_astro, db_stdp = v

    descr = db_astro.meta['descr']

    graphs=['weight']*2
    graphs.append('spikes')

    for i, d in enumerate(db_astro):
        tl = d['tl']
        prefix = str(float(d['mu']))
        if i == 0:
            fig, axes = graph_lif_astro_compare(tl, 0, graphs=graphs, prefix=prefix)
            axes['weight'][0].set_title("Astrocyte Plasticity Response to Various mu")
            axes['weight'][1].set_title("Classic STDP Response to Spike Impulses for Various mu")
        else:
            fig, axes = graph_lif_astro_compare(tl, 0, fig=fig, axes=axes, prefix=prefix)

    for d in db_stdp:
        prefix = str(float(d['mu']))
        tl = d['tl']
        fig, axes = graph_lif_astro_compare(tl, 1, fig=fig, axes=axes, prefix=prefix)

    return ("{}_0.svg".format(descr), fig, axes)


def _graph_sweep_assoc(v):
    db_stdp, db_astro = v[0], v[1]
    descr = db_stdp.meta['descr']

    graphs = ['weight']*2
    graphs.append('spikes')

    # Graph stdp traces on subplot 0
    for i, d in enumerate(db_stdp):
        prefix = d['assoc']
        tl = d['tl']

        if i == 0:
            fig, axes = graph_lif_astro_compare(tl, 0, graphs=graphs, prefix=prefix)
            axes['weight'][0].set_title("Classic STDP Response to Various Spike Associations")
            axes['weight'][1].set_title("Astrocyte Response to Various Spike Associations")
        else:
            prefix = d['assoc']
            fig, axes = graph_lif_astro_compare(tl, 0, fig=fig, axes=axes, prefix=prefix)

    # Graph astrocyte traces on subplot 1
    for d in db_astro:
        prefix = d['assoc']
        fig, axes = graph_lif_astro_compare(tl, 1, fig=fig, axes=axes, prefix=prefix)

    return ("{}_0.svg".format(descr), fig, axes)


def _graph_sweep_tau(db, title):
    """
    Generate a figure for a single db
    """

    fig, axes = None, None
    descr = db.meta['descr']

    # Generate one weight subplot plot per sweep type, which is one of [ip3, k+, both]
    db_by_sweep = db.group_by('sweep')
    num_subplots = len(db_by_sweep)

    graphs = ['weight']*num_subplots
    graphs.append('spikes')

    for i, (sweep, by_sweep) in enumerate(db_by_sweep.items()):
        # graph each timeline
        for d in by_sweep:
            prefix = d['tau']
            tl = d['tl']

            if fig is None or axes is None:
                fig, axes = graph_lif_astro_compare(tl, i, graphs=graphs, prefix=prefix)
            else:
                fig, axes = graph_lif_astro_compare(tl, i, fig=fig, axes=axes, prefix=prefix)

        # for each plot, set the title
        axes['weight'][i].set_title("{} Response to Sweeping {} Time Constant (tau)".format(title, sweep))

    return ("{}_0.svg".format(descr), fig, axes)


def _graph_sweep_dw_factor(db, title):
    descr = db.meta['descr']
    fig, axes = None, None

    # Generate one weight subplot plot per sweep type, which is one of [ip3, k+, both]
    db_by_sweep = db.group_by('sweep')
    num_subplots = len(db_by_sweep)

    graphs = ['weight']*num_subplots
    graphs.append('spikes')

    for i, (sweep, by_sweep) in enumerate(db_by_sweep.items()):
        # graph each timeline
        for d in by_sweep:
            prefix = d['dw']
            tl = d['tl']

            if fig is None or axes is None:
                fig, axes = graph_lif_astro_compare(tl, i, graphs=graphs, prefix=prefix)
            else:
                fig, axes = graph_lif_astro_compare(tl, i, fig=fig, axes=axes, prefix=prefix)

        # for each plot, set the title
        axes['weight'][i].set_title("{} Response to Sweeping dw Factor for {}".format(title, sweep))

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


def _graph_average_pulse_pair_sweep(sim_results):
    figures = []

    for entry in sim_results:
        k = entry[0]
        v = entry[1:]

        print("graphing: ", k)

        # If Sweeping mu
        if k == 'mu':
            figures.append(_graph_sweep_mu(v))

        # If sweeping different spike associations
        elif k == 'assoc':
            figures.append(_graph_sweep_assoc(v))

        # If sweeping tau
        elif k == 'tau':
            db_stdp, db_astro = v[0], v[1]
            figures.append(_graph_sweep_tau(db_stdp, 'Classic STDP'))
            figures.append(_graph_sweep_tau(db_astro, 'Astrocyte'))

        elif k == 'dw_factor':
            db_stdp, db_astro = v[0], v[1]
            figures.append(_graph_sweep_dw_factor(db_stdp, 'Classic STDP'))
            figures.append(_graph_sweep_dw_factor(db_astro, 'Astrocyte'))

        elif k == 'v_th':
            figures.append(_graph_sweep_v_th(v))

        elif k == 'ca_th':
            figures.append(_graph_sweep_ca_th(v))


    for fig_path, fig, axes in figures:
        print("saving: ", fig_path)
        # draw legend for all axes
        for k, axs in axes.items():
            for ax in axs:
                ax.legend()

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
