import argparse
import os
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
import yaml
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
    graph_sgnn,
    gen_and_spikes,
    sim_astro,
    astro_and_region,
    astro_check_respose,
    get_lif_astro_net_err
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


def _n_syn_coupled_astro_stats(db_syn, n_synapse):
    db_valid = db_syn.filter(invalid=False)
    db_invalid = db_syn.filter(invalid=True)
    
    stats = {
        'total': len(db_syn),
        'valid': len(db_valid),
        'invalid': {},
        'invalid-pairs': {},
    }

    db_rec_last = None
    for db_rec in db_invalid:
        region = db_rec['region'][0]
        
        if db_rec_last:
            last_region = db_rec_last['region'][0]
            pair_key = "{}->{}".format(last_region, region)
            if not (pair_key in stats['invalid-pairs']):
                stats['invalid-pairs'][pair_key] = 0
            stats['invalid-pairs'][pair_key] += 1
            
        if not (region in stats['invalid']):
            stats['invalid'][region] = 0
        stats['invalid'][region] += 1

        db_rec_last = db_rec

    return stats

    
def _graph_n_syn_coupled_astro(db_rec, n_synapse, fig=None, axes=None):

    if fig is None or axes is None:
        gs = plot.gs(n_synapse + 2, 1)
        fig, axes = plot.gen_axes(
            ('spikes', gs[0]),
            ('regions', gs[-1]),
            figsize=(15,12),
        )

        for i in range(n_synapse):
            fig, axes = plot.gen_axes(
                ('astro', gs[i+1]),
                fig=fig, axes=axes
            )

    plot.plot_spikes(
        axes, ('spikes',),
        db_rec['tl']['z_pre'],
        db_rec['tl']['z_post'])

    plot.plot_astro(
        axes, ('astro',),
        db_rec['tl']['ip3'], db_rec['tl']['kp'], db_rec['tl']['ca'], db_rec['tl']['dser'], db_rec['tl']['serca'])

    if 'region' in db_rec:
        plot.plot_coupling_region(
            axes, ('regions',),
            db_rec['region'])


    return fig, axes


def _graph_n_syn_coupled_astro_lif(
    db_rec,
    n_synapse,
    xlim=None,
    w_dw_only=False,
    w_err_only=False,
    fig=None,
    axes=None
):

    if (fig is None) or (axes is None):
        if w_err_only or w_dw_only:
            n_plots = 2
        else:
            n_plots = n_synapse + 3

        gs = plot.gs(n_plots, 1)

        if w_err_only:
            fig, axes = plot.gen_axes(
                ('w', gs[0]),
                ('err', gs[1]),
                figsize=(8,6),
            )
        elif w_dw_only:
            fig, axes = plot.gen_axes(
                ('dw', gs[0]),
                ('w', gs[1]),
                figsize=(15,8),
            )

        else:
            fig, axes = plot.gen_axes(
                ('spikes', gs[0]),
                ('dw', gs[-2]),
                ('w', gs[-1]),
                figsize=(15,12),
            )
            for i in range(n_synapse):
                fig, axes = plot.gen_axes(
                    ('astro', gs[i+1]),
                    fig=fig, axes=axes
                )

    if not (w_dw_only or w_err_only):
        _graph_n_syn_coupled_astro(db_rec, n_synapse, fig=fig, axes=axes)

    if not w_err_only:
        plot.plot_dw(
            axes, ('dw',),
            db_rec['tl']['dw'])

    plot.plot_w(
        axes, ('w',),
        db_rec['tl']['w'])

    if w_err_only:
        plot.plot_err(
            axes, ('err',),
            db_rec['err'])

        

    if not (xlim is None):
        plot.xlim(axes, xlim)

    return fig, axes


def _graph_n_syn(db, xlim=None, w=None, prefix=None, graphs=None):
    descr = db.meta['descr']

    if not (w is None):
        w_match = []
        for w_rec in tqdm(w, desc='Filter Sims'):
            w_rec = torch.as_tensor(w_rec)
            dists = []
            for db_rec in db:
                dists.append(torch.norm(db_rec['w'] - w_rec))
            dists = torch.as_tensor(dists)
            w_match.append(int(dists.argmin()))

        db = db.subset(w_match)

    fig, axes = gen_sgnn_axes(
        2,
        offset=True,
        figsize=(18, 14),
        graphs=graphs
    )

    if prefix is None and len(db) > 1:
        raise ValueError("To graph multiple db_rec's, a prefix is needed")

    for db_rec in db:
        prefix_i = None
        if not (prefix is None):
            prefix_i = ["{:4.2f}".format(w) for w in db_rec[prefix[1]]]
            prefix_i = ",".join(prefix_i)
            prefix_i = prefix[0].format(prefix_i)
        graph_sgnn(db_rec, fig, axes, 0, prefix=prefix_i, plot=graphs)

    # Restrict all graphs to xlim range
    if not (xlim is None):
        descr = descr + "_xlim"
        i = 0
        for _, ax in axes.items():
            # ax is [sp0, sp1, ...] where spN is [syn0, syn1, ...] or just spN
            if type(ax) == list:
                assert len(ax) == 1
                ax = ax[0]
            else:
                raise ValueError("Expected each axis to be a list of axes, one for each subplot")

            if not (type(ax) == list):
                ax = [ax]

            for a in ax:
                a.set_xlim(*xlim)
    else:
        descr = descr + "_tl"

    fig_path = "{}.svg".format(descr)
    print("Saving: ", fig_path)
    fig.savefig(fig_path)

    
def _spikes_gen(
    cfg_gen,
    window=10,
    num_impulses=5,
    padding=10,
    gen_post=False,
    concat=False,
    no_post_pre=False
):

    n = None
    for cfg in cfg_gen:
        if n is None or not (n == cfg['linear_params']['synapse']):
            n = cfg['linear_params']['synapse']
            spikes = gen_and_spikes(
                n, window=window,
                num_impulses=num_impulses,
                gen_post=gen_post,
                padding=padding,
                no_post_pre=no_post_pre,
            )

            if concat:
                spikes = spikes.view(1, spikes.shape[0]*spikes.shape[1], *spikes.shape[2:])


        yield cfg, spikes

    
def _cfg_gen(
    cfg_path,
    astro_p=True,
    w_sweep=None,
    n=2,
    all_stdp=True,
    all_ip3_ca=True,
    dw=True,
    c_and=[],
    all_and=False,
    c_nand=[],
    ca_th=None):
    """
    Generate some number of configs based on params
    """

    def cfg_apply_settings(cfgs, **kwargs):
        """
        Static Settings that don't result in additional configs
        """
        n = cfg['linear_params']['synapse']
        if 'n' in kwargs:
            n = kwargs['n']
            cfg['linear_params']['synapse'] = n
        if 'w' in kwargs:
            cfg['linear_params']['mu'] = kwargs['w']
            cfg['linear_params']['init'] = 'fixed'

        cfg['sim']['dw'] = dw
        if astro_p:
            cfg['astro_params'] = cfg['astro_plasticity']
        if not (ca_th is None):
            cfg['astro_params']['ca_th'] = ca_th

        # Figure out synapse association with STDP, ip3->ca, and coupling
        if all_stdp:
            cfg['astro_params']['local']['stdp'] = list(range(n))
        if all_ip3_ca:
            cfg['astro_params']['local']['ip3_ca'] = list(range(n))
        if all_and:
            c_and = list(range(n))

        if any([(i in c_nand) for i in c_and]):
            raise ValueError("There can't be overlap between c_and and c_nand")

        if all([i < n for i in c_and]):
            cfg['astro_params']['coupling']['and'] = c_and
        if all([i < n for i in c_nand]):
            cfg['astro_params']['coupling']['nand'] = c_nand

        return cfg


    def cfg_sweep_w(cfgs):
        """
        For each input config, generate N configs, one for each w
        """
        w_min, w_max, num_w, method = w_sweep
        for cfg in cfgs:
            all_w = []
            for i in range(cfg['linear_params']['synapse']):
                if method == 'linspace':
                    idx = torch.randperm(w_sweep[2])
                    all_w.append(torch.linspace(w_min, w_max, num_w)[idx])
                elif method == 'random':
                    all_w.append((w_max - w_min) * torch.rand(num_w) + w_min)
                    
            all_w = torch.vstack(all_w).permute((1,0))

            for w in all_w:
                cfg = cfg_apply_settings(cfg, w=w.tolist())
                yield cfg


    def cfg_sweep_n(cfgs):
        """
        For each input config, generate N configs, one for each value n
        """

        for cfg in cfgs:
            for n_i in n:
                cfg = cfg_apply_settings(cfg, n=n_i)
                yield cfg

    # Fix input types
    if type(n) in [int, float]:
        n = [n]

    # Get config from file
    cfg = config.Config(cfg_path)
    cfgs = [cfg]
    
    num_cfgs = 1

    # Always apply a number of synapses
    cfgs = cfg_sweep_n(cfgs)
    num_cfgs *= len(n)

    # Sweep W
    if w_sweep:
        cfgs = cfg_sweep_w(cfgs)
        num_cfgs *= w_sweep[2]

    return cfgs, num_cfgs


def _exp_n_syn_coupled_astro_lif(
    rt_gen,
    num_cfgs,
    sim=False,
    keep_state=False,
    dw=False,
    sim_name='_exp_n_syn_astro',
    descr='snn_1n{}s1a_and'
):
    seed_many()

    # db_path = Path("{}.db".format(sim_name))

    # if not sim:
    #     print("Checking if db is in: ", db_path, end='')
    #     if db_path.exists():
    #         print('... yes, returning')
    #         db = ExpStorage(path=db_path)
    #         return db
    #     else:
    #         print('... no, re-running')

    dbs = []
    for cfg, spikes in tqdm(rt_gen, total=num_cfgs):
        db = ExpStorage()
        db.meta['descr'] = descr
        db.meta['n'] = cfg['linear_params']['synapse']
        db.meta['w'] = cfg['linear_params']['mu']

        # print("n={}, spikes.sum(): {}, cfg-hash: {}".format(db.meta['n'], spikes.sum(), cfg.md5()))
        sim_lif_astro_net(cfg, spikes, db, dw=dw, keep_state=keep_state, err=True)

        dbs.append(db)

    # db.save(db_path)
    return dbs


def _exp_n_syn_coupled_astro(
    rt_gen,
    num_cfgs,
    sim=False,
    keep_state=False,
    sim_name='_exp_{}_syn_astro',
    descr='astro_{}s1a_and'
):
    seed_many()

    db = ExpStorage()

    if keep_state:
        descr = descr + '_cont'
    db.meta['descr'] = descr

    n = None
    spikes = None
    for cfg, spikes in tqdm(rt_gen, total=num_cfgs):
        # # Attempt to load db, assume number of synapse doesn't change
        # if n is None:
        #     n = cfg['linear_params']['synapse']
        #     sim_name = sim_name.format(n)

        #     db_path = Path("{}.db".format(sim_name))
        #     if not sim:
        #         if db_path.exists():
        #             db = ExpStorage(path=db_path)
        #             return db

        sim_astro(cfg, spikes, db, keep_state=keep_state)

        for i, db_rec in enumerate(db):
            db_rec['region'] = astro_and_region(db_rec['tl'])
            db_rec['invalid'] = not astro_check_respose(db_rec['tl'], db_rec['region'][0])

    return db


def _exp_n_syn_coupled(
    rt_gen,
    num_cfgs,
    sim=False,
    sim_name='_exp_n_syn_poisson'
):
    seed_many()

    db = ExpStorage()

    n = None
    spikes = None
    for cfg, spikes in tqdm(rt_gen, total=num_cfgs):
        assert (n is None) or (n == cfg['linear_params']['synapse'])

        ss = spikes.shape
        spikes = spikes.view((ss[0]*ss[1], *ss[2:]))

        # Attempt to load db, assume number of synapse doesn't change
        if n is None:
            n = cfg['linear_params']['synapse']
            sim_name = sim_name.format(n)

        db_path = Path("{}.db".format(sim_name))
        if not sim:
            if db_path.exists():
                db = ExpStorage(path=db_path)
                return db

        sim_lif_astro_net(cfg, [spikes], db, dw=cfg['sim']['dw'])

    num_invalid = 0
    for i, db_rec in enumerate(db):
        db_rec['region'] = astro_and_region(db_rec['tl'])
        db_rec['invalid'] = not astro_check_respose(db_rec['tl'], db_rec['region'][0])
        
        if db_rec['invalid']:
            num_invalid += 1

    db.meta['descr'] = sim_name
    db.save(db_path)

    return db


def _exp_n_syn_poisson(
    rt_gen,
    num_cfgs,
    sim=False,
    sim_name='_exp_n_syn_poisson'
):
    """
    Simulate in the 1nNs1a configuration with poisson input
    """

    seed_many()

    spikes = gen_impulse_spikes(1, num_impulses=5, poisson=True, rate=0.8)
    spikes += gen_impulse_spikes(1, num_impulses=5, poisson=True, rate=0.8)
    spikes = torch.cat(spikes, axis=-1)

    db = ExpStorage()

    n = None
    for cfg, spikes in tqdm(rt_gen, total=num_cfgs, position=0):
        assert (n is None) or (n == cfg['linear_params']['synapse'])
        n = cfg['linear_params']['synapse']

        # Attempt to load db, assume number of synapse doesn't change
        sim_name = sim_name.format(n)

        db_path = Path("{}.db".format(sim_name))
        if not sim:
            if db_path.exists():
                db = ExpStorage(path=db_path)
                return db

        sim_lif_astro_net(cfg, [spikes], db, dw=cfg['sim']['dw'])

    db.meta['descr'] = sim_name

    db.save(db_path)
    return db


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--astro-nsyn-poisson', action='store_true')
    parser.add_argument('--astro-nsyn-and', action='store_true')
    parser.add_argument('--astro-and-spike-response', action='store_true')
    parser.add_argument('--astro-and-cont', action='store_true')
    parser.add_argument('--astro-and-lif', action='store_true')
    parser.add_argument('--astro-and-lif-w', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--sim', action='store_true')

    return parser.parse_args()


def _main(args):
    os.makedirs('./obj3', exist_ok=True)
    os.chdir('./obj3')

    cfg_path = '../../config/1nNs1a_obj3.yaml'

    w_inspect = [
        (0.6316, 0.8421),
        (0.9474, 0.8421),
        # (1.0526, 0.8421),
        # (0.8421, 0.8421),
    ]
    w_sweep = (0.5, 1.0, 50)
    # w_sweep = (4.0, 10.0, 20)
    ca_th = 0.81
    xlim = (75, 100)

    plot.rc_config({'font.size': 14})

    if args.astro_nsyn_poisson:
        cfgs, num_cfgs = _cfg_gen(
            cfg_path,
            n=2, ca_th=ca_th, dw=False,
            w_sweep=w_sweep)

        cfg_and_spikes = _spikes_gen(cfgs, num_impulses=5)

        db = _exp_n_syn_poisson(cfgs_and_spikes, num_cfgs, sim=args.sim, sim_name='snn_1n{}s1a_poisson')

        _graph_sweep_w(db)

        # Graph timelines for these weight combinations
        _graph_n_syn(
            db, w=w_inspect,
            prefix = ('w={}', 'w'),
            graphs = ['spikes', 'astro-ca']
        )

        _graph_n_syn(
            db, w=w_inspect,
            prefix = ('w={}', 'w'),
            graphs = ['spikes', 'astro-ca'],
            xlim=xlim,
        )

    # Look at a number of 10ms bouts of activity
    # Determine if the Astrocyte responded correctly
    if args.astro_and_spike_response or args.all:
        # Generate single pre, pre, post events in a given time window
        # and examine astrocyte response.
        n_synapse = [2,3,4]

        cfgs, num_cfgs = _cfg_gen(
            cfg_path,
            n=n_synapse, ca_th=ca_th, dw=False, all_and=True
        )
        cfg_and_spikes = _spikes_gen(cfgs, window=10, num_impulses=1000, gen_post=True)

        db = _exp_n_syn_coupled_astro(cfg_and_spikes, num_cfgs, sim=args.sim)

        for n, db_syn in db.group_by('n').items():
            if len(db_invalid) > 10:
                db_cat = db_invalid.slice(0, 10).cat()
            elif len(db_invalid) == 0:
                db_cat = db_syn.slice(0, 10).cat()
            else:
                db_cat = db_invalid.cat()

            fig, axes = _graph_n_syn_coupled_astro(db_cat, n)

            fig_path = "{}.svg".format(db_syn.meta['descr'].format(n))
            print('Saving: ', fig_path)
            fig.tight_layout()
            fig.savefig(fig_path)

    # Look at a number of continuous 10ms bouts of activity
    # Determine if the Astrocyte responded correctly
    if args.astro_and_cont or args.all:
        # Generate single pre, pre, post events in a given time window
        # and examine astrocyte response.
        n_synapse = [2,3,4]

        cfgs, num_cfgs = _cfg_gen(
            cfg_path,
            n=n_synapse, ca_th=ca_th, dw=False, all_and=True
        )

        cfg_and_spikes = _spikes_gen(
            cfgs, window=10, num_impulses=5000, gen_post=True, no_post_pre=True)

        db = _exp_n_syn_coupled_astro(
            cfg_and_spikes,
            num_cfgs,
            sim=args.sim,
            keep_state=True
        )

        for n, db_syn in db.group_by('n').items():
            invalid_regions = {}
            for i, db_rec in enumerate(db_syn):
                if db_rec['invalid'] and not (db_rec['region'][0] in invalid_regions):
                    invalid_regions[db_rec['region'][0]] = None
                    db_rec['graph'] = True
                    if i > 0: db_syn[i - 1]['graph'] = True
                    if i < len(db_syn) - 1: db_syn[i + 1]['graph'] = True

                if len(invalid_regions) == 5:
                    break

            db_graph = db_syn.filter(graph=True).cat()

            fig, axes = _graph_n_syn_coupled_astro(db_graph, n)

            fig_path = "{}_invalid.svg".format(db_syn.meta['descr'].format(n))
            print('Saving: ', fig_path)
            fig.tight_layout()
            fig.savefig(fig_path)

            # Bar graph            
            stats = _n_syn_coupled_astro_stats(db_syn, n)
            stats_path = "{}_stats.yaml".format(db_syn.meta['descr'].format(n))
            print('Saving: ', stats_path)
            with open(stats_path, 'w') as fp: yaml.dump(stats, fp)

            gs = plot.gs(1, 1)
            fig, axes = plot.gen_axes(
                ('stats', gs[0]),
                figsize=(10,6),
            )

            total_invalid = stats['total'] - stats['valid']
            bar_spec = {k: v/total_invalid for k,v in stats['invalid'].items()}

            plot.plot_mismatch_bar(
                axes, ('stats',),
                bar_spec)

            fig_path = "{}_stats.svg".format(db_syn.meta['descr'].format(n))
            print('Saving: ', fig_path)
            fig.savefig(fig_path)

    # Look at a number of continuous 10ms bouts of inputs, simulate an Astro-LIF Configuration
    if args.astro_and_lif or args.all:
        ca_th = 0.85
        n_synapse = [2,3,4]
        # n_synapse = [4]

        cfgs, num_cfgs = _cfg_gen(
            cfg_path,
            n=n_synapse,
            ca_th=ca_th,
            dw=False,
            all_and=True,
        )
        cfg_and_spikes = _spikes_gen(cfgs, window=10, num_impulses=500, gen_post=False)

        dbs = _exp_n_syn_coupled_astro_lif(cfg_and_spikes, num_cfgs, sim=args.sim, dw=True, keep_state=True)

        # Graph for each distinct value of n
        # for n, db_syn in db.group_by('n').items():
        for db_syn in dbs:
            n = db_syn.meta['n']

            # Get plot descr
            descr = db_syn.meta['descr'].format(n)

            # Cat all entries in db together
            db_cat = db_syn.cat()

            # Graph error
            gs = plot.gs(1, 1)
            fig, axes = plot.gen_axes(
                ('err', gs[0]),
                figsize=(8,6),
            )
            plot.plot_err(axes, ('err',), db_cat['err'])
            fig_path = "{}_err.svg".format(descr)
            print("Saving: ", fig_path)
            fig.tight_layout()
            fig.savefig(fig_path)

            # Graph weight timeline
            fig, axes = _graph_n_syn_coupled_astro_lif(db_cat, n, xlim=None, w_err_only=True)
            fig_path = "{}.svg".format(descr)
            print('Saving: ', fig_path)
            fig.tight_layout()
            fig.savefig(fig_path)

            # Limit X range
            fig, axes = _graph_n_syn_coupled_astro_lif(db_cat, n, xlim=(0,200))
            fig_path = "{}_xlim.svg".format(descr)
            print('Saving: ', fig_path)
            fig.tight_layout()
            fig.savefig(fig_path)



    if args.astro_and_lif_w or args.all:
        # 2 Synapse, 5 different random weight configs
        n_synapse = 2
        cfgs, num_cfgs = _cfg_gen(
            cfg_path,
            n=n_synapse,
            ca_th=ca_th,
            dw=False,
            all_and=True,
            w_sweep=(0.0,2.0,5,'random')
        )
        cfg_and_spikes = _spikes_gen(cfgs, window=10, num_impulses=500, gen_post=False)

        dbs = _exp_n_syn_coupled_astro_lif(cfg_and_spikes, num_cfgs, sim=args.sim, dw=True, keep_state=True, descr='snn_{}s1a_and_w')

        # Graph timline for each value of w
        fig, axes = None, None
        for db_syn in dbs:
            w = db_syn.meta['w']

            if fig is None:
                fig, axes = _graph_n_syn_coupled_astro_lif(db_syn.cat(), n_synapse, xlim=None, w_err_only=True)
            else:
                fig, axes = _graph_n_syn_coupled_astro_lif(db_syn.cat(), n_synapse, xlim=None, w_err_only=True, fig=fig, axes=axes)

        descr = dbs[0].meta['descr'].format(n_synapse)
        fig_path = "{}.svg".format(descr)
        print('Saving: ', fig_path)
        fig.tight_layout()
        fig.savefig(fig_path)

        # 4 Synapse, 2 different random weight configs
        n_synapse = 4
        cfgs, num_cfgs = _cfg_gen(
            cfg_path,
            n=n_synapse,
            ca_th=ca_th,
            dw=False,
            all_and=True,
            w_sweep=(0.0,2.0,2,'random')
        )
        cfg_and_spikes = _spikes_gen(cfgs, window=10, num_impulses=500, gen_post=False)

        dbs = _exp_n_syn_coupled_astro_lif(cfg_and_spikes, num_cfgs, sim=args.sim, dw=True, keep_state=True, descr='snn_{}s1a_and_w')

        # Graph timline for each value of w
        fig, axes = None, None
        for db_syn in dbs:
            w = db_syn.meta['w']

            if fig is None:
                fig, axes = _graph_n_syn_coupled_astro_lif(db_syn.cat(), n_synapse, xlim=None, w_err_only=True)
            else:
                fig, axes = _graph_n_syn_coupled_astro_lif(db_syn.cat(), n_synapse, xlim=None, w_err_only=True, fig=fig, axes=axes)

        descr = dbs[0].meta['descr'].format(n_synapse)
        fig_path = "{}.svg".format(descr)
        print('Saving: ', fig_path)
        fig.tight_layout()
        fig.savefig(fig_path)

if __name__ == '__main__':
    args = _parse_args()
    with torch.no_grad():
        _main(args)
