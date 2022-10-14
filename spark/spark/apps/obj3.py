import argparse
import os
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import itertools

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
    astro_and_region
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


def _graph_sweep_w(db):
    w_points = None

    for d in db:
        ca_abs_max_idx = d['tl']['ca'].abs().argmax(dim=0)
        ca_max = d['tl']['ca'][ca_abs_max_idx].max()

        w_point = torch.as_tensor([*d['w'], ca_max]).reshape(1, -1)

        # Build tensor of weights and Ca activity
        if w_points is None:
            w_points = w_point
        else:
            w_points = torch.cat((w_points, w_point), axis=0)

    w0 = torch.unique(w_points[:, 0])
    w1 = torch.unique(w_points[:, 1])

    fig = plt.Figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    ax.set_title("Ca Threshold Events Given synaptic weights w0 and w1")
    ax.set_yticks(
        list(range(len(w1))),
        labels=["{:2.4f}".format(float(a)) for a in w1],
        rotation=45)
    ax.set_ylabel('W0')

    ax.set_xticks(
        list(range(len(w0))),
        labels=["{:2.4f}".format(float(a)) for a in w0])
    ax.set_xlabel('W1')

    heat_img = torch.zeros((len(w0), len(w1)))

    assert len(w0) * len(w1) == w_points.shape[0]

    for i in range(w_points.shape[0]):
        x = i // len(w1)
        y = i % len(w1)

        ca_act = w_points[i, 2]
        w0_i = w_points[i, 0]
        w1_i = w_points[i, 1]

        heat_img[x, y] = ca_act
        ax.text(
            y, x,
            "{:1.2f}".format(float(heat_img[x, y])),
            ha="center", va="center", color="w")

    ax.imshow(heat_img)
    fig.tight_layout()

    fig_path = "{}_syn0.svg".format(db.meta['descr'])
    print("Saving: ", fig_path)
    fig.savefig(fig_path)


def _graph_n_syn_cat(db, graphs=None, xlim=None):
    def _cat_dict(d_a, d_b):
        _next = [(d_a, d_b)]
        while len(_next) > 0:
            a, b = _next.pop(0)

            for k in a:
                if not (k in b):
                    continue

                if type(a[k]) == dict:
                    _next.append((a[k], b[k]))
                elif type(a[k]) == torch.Tensor:
                    a[k] = torch.cat((a[k], b[k]))
                else:
                    raise ValueError("Can't merge type: {}", type(a[k]))

        return d_a


    descr = db.meta['descr']

    db_rec_cat = None
    for db_rec in db:
        if db_rec_cat is None:
            db_rec_cat = db_rec
            continue

        db_rec_cat = _cat_dict(db_rec_cat, db_rec)

    fig, axes = gen_sgnn_axes(
        2,
        offset=True,
        figsize=(18, 14),
        graphs=graphs
    )

    graph_sgnn(db_rec_cat, fig, axes, 0, plot=graphs)


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
    gen_post=False):

    n = None
    for cfg in cfg_gen:
        if n is None or not (n == cfg['linear_params']['synapse']):
            n = cfg['linear_params']['synapse']
            spikes = gen_and_spikes(n, window=window, num_impulses=num_impulses, gen_post=gen_post, padding=padding)

        yield cfg, spikes

    
def _cfg_gen(
    cfg_path,
    n=2,
    astro_p=True,
    w_sweep=None,
    dw=True,
    c_and=[],
    c_nand=[],
    ca_th=None):
    """
    Generate some number of configs based on params
    """

    def cfg_apply_settings(cfgs):
        """
        Static Settings that don't result in additional configs
        """
        for cfg in cfgs:
            cfg['linear_params']['synapse'] = n
            cfg['sim']['dw'] = dw
            if astro_p:
                cfg['astro_params'] = cfg['astro_plasticity']
            if not (ca_th is None):
                cfg['astro_params']['ca_th'] = ca_th

            if any([(i in c_nand) for i in c_and]):
                raise ValueError("There can't be overlap between c_and and c_nand")

            if all([i < n for i in c_and]):
                cfg['astro_params']['coupling']['and'] = c_and
            if all([i < n for i in c_nand]):
                cfg['astro_params']['coupling']['nand'] = c_nand

            yield cfg


    def cfg_sweep_w(cfgs):
        """
        For each input config, generate N configs, one for each w
        """
        all_w = [torch.linspace(*w_sweep) for i in range(n)]
        all_w = torch.cartesian_prod(*all_w)

        for cfg in cfgs:
            for w in all_w:
                cfg['linear_params']['mu'] = w
                cfg['linear_params']['init'] = 'fixed'

                yield cfg

    cfg = config.Config(cfg_path)
    num_cfgs = 1

    cfgs = cfg_apply_settings([cfg])

    if w_sweep:
        cfgs = cfg_sweep_w(cfgs)
        num_cfgs *= w_sweep[2]**2

    return cfgs, num_cfgs


def _exp_n_syn_coupled_astro(
    rt_gen,
    num_cfgs,
    sim=False,
    sim_name='_exp_{}_syn_astro'
):
    seed_many()

    db = ExpStorage()

    n = None
    spikes = None
    for cfg, spikes in tqdm(rt_gen, total=num_cfgs):
        assert (n is None) or (n == cfg['linear_params']['synapse'])

        # Attempt to load db, assume number of synapse doesn't change
        if n is None:
            n = cfg['linear_params']['synapse']
            sim_name = sim_name.format(n)

            db_path = Path("{}.db".format(sim_name))
            if not sim:
                if db_path.exists():
                    db = ExpStorage(path=db_path)
                    return db

            db.meta['descr'] = 'astro_{}s1a_and'.format(n)

        sim_astro(cfg, spikes, db)


        for db_rec in db:
            db_rec['region'] = [astro_and_region(db_rec['tl'])]

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
    for cfg, spikes in tqdm(rt_gen, total=num_cfgs):
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
    ca_th = 0.8
    xlim = (75, 100)

    if args.astro_nsyn_poisson or args.all:
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


    if args.astro_nsyn_and or args.all:

        # Examine response to single spiking events (on each synapse) with settle time between
        cfgs, num_cfgs = _cfg_gen(
            cfg_path,
            n=2, ca_th=ca_th, dw=False, c_and=[0,1],
            w_sweep=w_sweep)

        cfg_and_spikes = _spikes_gen(cfgs, window=10, num_impulses=5, padding=40)
        db = _exp_n_syn_coupled(cfg_and_spikes, num_cfgs, sim=args.sim, sim_name='snn_1n{}s1a_and_poisson')

        _graph_sweep_w(db)

        _graph_n_syn(
            db, w=w_inspect,
            prefix = ('w={}', 'w'),
            graphs = ['spikes', 'astro', 'neuron']
        )

        _graph_n_syn(
            db, w=w_inspect,
            prefix = ('w={}', 'w'),
            graphs = ['spikes', 'astro'],
            xlim=xlim,
        )

        # Examine response to the same set of spikes repeated multiple times, each time with a different weight value
        cfgs, num_cfgs = _cfg_gen(
            cfg_path,
            n=2, ca_th=ca_th, dw=False, c_and=[0,1],
            w_sweep=w_sweep)
        cfg_and_spikes = _spikes_gen(cfgs, window=10, num_impulses=1)

        db = _exp_n_syn_coupled(cfg_and_spikes, num_cfgs, sim=args.sim, sim_name='snn_1n{}s1a_and_poisson_fixed')

        _graph_n_syn_cat(db, graphs=['spikes', 'astro'], xlim=(300,800))


    if args.astro_and_spike_response or args.all:
        # Generate single pre, pre, post events in a given time window
        # and examine astrocyte response.
        n_synapse = 2

        cfgs, num_cfgs = _cfg_gen(
            cfg_path,
            n=n_synapse, ca_th=ca_th, dw=False, c_and=[0,1]
        )

        cfg_and_spikes = _spikes_gen(cfgs, window=10, num_impulses=10, gen_post=True)

        db = _exp_n_syn_coupled_astro(cfg_and_spikes, num_cfgs, sim=args.sim)

        # Inspect db, and see how many impulses have activity matching the region
        num_invalid=0
        for i, db_rec in enumerate(db):
            val_res=True  # Check for valid response
            region = db_rec['region'][0][0]

            any_pre_spike = torch.any(db_rec['tl']['z_pre'] > 0.5, dim=0)
            no_serca = torch.all(db_rec['tl']['serca'] < 0.5, dim=0)
            yes_serca = torch.any(db_rec['tl']['serca'] > 0.5, dim=0)
            no_dser= torch.all(torch.abs(db_rec['tl']['dser']) < 0.5, dim=0)
            yes_dser_ltp = torch.any(db_rec['tl']['dser'] > 0.5, dim=0)
            yes_dser_ltd = torch.any(db_rec['tl']['dser'] < -0.5, dim=0)


            if region in ['other-influence', 'and']:
                val_res = torch.all(yes_serca == any_pre_spike) and torch.all(no_dser)

            elif region == 'ltp':
                val_res = torch.all(no_serca) and torch.all(yes_dser_ltp)

            elif region == 'early-spike':
                val_res = torch.any(yes_serca) and torch.any(yes_dser_ltd)

            if not (val_res):
                print("{} ({}) Invalid".format(region, i))
                # import code
                # code.interact(local=dict(globals(), **locals()))
                # exit(1)
                
                num_invalid += 1
                print()

        print("{}/{} Invalid".format(num_invalid, i+1))

                    
        
        # Graph: spikes, synapse specific ip3, k+ and Ca, category of event (AND, Early spike, etc..)
        # Layout: Spikes in a SP (lables on X axis) Other traces in separate SPs with labels in title
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

        db_cat = db.cat()

        plot.plot_spikes(
            axes, ('spikes',),
            db_cat['tl']['z_pre'],
            db_cat['tl']['z_post'])

        plot.plot_astro(
            axes, ('astro',),
            db_cat['tl']['ip3'], db_cat['tl']['kp'], db_cat['tl']['ca'], db_cat['tl']['dser'], db_cat['tl']['serca'])

        plot.plot_coupling_region(
            axes, ('regions',),
            db_cat['region'])

        fig_path = "{}.svg".format(db.meta['descr'])
        print('Saving: ', fig_path)
        fig.tight_layout()
        fig.savefig(fig_path)        
            

if __name__ == '__main__':
    args = _parse_args()
    with torch.no_grad():
        _main(args)