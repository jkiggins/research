import torch

from ..module.astrocyte import Astro
from ..module.neuron import LIFNeuron
from ..utils import config, plot
from ..data import spiketrain
from ..experiment import VSweep, ExpStorage

import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

def _astro_sim(astro, cfg, spikes, db):
    state = None
    
    timeline = {
        key: torch.as_tensor([val])
        for key, val in [
            ('ip3', 0.0),
            ('kp', 0.0),
            ('z_pre', 0.0),
            ('z_post', 0.0),
            ('ca', 0.0),
            ('eff', 0.0),
        ]
    }

    for z in tqdm(spikes.transpose(1,0)):
        if len(z) == 1:
            eff, state = astro(state, z_pre=z[0], z_post=torch.zeros(1)[0])
            timeline['z_pre'] = torch.cat((timeline['z_pre'], z.reshape(1)))
        elif len(z) == 2:
            eff, state = astro(state, z_pre=z[0], z_post=z[1])
            timeline['z_pre'] = torch.cat((timeline['z_pre'], z[0].reshape(1)))
            timeline['z_post'] = torch.cat((timeline['z_post'], z[1].reshape(1)))
            timeline['kp'] = torch.cat((timeline['kp'], state['kp'].reshape(1)))
            
        timeline['eff'] = torch.cat((timeline['eff'], eff.reshape(1)))
        timeline['ip3'] = torch.cat((timeline['ip3'], state['ip3'].reshape(1)))

        timeline['ca'] = torch.cat((timeline['ca'], state['ca'].reshape(1)))

    db.store({
        'timeline': timeline
    })


def sim_sweep_u(cfg):
    tau_u = torch.linspace(50, 500, 20)
    db = ExpStorage()
    
    spike_trains = []
    
    impulse_spikes = spiketrain.impulse(0, 10, 100).repeat((2,1))
    spike_trains.append(impulse_spikes)
    for r in [0.1, 0.5, 0.7]:
        spike_trains.append(spiketrain.poisson([r, r], 100))

    suffix = ['impulse', 'p0.1', 'p0.5', 'p0.7']

    param_sweep = VSweep(tau_u)
    param_sweep = param_sweep.foreach(spike_trains)

    for tau_u, spikes in param_sweep.head.run():
        cfg('astro_params.tau_u', tau_u)
        astro = Astro.from_cfg(cfg['astro_params'], 1, cfg['sim']['dt'])
        _astro_sim(astro, cfg, spikes, db)
        db.last()['tau_u'] = tau_u
        db.last()['spikes'] = spikes


    # One figure per spike-train
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
            ax.plot(tl['ca'], label='tau_u={}'.format(d['tau_u']))
            ax.set_xlim((0, len(tl['z_pre'])))
        # ax.legend()


        # ip3, kp, and spikes are the same across varying u
        tl = by_spike[0]['timeline']
        ip3 = tl['ip3']
        kp = tl['kp']

        ax = fig.add_subplot(3,1,2)
        ax.set_title("Astrocyte Input Traces")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Value")
        c1 = ax.plot(ip3, label='ip3')[0].get_color()
        c2 = ax.plot(kp, label='k+')[0].get_color()
        ax.legend()

        ax = fig.add_subplot(3,1,3)
        ax.set_ylabel("Spikes")
        ax.set_xlabel("Time (ms)")
        plot.plot_events(ax, spikes.tolist(), colors=(c1,c2))
        ax.set_title("Pre and Post-synapic Spikes")

        fig.tight_layout()
        fig.savefig("sweep_u_tau_{}.svg".format(suffix[i]))


def sim_sweep_pre_i(cfg):
    alpha_ip3_vals = torch.linspace(0.1, 1.0, 3)
    tau_ip3_vals = torch.logspace(1, 3, 5)
    spike_trains = []
    spike_trains.append(spiketrain.impulse(0, 10, 100))

    for r in [0.1, 0.5, 0.7]:
        spike_trains.append(spiketrain.poisson(r, 100))

    param_sweep = VSweep(tau_ip3_vals)
    param_sweep = param_sweep.foreach(alpha_ip3_vals)
    param_sweep = param_sweep.foreach(spike_trains)
    db = ExpStorage()

    # Simulation
    for tau_i, alpha, spikes in param_sweep.head.run():
        cfg('astro_params.tau_ip3', tau_i)
        cfg('astro_params.alpha_pre', alpha)

        astro = Astro.from_cfg(cfg['astro_params'], 1, cfg['sim']['dt'])
        _astro_sim(astro, cfg, spikes, db)
        
        db.last()['tau_i'] = float(tau_i)
        db.last()['alpha'] = float(alpha)
        db.last()['spikes'] = spikes

    # Get the timelines by spike train
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

            # Plot ip3 and u for each tau on a single plot
            for d in by_alpha:
                tl = d['timeline']
                spikes = d['spikes']
                ax.plot(tl['ip3'].tolist(), label='tau_ip3={}'.format(d['tau_i']))
                ax.set_xlim((0, len(tl['z_pre'])))
            ax.legend()

        # Last subplot has spike train
        ax = fig.add_subplot(num_subplots, 1, num_subplots)
        
        plot.plot_events(ax, spikes)
        ax.set_title("Spikes over time")
        ax.legend(['Z In'])

        fig.tight_layout()
        fig.savefig(
            'sweep-alpha_tau-ip3-spike{:1.3f}.svg'.format(
                torch.mean(spikes)
            )
        )


def sim_heatmap_alpha_u_thr_events(cfg):
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

        astro = Astro.from_cfg(cfg['astro_params'], 1, cfg['sim']['dt'])
        pre_spikes = spiketrain.poisson(spike_rate, 1000)
        post_spikes = spiketrain.poisson(spike_rate, 1000)
        state = None
        timeline = {
            'eff': [],
            'ca': []
        }

        for z_pre, z_post in zip(pre_spikes[0], post_spikes[0]):
            eff, state = astro(state, z_pre=z_pre, z_post=z_post)
            timeline['eff'].append(
                torch.isclose(eff, torch.as_tensor(1.0)).float()
            )
            timeline['ca'].append(state['ca'])

        # print(
        #     "alpha: {}, spike rate: {}, any_effect: {}, max(u): {}".format(
        #         alpha,
        #         spike_rate,
        #         any(timeline['eff']),
        #         max(timeline['ca']),
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
            eff = torch.as_tensor(tl['eff'])
            
            heat_img[j, i] = (1.0 - eff.mean()) * 1000.0
            ax.text(
                i, j,
                "{:1.2f}".format(float(1000.0 - heat_img[j, i])),
                ha="center", va="center", color="w")

    ax.imshow(heat_img)
    fig.tight_layout()
    fig.savefig("u_thr_event_heatmap_spike-rate_alpha.svg")



def sim_heatmap_tau_u_thr_events(cfg):
    spike_rate_range = torch.linspace(0.05, 0.8, 20)
    tau_range = torch.linspace(50, 500, 20)

    param_sweep = VSweep(values=spike_rate_range)
    param_sweep = param_sweep.foreach(tau_range)

    dt = cfg['sim']['dt']

    db = ExpStorage()

    # Simulate
    for spike_rate, tau_u in tqdm(param_sweep):
        cfg('astro_params.tau_u', tau_u)

        astro = Astro.from_cfg(cfg['astro_params'], 1, cfg['sim']['dt'])
        pre_spikes = spiketrain.poisson(spike_rate, 1000)
        post_spikes = spiketrain.poisson(spike_rate, 1000)
        state = None
        timeline = {
            'eff': [],
            'ca': []
        }

        for z_pre, z_post in zip(pre_spikes[0], post_spikes[0]):
            eff, state = astro(state, z_pre=z_pre, z_post=z_post)
            timeline['eff'].append(
                torch.isclose(eff, torch.as_tensor(1.0)).float()
            )
            timeline['ca'].append(state['ca'])

        # print(
        #     "alpha: {}, spike rate: {}, any_effect: {}, max(u): {}".format(
        #         alpha,
        #         spike_rate,
        #         any(timeline['eff']),
        #         max(timeline['ca']),
        #     ))

        db.store({
            'spike_rate': spike_rate,
            'tau_u': tau_u,
            'timeline': timeline})

    # Graph
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
    for i, (spike_rate, alpha_db) in enumerate(db.group_by('spike_rate', sort=True).items()):
        for j, (alpha, elem_db) in enumerate(alpha_db.group_by('alpha', sort=True).items()):
            tl = elem_db[0]['timeline']
            eff = torch.as_tensor(tl['eff'])
            
            heat_img[j, i] = (1.0 - eff.mean()) * 1000.0
            ax.text(
                i, j,
                "{:1.2f}".format(float(1000.0 - heat_img[j, i])),
                ha="center", va="center", color="w")

    ax.imshow(heat_img)
    fig.tight_layout()
    fig.savefig("u_thr_heatmap_spike-rate_tau_u.svg")


def sim_heatmap_dt_tau(cfg):
    tau_range = torch.logspace(0.5, 3, 50)
    delta_t_range = torch.linspace(0, 20e-3, 21)

    param_sweep = VSweep(values=tau_range)
    param_sweep = param_sweep.foreach(delta_t_range)

    db = ExpStorage()

    # Simulate
    for tau, delta_t in tqdm(param_sweep):

        # Create astro with modified config
        cfg('astro_params.tau_ip3', tau)
        cfg('astro_params.tau_kp', tau)

        astro = Astro.from_cfg(cfg['astro_params'], 1, cfg['sim']['dt'])
        state = None

        pulse_pair_spikes = spiketrain.pre_post_pair(delta_t, cfg['sim']['dt'])
        delta_u = 0

        for z_pre, z_post in pulse_pair_spikes[0]:
            eff, state = astro(state, z_pre=z_pre, z_post=z_post)

            if int(z_post) == 1:
                delta_u = state['ca']
                break

        db.store({
            'tau': tau,
            'delta_t': delta_t,
            'delta_u': delta_u,
        })
        
        astro = Astro.from_cfg(cfg['astro_params'], 1, cfg['sim']['dt'])
        state = None

        pulse_pair_spikes = spiketrain.pre_post_pair(delta_t, cfg['sim']['dt'])
        delta_u = 0

        for z_pre, z_post in pulse_pair_spikes[0]:
            eff, state = astro(state, z_pre=z_pre, z_post=z_post)

            if int(z_post) == 1:
                delta_u = state['ca']
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


def _main(args):
    cfg = config.Config(args.config)
    cfg['astro_params'] = cfg['classic_stdp']
    sim_sweep_pre_i(cfg)

    cfg = config.Config(args.config)
    cfg['astro_params'] = cfg['classic_stdp']
    cfg['astro_params']['u_th'] = 100.0
    sim_sweep_u(cfg)

    cfg = config.Config(args.config)
    cfg['astro_params'] = cfg['classic_stdp']
    sim_heatmap_alpha_u_thr_events(cfg)

    cfg = config.Config(args.config)
    cfg['astro_params'] = cfg['classic_stdp']
    sim_heatmap_tau_u_thr_events(cfg)
    
    # cfg = config.Config(args.config)
    # cfg['astro_params'] = cfg['classic_stdp']
    # sim_heatmap_dt_tau(cfg)


if __name__ == '__main__':
    args = _parse_args()
    _main(args)
