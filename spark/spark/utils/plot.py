import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

spike_colors = {
    'post': 'tab:orange',
    'pre': 'tab:orange'
}

dw_colors = ['tab:green', 'tab:purple', 'tab:red']

astro_colors = {
    'k+': 'tab:orange',
    'ip3': 'tab:blue',
    'ca': 'tab:purple',
    'dser': 'tab:green',
    'serca': 'tab:red',
}

lif_colors = {
    'v_psp': 'tab:blue',
    'v_mem': 'tab:purple',
}

region_colors = {
    'and': 'tab:orange',
    'ltp': 'tab:red',
    'early-spike': 'tab:blue',
    'other-influence': 'tab:purple',
    'n/a': 'tab:green',
}

plt_round_box = dict(
    facecolor='tab:orange',
    alpha=0.5,
    edgecolor='tab:blue',
    boxstyle='round')

def rc_config(*args, **kwargs):
    matplotlib.rcParams.update(*args, **kwargs)

def _locate_ax(axes, loc, as_list=False):
    axs = axes
    for l in loc:
        axs = axs[l]

    if as_list and (not isinstance(axs, list)):
        axs = [axs]

    return axs
    

def gs(*args, **kwargs):
    return GridSpec(*args, **kwargs)

def gen_axes(*gridspecs, fig=None, axes=None, figsize=None):
    if fig is None and axes is None:
        fig = plt.Figure(figsize=figsize)
        axes = {}

    for ax_name, gs in gridspecs:
        ax = fig.add_subplot(gs)
        ax = Axis(ax)

        if not (ax_name in axes):
            axes[ax_name] = ax
        else:
            if not isinstance(axes[ax_name], list):
                axes[ax_name] = [axes[ax_name]]
            axes[ax_name].append(ax)

    return fig, axes


def xlim(axes, xlim):
    for k, axs in axes.items():
        if type(axs) == list:
            for ax in axs:
                ax.set_xlim(xlim)
        else:
            axs.set_xlim(xlim)


def plot_spikes(axes, loc, z_pre, z_post):
    # Locate axis for plotting, expecting one axis
    ax = _locate_ax(axes, loc)

    nsyns = z_pre.shape[-1]

    syn_range = list(reversed(range(nsyns)))

    events = []
    events = events + [z_pre[:, i] for i in syn_range]
    legend = ['pre-s{}'.format(i) for i in syn_range]
    events = events + [z_post[:,0]]
    legend += ['post']

    ax.set_title("Spikes")
    ax.set_ylabel("Time (ms)")

    ax.plot_events(
        events,
        labels=legend,
        colors=['tab:blue']*nsyns + ['tab:orange']
    )

    ax.set_xlim(-10, z_pre.shape[0] + 10)


def plot_dw(axes, loc, dw):
    ax = _locate_ax(axes, loc)
    
    nsyns = dw.shape[-1]

    for i in range(nsyns):
        colors_i = i % len(dw_colors)
        ax.plot(dw[:, i].tolist(), color=dw_colors[colors_i])

    ax.set_title("$\Delta$W over Time")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("$\Delta$W")

    ax.set_xlim(-10, dw.shape[0] + 10)


def plot_w(axes, loc, w):
    ax = _locate_ax(axes, loc)
    
    nsyns = w.shape[-1]

    for i in range(nsyns):
        colors_i = i % len(dw_colors)
        ax.plot(w[:, i].tolist(), color=dw_colors[colors_i])

    ax.set_title("Synaptic Weight over Time")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Synaptic Weight")

    ax.set_xlim(-10, w.shape[0] + 10)
    

def plot_lif(axes, loc, v_psp, v_mem):
    axs = _locate_ax(axes, loc, as_list=True)

    n_synapse = len(axs)
    
    assert n_synapse == v_psp.shape[-1]

    for i, ax in enumerate(axs):
        ax.set_title("LIF Neuron Voltages: S{}".format(i))
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage")
        ax.plot(v_psp[:, i], color=lif_colors['v_psp'], label='Synaptic Voltage')
        ax.plot(v_mem[:, i], color=lif_colors['v_mem'], label='Membrane Voltage')

        ax.legend()

        ax.set_xlim(-10, v_mem.shape[0] + 10)
    
    
def plot_astro(axes, loc, ip3, kp, ca, dser, serca, no_legend=False):
    # Locate axes for plotting, expecting n_synapse axes
    axs = _locate_ax(axes, loc, as_list=True)

    n_synapse = len(axs)

    traces = [ip3, kp, ca, dser, serca]
    duration = None
    for t in traces:
        if not (t is None):
            duration = t.shape[0]
            assert n_synapse == t.shape[-1]

    for i, ax in enumerate(axs):
        ax.set_title("Local Astrocyte State: S{}".format(i))
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Concentrations")

        if not (ip3 is None): ax.plot(ip3[:, i], color=astro_colors['ip3'], label='ip3')
        if not (kp is None): ax.plot(kp[:, i], color=astro_colors['k+'], label='k+')
        if not (ca is None): ax.plot(ca[:, i], color=astro_colors['ca'], label='ca')
        if not (dser is None): ax.plot(dser[:, i], color=astro_colors['dser'], label='D-Serine')
        if not (serca is None): ax.plot(serca[:, i], color=astro_colors['serca'], label='Serca')

        ax.set_xlim(-10, duration + 10)

        if not no_legend:
            ax.legend()


def plot_coupling_region(axes, loc, regions):
    # Locate one axis for plotting
    ax = _locate_ax(axes, loc)

    ax.set_title("Astrocyte AND Coupling Region")
    ax.set_xlabel("Time (ms)")

    reg_to_offset = {}

    x_prev = 0

    for reg_name, duration in regions:
        if not (reg_name in reg_to_offset):
            reg_to_offset[reg_name] = len(reg_to_offset)
        color = region_colors[reg_name]

        ax.plot(
            [x_prev, x_prev + duration],
            [reg_to_offset[reg_name]]*2,
            '-o', color=color)

        x_prev += duration

    ax.set_xlim(-10, x_prev + 10)

    ax.set_yticks(list(reg_to_offset.values()), list(reg_to_offset.keys()))


def plot_mismatch_bar(axes, loc, bar):
    ax = _locate_ax(axes, loc)

    ax.set_title("Fractions of Mismatches Belonging to Each Category")
    ax.set_ylabel("Fraction of total Mismatches")

    ax.bar(
        list(bar.keys()),
        list(bar.values())
    )


def plot_err(axes, loc, err):
    ax = _locate_ax(axes, loc)

    ax.set_title("Error Rate During Learning")
    ax.set_ylabel("Error rate")
    ax.set_xlabel("Learning Iterations")

    ax.plot(err)


class Axis:
    def __init__(self, ax, offset=False):
        self.ax = ax
        self.offset = 0
        self.event_offset = 0

        self.offset_ticks = []
        self.offset_labels = []
        self.enable_offset = offset

        self.yticks = []

        self.plot_count = 0


    def _step_offset(self, y, no_step=False):
        y = torch.as_tensor(y)

        # If no_step == True, just apply existing offset
        # This is generally used to plot over the previous plot, with offset enabled for
        # non-overlay type plots
        if no_step:
            return y + self.last_offset
        
        if self.offset != 0:
            self.offset += max(-y.min(), 1.5)

        y_max = y.max()
        y = y + self.offset

        # Tick locations, and labels for offset graphs
        max_extent = y[(y-self.offset).abs().argmax()]
        self.offset_ticks += [float(self.offset), float(max_extent)]
        self.offset_labels += ["0.0", "{:4.2f}".format(float(max_extent-self.offset))]

        self.last_offset = self.offset
        # Add the max amount above the y=0 point, so the next graph clears that
        self.offset += max(y_max, 1.5)

        return y


    def divider(self):
        self.offset += 3

    def event_divider(self):
        self.event_offset += 3


    # Pass-throug methods
    def legend(self, *args, **kwargs):
        self.ax.legend(*args, **kwargs)

    def set_title(self, *args, **kwargs):
        self.ax.set_title(*args, **kwargs)

    def set_xlabel(self, *args, **kwargs):
        self.ax.set_xlabel(*args, **kwargs)

    def set_ylabel(self, *args, **kwargs):
        self.ax.set_ylabel(*args, **kwargs)

    def set_xlim(self, *args, **kwargs):
        self.ax.set_xlim(*args, **kwargs)

    def set_xticks(self, *args, **kwargs):
        self.ax.set_xticks(*args, **kwargs)

    def set_yticks(self, *args, **kwargs):
        self.ax.set_yticks(*args, **kwargs)


    def lif_ax_yticks(self, ticks, append=False):
        is_list_of_pairs = len(ticks) > 0 and hasattr(ticks[0], '__len__') and len(ticks[0]) == 2
        is_pair = not is_list_of_pairs and len(ticks) == 2

        if is_pair:
            ticks = [ticks]

        if append:
            self.yticks += ticks
        else:
            self.yticks = ticks

        ytick_loc, ytick_labels = zip(*self.yticks)
        self.ax.set_yticks(ytick_loc, labels=ytick_labels)


    def bar(self, *args, **kwargs):
        return self.ax.bar(*args, **kwargs)


    def plot(self, *args, **kwargs):
        # Strip out some args from kwargs
        markup = False
        
        if 'markup' in kwargs:
            markup = kwargs['markup']
            del kwargs['markup']

        if self.enable_offset:
            # Apply offset to y axis, unless this is a decorative or 'markup' plot
            args = list(args)
            if len(args) > 1 and not isinstance(args[1], str):
                args[1] = self._step_offset(args[1], no_step=markup)
            else:
                args[0] = self._step_offset(args[0], no_step=markup)
            args = tuple(args)
            self.ax.set_yticks(self.offset_ticks)
            self.ax.set_yticklabels(self.offset_labels)

        line = self.ax.plot(*args, **kwargs)

        self.plot_count += 1

        return line


    def plot_events(self, events, labels=None, colors=None):
        event_idxs = []
        max_x = 0
        for z in events:
            if type(z) != np.ndarray:
                z = np.array(z)

            event_idx = np.where(z > 0)[0]
            max_x = max(len(z), max_x)
            event_idxs.append(event_idx.tolist())

        line_offsets = [i + self.event_offset for i in range(len(event_idxs))]

        # Y tick for each group of spikes
        for o, l in zip(line_offsets, labels):
            ytick = (o, l)
            self.lif_ax_yticks(ytick, append=True)

        self.ax.eventplot(event_idxs,
                     lineoffsets=line_offsets,
                     linelengths=0.5,
                     colors=colors)
        # self.ax.set_xlim((0, max_x))

        self.event_offset = line_offsets[-1] + 1
        self.plot_count += 1



## Legacy Functions ##
def plot_events(ax, events, colors=None, offset=0):
    event_idxs = []
    max_x = 0
    for z in events:
        if type(z) != np.ndarray:
            z = np.array(z)
            
        event_idx = np.where(z > 0)[0]
        max_x = max(len(z), max_x)
        event_idxs.append(event_idx.tolist())

    line_offsets = [i + offset for i in range(len(event_idxs))]

    ax.eventplot(event_idxs,
                 lineoffsets=line_offsets,
                 linelengths=0.5,
                 colors=colors)
    ax.set_xlim((0, max_x))

    return ax
