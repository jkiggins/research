import numpy as np

plt_round_bbox = dict(
    boxstyle="round",
    ec=(1., 0.5, 0.5),
    fc=(1., 0.8, 0.8))

def plot_events(ax, events, labels=[], colors=None, offset=0):
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


def astro_params_text(cfg, exclude=None):
    text = "U Tau :{:4.2f}\n".format(cfg['tau_u'])

    if np.isclose(cfg['tau_i_pre'], cfg['tau_i_post']):
        text += "Pre/Post Tau: {:4.2f}\n".format(cfg['tau_i_pre'])
    else:
        text += "Pre Tau: {:4.2f}\n".format(cfg['tau_i_pre'])
        text += "Post Tau: {:4.2f}\n".format(cfg['tau_i_post'])

    if np.isclose(cfg['alpha_pre'], cfg['alpha_post']):
        text += "Pre/Post Alpha: {:4.2f}\n".format(cfg['alpha_pre'])
    else:
        text += "Pre Tau: {:4.2f}\n".format(cfg['alpha_pre'])
        text += "Post Tau: {:4.2f}\n".format(cfg['alpha_post'])

    return text
        
