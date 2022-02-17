import numpy as np

def plot_events(ax, events):
    event_idxs = []
    max_x = 0
    for z in events:
        if type(z) != np.ndarray:
            z = np.array(z)
            
        event_idx = np.where(z > 0)[0]
        max_x = max(len(z), max_x)
        event_idxs.append(event_idx.tolist())
    
    ax.eventplot(event_idxs, lineoffsets=list(range(len(event_idxs))), linelengths=0.5)
    ax.set_xlim((0, max_x))

    return ax
