from ..functional.lif import lif_step

class LIFNeuron:
    def __init__(self, params, dt):
        self.params = params
        self.dt = dt

    @classmethod
    def from_cfg(cfg, dt):
        return LIFNeuron(cfg, dt)

    def __call__(self, z, state):
        if state is None:
            state = {'v': torch.as_tensor(0.0), 'i': torch.as_tensor(0.0)}

        z, state = lif_step(z, state, self.params, self.dt)
        return z, state        
