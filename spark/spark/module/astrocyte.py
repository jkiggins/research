import torch
from ..functional.astrocyte import (
    astro_step_decay,
    astro_step_z_pre,
    astro_step_z_post,
    astro_step_thr,
    astro_step_effect
)

class Astro:
    def __init__(self, params, dt):
        self.params = params
        self.dt = dt

    @classmethod
    def from_cfg(cls, cfg, dt):
        return Astro(cfg, dt)

    def __call__(self, state, z_pre=None, z_post=None):
        if state is None:
            state = {
                'u': torch.as_tensor(0.0),
                'i_pre': torch.as_tensor(0.0),
                'i_post': torch.as_tensor(0.0),
            }
                
        state = astro_step_decay(state, self.params, self.dt)

        if not (z_pre is None):
            state = astro_step_z_pre(z_pre, state, self.params, self.dt)

        if not (z_post is None):
            state = astro_step_z_post(z_post, state, self.params, self.dt)
            
        # state, u_spike = astro_step_thr(state, self.params)
        # eff = astro_step_effect(u_spike, state, self.params, self.dt)

        # return eff

        return None, state
