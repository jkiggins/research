import torch
from ..functional.astrocyte import (
    astro_step_decay,
    astro_step_z_pre,
    astro_step_z_post,
    astro_step_u_prod,
    astro_step_u_ordered_prod,
    astro_step_u_signal,
    astro_step_thr,
    astro_step_effect_weight,
)

class Astro:
    def __init__(self, params, dt):
        self.params = params
        self.dt = dt

    @classmethod
    def from_cfg(cls, cfg, dt):
        return Astro(cfg, dt)

    def init_state_if_none(self, state):
        if state is None:
            state = {
                'u': torch.as_tensor(0.0),
                'i_pre': torch.as_tensor(0.0),
                'i_post': torch.as_tensor(0.0),
            }

        return state
    

    def _stdp_mode_step(self, state, z_pre=None, z_post=None):
        state = self.init_state_if_none(state)

        state = astro_step_decay(state, self.params, self.dt)  # Decay of u
        if not (z_pre is None):
            # Update i_pre w/ input from z_pre
            state = astro_step_z_pre(z_pre, state, self.params, self.dt)
        if not (z_post is None):
            # Update i_post w/ input from z_pre
            state = astro_step_z_post(z_post, state, self.params, self.dt)

        # Update u, in this case it is the product of i_pre and i_post
        if self.params['u_update'] == 'stdp':
            state = astro_step_u_prod(state)
        elif self.params['u_update'] == 'stdp_ordered':
            state = astro_step_u_ordered_prod(state)

        state, u_spike = astro_step_thr(state, self.params)  # Apply thr to u
        eff = astro_step_effect_weight(u_spike, self.params)  # Get effect based on u exceeding thr

        return eff, state


    def _signal_respose_mode_step(self, state, z_pre=None, z_post=None):
        state = self.init_state_if_none(state)

        # Astro step
        state = astro_step_decay(state, self.params, self.dt)
        if not (z_pre is None):
            state = astro_step_z_pre(z_pre, state, self.params, self.dt)
        if not (z_post is None):
            state = astro_step_z_post(z_post, state, self.params, self.dt)
        state = astro_step_u_signal(state, self.params, self.dt)

        state, u_spike = astro_step_thr(state, self.params)
        

        return u_spike, state


    def __call__(self, state, z_pre=None, z_post=None):
        if self.params['mode'] == 'signal':
            return self._signal_respose_mode_step(state, z_pre=z_pre, z_post=z_post)
        elif self.params['mode'] == 'stdp':
            return self._stdp_mode_step(state, z_pre=z_pre, z_post=z_post)
