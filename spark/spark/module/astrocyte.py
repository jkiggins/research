import torch
from ..functional.astrocyte import (
    astro_step_decay,
    astro_step_z_pre,
    astro_step_z_post,
    astro_step_u_prod,
    astro_step_u_ordered_prod,
    astro_step_u_signal,
    astro_step_u_stdp,
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
    

    def _plastic_mode_step(self, state, z_pre=None, z_post=None):
        state = self.init_state_if_none(state)

        # Decay u
        state = astro_step_decay(state, self.params, self.dt)

        # Integrate pre or post-synaptic spikes
        if not (z_pre is None):
            state = astro_step_z_pre(z_pre, state, self.params, self.dt)
        if not (z_post is None):
            state = astro_step_z_post(z_post, state, self.params, self.dt)


        # Update u
        if self.params['u_step_params']['mode'] == 'u_prod':
            state = astro_step_u_prod(state)
        elif self.params['u_step_params']['mode'] == 'u_ordered_prod':
            state = astro_step_u_ordered_prod(state, self.params)
        elif self.params['u_step_params']['mode'] == 'stdp':
            state = astro_step_u_stdp(state, self.params, z_pre=z_pre, z_post=z_post)

        state, u_spike = astro_step_thr(state, self.params)  # Apply thr to u
        eff = astro_step_effect_weight(u_spike, self.params)  # Get effect based on u exceeding thr

        assert eff

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
        elif self.params['mode'] == 'plasticity':
            return self._plastic_mode_step(state, z_pre=z_pre, z_post=z_post)
