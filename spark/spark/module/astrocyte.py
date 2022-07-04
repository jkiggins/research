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
    astro_step_reward_effect,
    astro_step_activity,
    astro_step_effect_weight_prop,
    astro_track_activity
)

class Astro:
    def __init__(self, params, synapse, dt):
        self.params = params
        self.dt = dt
        self.num_process = synapse

    @classmethod
    def from_cfg(cls, cfg, synapse, dt):
        return Astro(cfg, synapse, dt)

    def init_state_if_none(self, state):
        if state is None:
            state = {
                'u': torch.zeros(self.num_process),
                'i_pre': torch.zeros(self.num_process),
                'i_post': torch.zeros(self.num_process),
            }

        return state


    def _astro_step(self, state, z_pre, z_post, reward=None):

        state = self.init_state_if_none(state)

        # Decay u
        state = astro_step_decay(state, self.params, self.dt)

        # Processes incoming spikes
        state = astro_step_z_pre(z_pre, state, self.params, self.dt)
        state = astro_step_z_post(z_post, state, self.params, self.dt)

        if self.params['weight_update'] == 'ip3_k+_fall':
            state = astro_track_activity(state, self.params)
            # print("act_gt_thr: ", state['act_gt_thr'], end=' ')

        # Step changes to Ca2+
        # Ca is incremented by the product of ip3 and k+
        if self.params['u_step_params']['mode'] == 'u_prod':
            state = astro_step_u_prod(state)
        # Ca is incremended by the product of ip3 and k+, with a sign depending on which is larger
        elif self.params['u_step_params']['mode'] == 'u_ordered_prod':
            state = astro_step_u_ordered_prod(state, self.params)
        # Ca is incremented according to an STDP-like rule
        elif self.params['u_step_params']['mode'] == 'stdp':
            state = astro_step_u_stdp(state, self.params, z_pre=z_pre, z_post=z_post)

        # Effect weight
        # Update weight when Ca > thr, by a fixed factor for LTD/LTP
        if self.params['weight_update'] == 'thr':
            state, u_spike = astro_step_thr(state, self.params)  # Apply thr to u
            eff = astro_step_effect_weight(u_spike, self.params)  # Get effect based on u exceeding thr

        elif self.params['weight_update'] == 'prop':
            ca = state['u']
            u_spike = torch.ones_like(ca)
            # print("ca: {}, ".format(ca), end='')
            state, eff = astro_step_effect_weight_prop(torch.ones_like(ca), state, self.params)
            # print("u_spike: {}, eff: {}".format(u_spike, eff))

        elif self.params['weight_update'] == 'ip3_k+_fall':
            state, u_spike = astro_step_activity(state, self.params)  # Detect falling edge on ip3/k+
            state, eff = astro_step_effect_weight_prop(u_spike, state, self.params)  # Get effect based on u exceeding thr

        assert torch.all(torch.abs(state['u']) <= torch.as_tensor(2.5))
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


    def __call__(self, state, z_pre=None, z_post=None, reward=None):
        return self._astro_step(state, z_pre=z_pre, z_post=z_post, reward=reward)
