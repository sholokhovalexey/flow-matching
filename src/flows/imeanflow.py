import torch
from torch import nn
from torch.nn import functional as F

from .utils import reshape_like
from .meanflow import MeanFlow


class ImprovedMeanFlow(MeanFlow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss(self, batch, **kwargs):
        x = batch[0]
        cond = batch[1]
        batch_size = x.shape[0]

        # Sample time interval (t, r)
        t, r = self.sample_t_r(batch_size, self.device)

        flow_ratio = self.sample_flow_ratio()
        t_equals_r_mask = torch.rand(batch_size, device=self.device) < flow_ratio
        r = torch.where(t_equals_r_mask, t, r)

        # Sample noise
        e = self.sample_noise_like(x)

        # Sample coupling
        e, x, cond = self.sample_couplings(e, x, cond)

        # Compute intermediate sample
        z = self.get_x_t(e, x, t)

        # Compute conditional (instantaneous) velocity
        v = self.get_instantaneous_velocity(e, x, t, z)

        # Boundary-condition velocity
        v_cond = self.net(z, t, t, cond=cond)

        # Guided velocity
        v_g = self.guidance_fn(v, v_cond.detach(), z, t, cond)

        # Cond dropout
        cond, v_g = self.cond_drop(v, v_g, cond)

        # JVP: average velocity u and time derivative dudt
        u, dudt = self.compute_jvp(z, t, r, v_cond, cond)

        # Predicted instantaneous velocity from mean-flow identity (stopgrad on dudt)
        V = u + reshape_like(t - r, dudt) * dudt.detach()
        v_tgt = v_g

        # Compute loss
        loss = F.mse_loss(V, v_tgt.detach())
        return {"loss": loss}

    def guidance_fn(self, v, v_cond, x_t, t, cond):
        cond_null = self.get_null_condition(cond)
        w = self.cfg_scale
        kappa = self.cfg_mix_scale
        if self.cfg_scale is not None:
            if self.cfg_mix_scale is not None:
                with torch.no_grad():
                    v_null = self.net(x_t, t, t, cond=cond_null)
                v_g = w * v + kappa * v_cond + (1 - w - kappa) * v_null  # eq. 21
            else:
                with torch.no_grad():
                    v_null = self.net(x_t, t, t, cond=cond_null)
                v_g = w * v + (1 - w) * v_null
        else:
            v_g = v
        return v_g
