import math

import torch
from torch import nn
from torch.nn import functional as F

from .utils import reshape_like
from .meanflow import MeanFlow


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class AlphaFlow(MeanFlow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_alpha(self):
        if self.scheduler is not None and hasattr(self.scheduler, "alpha"):
            alpha = self.scheduler.alpha
        else:
            alpha = 1
        return alpha

    def loss(self, batch, **kwargs):
        x = batch[0]
        cond = batch[1]
        batch_size = x.shape[0]

        # Sample time interval (t, r)
        t, r = self.sample_t_r(batch_size, self.device)

        flow_ratio = self.sample_flow_ratio()
        t_equals_r_mask = torch.rand(batch_size, device=self.device) < flow_ratio
        r = torch.where(t_equals_r_mask, t, r)

        # Curriculum: alpha interpolates between trajectory FM (alpha=1) and MeanFlow (alpha=0)
        alpha = self.sample_alpha()
        s = alpha * r + (1 - alpha) * t

        # Sample noise
        e = self.sample_noise_like(x)

        # Sample coupling
        e, x, cond = self.sample_couplings(e, x, cond)

        # Compute intermediate sample
        z = self.get_x_t(e, x, t)

        # Compute conditional (instantaneous) velocity
        v = self.get_instantaneous_velocity(e, x, t, z)

        # Guided velocity
        v_g = self.guidance_fn(v, z, t, cond)

        # Cond dropout
        cond, v_g = self.cond_drop(v, v_g, cond)

        if alpha < 1e-3:
            # MeanFlow branch: target average velocity via JVP
            u, dudt = self.compute_jvp(z, t, r, v_g, cond)
            u_tgt = v_g - reshape_like(t - r, dudt) * dudt.detach()
        else:
            # Trajectory FM branch: mix instantaneous v and net on mid-point
            u = self.net(z, t, r, cond=cond)
            z2 = z - reshape_like(t - s, v) * v
            u_tgt = alpha * v + (1 - alpha) * self.net(z2, s, r, cond=cond)

        # Compute loss
        loss = F.mse_loss(u, u_tgt.detach())
        return {"loss": loss}
