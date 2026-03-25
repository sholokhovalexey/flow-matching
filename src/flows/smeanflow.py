import torch
from torch import nn
from torch.nn import functional as F

from .utils import reshape_like
from .meanflow import MeanFlow


class SplitMeanFlow(MeanFlow):

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
        t_equals_r_indices = torch.where(t_equals_r_mask)

        # Sample split interpolation weight
        lmbda = torch.rand_like(t)

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

        # Intermediate step for interval-splitting consistency
        s = (1 - lmbda) * t + lmbda * r

        # Target average velocity via split (no JVP)
        with torch.no_grad():
            u2 = self.net(z, t, s, cond=cond)
            x_s = z - reshape_like(t - s, u2) * u2
            u1 = self.net(x_s, s, r, cond=cond)
            u_tgt = reshape_like(1 - lmbda, u1) * u1 + reshape_like(lmbda, u2) * u2
            u_tgt[t_equals_r_indices] = v_g[t_equals_r_indices]

        # Predicted average velocity
        u = self.net(z, t, r, cond=cond)

        # Compute loss
        loss = F.mse_loss(u, u_tgt.detach())

        return {"loss": loss}
