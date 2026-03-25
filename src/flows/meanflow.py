import numpy as np
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from .utils import reshape_like
from .base import CFM


class MeanFlow(CFM):

    def __init__(
        self,
        net,
        # shape,
        # odeint_kwargs=dict(atol=1e-5, rtol=1e-5, method="euler"),
        cond_drop_prob=0.1,
        time_sampler="lognorm",
        cfg_scale=2.0,
        cfg_mix_scale=0.0,
        flow_ratio=0.5,
        jvp_api="funtorch",  # autograd, funtorch
        scheduler=None,
        # cond_type="identity",
        # num_classes=None,
        coupling_type="ot-exact",
    ):
        super().__init__(
            net,
            # shape,
            # odeint_kwargs,
            cond_drop_prob=cond_drop_prob,
            time_sampler=time_sampler,
            cfg_scale=cfg_scale,
            scheduler=scheduler,
            # cond_type=cond_type,
            # num_classes=num_classes,
            coupling_type=coupling_type,
        )
        self.cfg_mix_scale = cfg_mix_scale
        self.flow_ratio = flow_ratio
        self.jvp_api = jvp_api

    def _delta_gamma_now(self, p):
        gamma_start = 2.0
        gamma_end = 0.05
        warmup_fraction = 1.0
        frac = min(1.0, p / warmup_fraction)
        return gamma_start + frac * (gamma_end - gamma_start)

    def sample_r_given_t(self, t, gamma=1.0):
        t_eps = 0.03
        u = torch.rand_like(t) ** gamma
        r = t - u * (t - t_eps)
        return r

    def sample_t_r(self, batch_size, device):
        if self.time_sampler == "uniform":

            t = torch.rand((batch_size,), device=device)
            r = self.sample_r_given_t(t, 1.0)

        elif self.time_sampler == "power_v1":

            if self.scheduler is not None and hasattr(self.scheduler, "progress"):
                p = self.scheduler.progress

            t_pow = max(0.5, (1 - p) ** 0.5)
            t = torch.rand((batch_size,), device=device) ** t_pow
            gamma = self._delta_gamma_now(p)
            r = self.sample_r_given_t(t, gamma)

        elif self.time_sampler == "power_v2":

            if self.scheduler is not None and hasattr(self.scheduler, "progress"):
                p = self.scheduler.progress

            t_pow = 1.0 * (1 - p) + 2.0 * p
            t = 1 - torch.rand((batch_size,), device=device) ** t_pow
            gamma = self._delta_gamma_now(p)
            r = self.sample_r_given_t(t, gamma)

        elif self.time_sampler == "lognorm":
            mu, sigma = -0.4, 1.0

            normal_samples = (
                np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            )
            samples = 1 / (1 + np.exp(-normal_samples))

            t_np = np.maximum(samples[:, 0], samples[:, 1])
            r_np = np.minimum(samples[:, 0], samples[:, 1])

            t = torch.tensor(t_np, device=device)
            r = torch.tensor(r_np, device=device)

        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")

        return t, r

    def sample_flow_ratio(self):
        if self.scheduler is not None and hasattr(self.scheduler, "flow_ratio"):
            flow_ratio = self.scheduler.flow_ratio
        else:
            flow_ratio = self.flow_ratio
        return flow_ratio

    def get_x_t(self, noise, data, t):
        tt = reshape_like(t, data)
        return (1 - tt) * data + tt * noise

    def get_instantaneous_velocity(self, noise, data, t, x_t):
        v = noise - data
        return v

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

        # Guided velocity
        v_g = self.guidance_fn(v, z, t, cond)

        # Cond dropout
        cond, v_g = self.cond_drop(v, v_g, cond)

        # JVP: average velocity u and its time derivative dudt
        u, dudt = self.compute_jvp(z, t, r, v_g, cond)

        # Target average velocity (reduces to FM loss if t == r)
        u_tgt = v_g - reshape_like(t - r, dudt) * dudt.detach()

        # Compute loss
        loss = F.mse_loss(u, u_tgt.detach())
        return {"loss": loss}

    def compute_jvp(self, z, t, r, v, cond):

        model_partial = partial(self.net.forward, cond=cond)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.jvp_api == "funtorch":
            jvp_fn = torch.func.jvp
            create_graph = False
        elif self.jvp_api == "autograd":
            jvp_fn = torch.autograd.functional.jvp
            create_graph = True

        if create_graph:
            u, dudt = jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = jvp_fn(*jvp_args)
        return u, dudt

    def guidance_fn(self, v, x_t, t, cond):
        batch_size = v.shape[0]
        cond_null = self.get_null_condition(cond)
        if self.cfg_scale is not None:
            if isinstance(self.cfg_scale, float):
                cfg_scale = self.cfg_scale
            else:  # if .net is conditioned by cfg_scale, not implememnted for now
                assert len(self.cfg_scale) == 2
                cfg_min, cfg_max = self.cfg_scale
                rand = torch.rand(batch_size, device=v.device)
                cfg_scale = cfg_min + rand * (cfg_max - cfg_min)
                cfg_scale = reshape_like(cfg_scale, v)
            omega = cfg_scale
            kappa = self.cfg_mix_scale
            if self.cfg_mix_scale is not None:
                with torch.no_grad():
                    v_cond = self.net(
                        x_t, t, t, cond=cond
                    )  # TODO: merge and split batches
                    v_null = self.net(x_t, t, t, cond=cond_null)
                v_g = omega * v + kappa * v_cond + (1 - omega - kappa) * v_null  # eq. 21
            else:
                with torch.no_grad():
                    v_null = self.net(x_t, t, t, cond=cond_null)
                v_g = omega * v + (1 - omega) * v_null
        else:
            v_g = v
        return v_g # TODO: guidance interval

    def cond_drop(self, v, v_guided, cond):
        batch_size = v.shape[0]
        if self.cond_drop_prob > 0:
            cond_null = self.get_null_condition(cond)
            drop_mask = torch.rand(batch_size) < self.cond_drop_prob
            num_drop = torch.sum(drop_mask)
            if num_drop > 0:
                drop_indices = torch.where(drop_mask)
                cond[drop_indices] = cond_null[drop_indices]
                v_guided[drop_indices] = v[drop_indices]
        return cond, v_guided
