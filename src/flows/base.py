import torch
from torch import nn
from torch.nn import functional as F

from .utils import reshape_like
from .optimal_transport import OTPlanSampler


class CFM(nn.Module):

    def __init__(
        self,
        net,
        cond_drop_prob=0.1,
        time_sampler="uniform",
        cfg_scale=2.0,
        scheduler=None,
        coupling_type="ot-exact",
        contrastive_weight=None,
    ):
        super().__init__()
        self.net = net
        self.cond_drop_prob = cond_drop_prob
        self.time_sampler = time_sampler
        self.cfg_scale = cfg_scale
        self.scheduler = scheduler
        self.coupling_type = coupling_type
        if self.coupling_type.startswith(("ot", "OT")):
            method = coupling_type.split("-")[1]
            assert method in ["exact", "sinkhorn", "unbalanced", "partial"]
            self.ot_sampler = OTPlanSampler(method=method)
        self.contrastive_weight = contrastive_weight

    @property
    def device(self):
        return next(self.parameters()).device

    def get_null_condition(self, cond):
        return self.net.get_null_condition(cond.shape[0])

    def forward(self, batch, **kwargs):
        loss = self.loss(batch, **kwargs)
        if self.training and self.scheduler is not None:
            self.scheduler.step()
        return loss

    def sample_t(self, batch_size, device):
        if self.time_sampler == "uniform":
            t = torch.rand((batch_size,), device=device)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        return t

    def sample_couplings(self, x0, x1, *args):
        if self.coupling_type == "random":
            args_out = args
        elif self.coupling_type.startswith(("ot", "OT")):
            pi = self.ot_sampler.get_map(x0, x1)
            i, j = self.ot_sampler.sample_map(pi, x0.shape[0], replace=True)
            x0 = x0[i]
            x1 = x1[j]
            args_out = [None if x is None else x[j] for x in args]
        else:
            raise ValueError()
        return x0, x1, *args_out

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def get_x_t(self, noise, data, t):
        tt = reshape_like(t, data)
        return (1 - tt) * noise + tt * data

    def get_instantaneous_velocity(self, noise, data, t, x_t):
        v = data - noise
        return v

    def loss(self, batch, x_neg=None, **kwargs):
        x1 = batch[0]
        cond = batch[1]
        batch_size = x1.shape[0]
        # Sample time step
        t = self.sample_t(batch_size, x1.device)

        # Sample noise
        x0 = self.sample_noise_like(x1)

        # Sample coupling
        x0, x1, cond, x_neg = self.sample_couplings(x0, x1, cond, x_neg)

        # Compute intermediate sample
        x_t = self.get_x_t(x0, x1, t)

        # Compute conditional velocity
        v = self.get_instantaneous_velocity(x0, x1, t, x_t)

        # Conditional negative velocity for contrastive learning (optional)
        if self.contrastive_weight is not None:
            v_neg = self.get_instantaneous_velocity(x0, x_neg, t, x_t)

        # Cond dropout
        cond = self.cond_drop(cond)

        # Predicted velocity
        v_pred = self.net(x_t, t, cond=cond)

        # Compute loss
        v_tgt = v
        loss = F.mse_loss(v_pred, v_tgt.detach())
        if self.contrastive_weight is not None:
            loss += (-1) * self.contrastive_weight * F.mse_loss(v_pred, v_neg)
        return {"loss": loss}

    def cond_drop(self, cond):
        batch_size = cond.shape[0]
        if self.cond_drop_prob > 0:
            cond = cond.clone()
            cond_null = self.get_null_condition(cond)
            drop_mask = torch.rand(batch_size) < self.cond_drop_prob
            num_drop = torch.sum(drop_mask)
            if num_drop > 0:
                drop_indices = torch.where(drop_mask)
                cond[drop_indices] = cond_null[drop_indices]
        return cond
