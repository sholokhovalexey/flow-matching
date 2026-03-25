import math

import torch
from torch import nn
from torch.nn import functional as F

from .utils import reshape_like
from .base import CFM


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class VariationalFlow(CFM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss(self, batch, **kwargs):
        x1 = batch[0]
        cond = batch[1]
        batch_size = x1.shape[0]

        # Sample time step
        t = self.sample_t(batch_size, x1.device)

        # Sample noise
        x0 = self.sample_noise_like(x1)

        # Sample coupling
        x0, x1, cond = self.sample_couplings(x0, x1, cond)

        # Compute intermediate sample
        x_t = self.get_x_t(x0, x1, t)

        # Cond dropout
        cond = self.cond_drop(cond)

        # Predicted posterior mean (endpoint x1 given x_t, t)
        mu = self.net(x_t, t, cond=cond)

        mu = torch.flatten(mu, start_dim=1, end_dim=-1)
        x1 = torch.flatten(x1, start_dim=1, end_dim=-1)

        # Time-dependent variance
        sigma = (1 - (1 - 0.01) * t) ** 2

        # Compute loss (weighted MSE approximating -log q(x1|x_t))
        weight = 1.0 / (reshape_like(sigma, x1) + 1e-3)
        loss = F.mse_loss(mu, x1, weight=weight.expand_as(x1))
        return {"loss": loss}

    
