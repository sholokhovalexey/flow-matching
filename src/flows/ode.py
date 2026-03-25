"""Borrowed from https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/ode_solver.py"""

import math
import torch
from torch import Tensor
from torchdiffeq import odeint
from ..utils.torch_utils import gradient


class ODESolver:
    def __init__(
        self,
        velocity_model,
        odeint_kwargs=dict(method="euler", atol=1e-5, rtol=1e-5),
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.odeint_kwargs = odeint_kwargs

    def sample(self, x_init, **kwargs):
        kwargs_upd = self.odeint_kwargs.copy()
        kwargs_upd.update(kwargs)
        return sample(self.velocity_model, x_init, **kwargs_upd)

    def compute_likelihood(self, x, **kwargs):
        kwargs_upd = self.odeint_kwargs.copy()
        kwargs_upd.update(kwargs)
        return compute_likelihood(self.velocity_model, x, log_p_base, **kwargs_upd)


def sample(
    velocity_model,
    x_init,
    step_size,
    method="euler",
    atol=1e-5,
    rtol=1e-5,
    time_grid=torch.tensor([0.0, 1.0]),
    return_intermediates=False,
    enable_grad=False,
    **model_extras,
):
    time_grid = time_grid.to(x_init.device)

    def ode_func(t, x):
        return velocity_model(x=x, t=t, **model_extras)

    ode_opts = {"step_size": step_size} if step_size is not None else {}

    with torch.set_grad_enabled(enable_grad):
        # Approximate ODE solution with numerical ODE solver
        sol = odeint(
            ode_func,
            x_init,
            time_grid,
            method=method,
            options=ode_opts,
            atol=atol,
            rtol=rtol,
        )

    if return_intermediates:
        return sol
    else:
        return sol[-1]


def compute_likelihood(
    velocity_model,
    x_1: Tensor,
    log_p0,
    step_size,
    method="euler",
    atol=1e-5,
    rtol=1e-5,
    time_grid=torch.tensor([1.0, 0.0]),
    return_intermediates=False,
    exact_divergence=False,
    enable_grad=False,
    **model_extras,
):

    assert (
        time_grid[0] == 1.0 and time_grid[-1] == 0.0
    ), f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}"

    # Fix the random projection for the Hutchinson divergence estimator
    if not exact_divergence:
        z = (torch.randn_like(x_1).to(x_1.device) < 0) * 2.0 - 1.0

    def ode_func(x, t):
        return velocity_model(x=x, t=t, **model_extras)

    def dynamics_func(t, states):
        xt = states[0]
        with torch.set_grad_enabled(True):
            xt.requires_grad_()
            ut = ode_func(xt, t)

            if exact_divergence:
                # Compute exact divergence
                div = 0
                for i in range(ut.flatten(1).shape[1]):
                    g = gradient(ut[:, i], xt, create_graph=True)[:, i]
                    if not enable_grad:
                        g = g.detach()
                    div += g
            else:
                # Compute Hutchinson divergence estimator E[z^T D_x(ut) z]
                ut_dot_z = torch.einsum(
                    "ij,ij->i", ut.flatten(start_dim=1), z.flatten(start_dim=1)
                )
                grad_ut_dot_z = gradient(ut_dot_z, xt, create_graph=enable_grad)
                div = torch.einsum(
                    "ij,ij->i",
                    grad_ut_dot_z.flatten(start_dim=1),
                    z.flatten(start_dim=1),
                )

        if not enable_grad:
            ut = ut.detach()
            div = div.detach()
        return ut, div

    y_init = (x_1, torch.zeros(x_1.shape[0], device=x_1.device))
    ode_opts = {"step_size": step_size} if step_size is not None else {}

    with torch.set_grad_enabled(enable_grad):
        sol, log_det = odeint(
            dynamics_func,
            y_init,
            time_grid,
            method=method,
            options=ode_opts,
            atol=atol,
            rtol=rtol,
        )

    x_source = sol[-1]
    source_log_p = log_p0(x_source)

    if return_intermediates:
        return sol, source_log_p + log_det[-1]
    else:
        return sol[-1], source_log_p + log_det[-1]


def log_p_base(x):
    PI = torch.tensor(math.pi)
    log_p = -0.5 * torch.log(2.0 * PI) - 0.5 * x.flatten(1, -1) ** 2
    return torch.sum(log_p, dim=1)
