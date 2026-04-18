import torch
from .base import CFM
from .meanflow import MeanFlow
from .imeanflow import ImprovedMeanFlow
from .smeanflow import SplitMeanFlow
from .alphaflow import AlphaFlow
from .variational import VariationalFlow
from .utils import reshape_like
from ..backbones import CFGScaledModel


def construct_sampler(
    flow_model,
    solver_cls,
    cond,
    num_steps=32,
    cfg_scale=1.0,
    seed=42,
    device="cuda",
    initial_noise=None,
):
    """Build a sampling closure that generates samples via ODE integration with CFG.

    Wraps the flow model's network with classifier-free guidance and configures
    the solver (velocity field and time grid) according to the flow type.

    Args:
        flow_model: A flow-matching model (e.g. MeanFlow, VariationalFlow, AlphaFlow)
            with a .net attribute and compatible forward signature.
        solver_cls: ODE solver class that takes a callable (t, x) -> velocity
            and provides a .sample(x, step_size=..., time_grid=..., method=...) method.
        cond: Conditioning tensor.
        num_steps: Number of integration steps for the ODE solver.
        cfg_scale: Classifier-free guidance scale.
        seed: Random seed for the initial noise.
        device: Device for tensors.

    Returns:
        A callable sample() with no arguments that returns a tensor of generated
        samples of shape (batch_size, *net.shape).
    """
    rng = torch.Generator(device="cuda").manual_seed(seed)
    batch_size = cond.shape[0]

    net = flow_model.net

    cond_null = net.get_null_condition(batch_size)
    cfg_scaled_model = CFGScaledModel(net, cond, cond_null, cfg_scale=cfg_scale)

    x = torch.randn((batch_size, *net.shape), device=device, generator=rng)
    if initial_noise is not None:
        x = initial_noise
    step_size = 1 / num_steps

    if isinstance(flow_model, MeanFlow):

        def u_fn(t, x):
            r = t - step_size
            r = torch.clamp(r, min=0)
            return cfg_scaled_model(x, t, r)

        solver = solver_cls(u_fn)

        def sample():
            x = (
                initial_noise
                if initial_noise is not None
                else torch.randn((batch_size, *net.shape), device=device, generator=rng)
            )
            x_gen = solver.sample(
                x,
                step_size=step_size,
                method="euler",
                time_grid=torch.tensor([1.0, 0.0]),
            )
            return x_gen

    elif isinstance(flow_model, VariationalFlow):

        def v_fn(t, x):
            mu = cfg_scaled_model(x, t)
            v = (mu - (1 - 0.01) * x) / (1 - (1 - 0.01) * reshape_like(t, x))
            return v

        solver = solver_cls(v_fn)

        def sample():
            x = (
                initial_noise
                if initial_noise is not None
                else torch.randn((batch_size, *net.shape), device=device, generator=rng)
            )
            x_gen = solver.sample(
                x, step_size=step_size, time_grid=torch.tensor([0.0, 1.0])
            )
            return x_gen

    else:

        def v_fn(t, x):
            return cfg_scaled_model(x, t)

        solver = solver_cls(v_fn)

        def sample():
            x = (
                initial_noise
                if initial_noise is not None
                else torch.randn((batch_size, *net.shape), device=device, generator=rng)
            )
            x_gen = solver.sample(
                x, step_size=step_size, time_grid=torch.tensor([0.0, 1.0])
            )
            return x_gen

    return sample
