import torch

def gradient(
    output,
    x,
    grad_outputs=None,
    create_graph=False,
):
    """
    Compute the gradient of the inner product of output and grad_outputs w.r.t :math:`x`.

    Args:
        output (Tensor): [N, D] Output of the function.
        x (Tensor): [N, d_1, d_2, ... ] input
        grad_outputs (Optional[Tensor]): [N, D] Gradient of outputs, if `None`,
            then will use a tensor of ones
        create_graph (bool): If True, graph of the derivative will be constructed, allowing
            to compute higher order derivative products. Defaults to False.
    Returns:
        Tensor: [N, d_1, d_2, ... ]. the gradient w.r.t x.
    """

    if grad_outputs is None:
        grad_outputs = torch.ones_like(output).detach()
    grad = torch.autograd.grad(
        output, x, grad_outputs=grad_outputs, create_graph=create_graph
    )[0]
    return grad