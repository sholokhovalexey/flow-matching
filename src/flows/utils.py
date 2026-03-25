import torch


def reshape_like(t, x):
    """pad t like x
    """
    batch_size = x.shape[0]
    if isinstance(t, (float, int)):
        return t
    # assert t.dim() == 1
    if t.dim() == 0:
        t = t.repeat(batch_size)
    if t.dim() == 1:
        dim_diff = x.dim() - t.dim()
        ones = ([1] * dim_diff)
        return t.reshape(batch_size, *ones)
    else:
        raise ValueError()
    