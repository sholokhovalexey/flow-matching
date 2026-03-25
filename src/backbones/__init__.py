import torch


class CFGScaledModel(torch.nn.Module):
    """Wraps a conditional model to apply classifier-free guidance (CFG)."""

    def __init__(self, model, cond, cond_null=None, cfg_scale=1.0):
        """Initialize the CFG-scaled wrapper.

        Args:
            model: The underlying conditional model.
            cond: Conditioning tensor passed when computing the guided output.
            cond_null: Conditioning tensor for the "unconditional" case.
            cfg_scale: Guidance strength.
        """
        super().__init__()
        self.model = model
        self.cond = cond
        self.cond_null = cond_null
        self.cfg_scale = cfg_scale

    def forward(self, x, t, r=None, **kwargs):
        """Forward with classifier-free guidance.

        Args:
            x: Input tensor
            t: Time step.
            r: Optional second time argument for MeanFlow-style nets.

        Returns:
            Guided prediction tensor: cfg_scale * v_cond + (1 - cfg_scale) * v_null.
        """
        v_cond = self.model(x, t, r=r, cond=self.cond, **kwargs)
        if self.cfg_scale != 1:
            v_null = self.model(x, t, r=r, cond=self.cond_null, **kwargs)
            return self.cfg_scale * v_cond + (1 - self.cfg_scale) * v_null
        else:
            return v_cond


class ModelWrapper(torch.nn.Module):
    """Generic wrapper that injects conditioning and null-condition handling into a backbone."""

    def __init__(
        self,
        model,
        shape,
        cond_type="label",
        num_classes=None,
    ):
        """Initialize the model wrapper.

        Args:
            model: The backbone model (callable with x, t, r, cond, **kwargs).
            shape: Data shape.
            cond_type: Type of conditioning.
            num_classes: Number of classes. Required when cond_type == "label".
        """
        super().__init__()
        self.model = model
        self.shape = shape
        self.cond_type = cond_type
        self.num_classes = num_classes
        # if self.cond_type == "label":
        #     assert self.num_classes is not None
        self.unconditional = False
        if self.cond_type is None or self.cond_type == "label" and self.num_classes in [None, 0, 1]:
            self.unconditional = True

    @property
    def device(self):
        """Device of the first model parameter (e.g. cuda:0 or cpu)."""
        return next(self.parameters()).device

    def get_null_condition(self, batch_size):
        """Build a batch of unconditional conditioning."""
        if self.cond_type == "identity":
            return torch.zeros_like(batch_size, *self.shape)
        elif self.cond_type == "label":
            if self.num_classes is None:
                null_label = 0
            else:
                null_label = self.num_classes
            return torch.full((batch_size,), null_label, device=self.device).long()
        else:
            return torch.zeros(batch_size, device=self.device)

    def forward(self, x, t, r=None, cond=None, **kwargs):
        """Forward through the backbone with optional conditioning.

        Args:
            x: Input tensor.
            t: Time step.
            r: Optional second time argument for MeanFlow-style nets.
            cond: Conditioning tensor. If None, replaced by get_null_condition(batch_size).

        Returns:
            Model output tensor, e.g. velocity.
        """
        batch_size = x.shape[0]
        if t.dim() == 0:
            t = t.repeat(batch_size)
        if r is not None and r.dim() == 0:
            r = r.repeat(batch_size)

        if cond is None or self.unconditional:
            cond = self.get_null_condition(batch_size)
        return self.model(x, t, r=r, cond=cond, **kwargs)
