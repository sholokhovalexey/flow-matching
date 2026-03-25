import numpy as np
import torch
import torch.distributed as dist
from torchmetrics import MaxMetric, MeanMetric
from lightning import LightningModule
from ema_pytorch import EMA


class ImaGenLitModule(LightningModule):

    def __init__(
        self,
        flow_model,
        net,
        solver,
        optimizer,
        scheduler,
        ema_decay=0.9999,
        num_steps=32,
        cfg_scale=2.0,
        compile=False,
    ):
        super().__init__()

        # this line allows to access init params with "self.hparams" attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.model = flow_model(self.net)
        self.solver = solver

        # EMA
        self.ema_decay = ema_decay
        self.ema = EMA(self.model, beta=ema_decay, update_after_step=0, update_every=1)
        self._error_loading_ema = False

        # inference
        self.num_steps = num_steps
        self.cfg_scale = cfg_scale

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    @torch.no_grad()
    def sample(self, cond, num_steps=32, cfg_scale=1.0, seed=42):
        from src.flows import construct_sampler
        self.model.eval()
        sample = construct_sampler(
            self.ema.ema_model,
            self.solver,
            cond,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            seed=seed,
            device=self.device,
        )
        x_gen = sample()
        return x_gen

    def image_to_tensor(self, x):
        # scaling from [0, 1] to [-1, 1]
        return x * 2.0 - 1.0

    def tensor_to_image(self, x):
        # scaling from [-1, 1] to [0, 1]
        return (x / 2.0 + 0.5).clip(0, 1)

    def forward(self, batch, use_ema=False):
        x, y = batch
        x = self.image_to_tensor(x)
        batch = (x, y)
        if use_ema:
            return self.ema.ema_model(batch)
        else:
            return self.model(batch)

    def on_train_start(self):
        self.val_loss.reset()

    def model_step(self, batch, use_ema=False):
        out_dict = self.forward(batch, use_ema=use_ema)
        loss = out_dict["loss"]
        return loss

    def training_step(self, batch, batch_idx: int):
        loss = self.model_step(batch, use_ema=False)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        return

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        loss = self.model_step(batch, use_ema=True)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        loss = self.model_step(batch, use_ema=True)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def setup(self, stage: str):
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update()

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
    #     self.clip_gradients(
    #         optimizer,
    #         gradient_clip_val=self.trainer.gradient_clip_val,
    #         gradient_clip_algorithm="norm"
    #     )
    #     optimizer.step(closure=optimizer_closure)
    #     self.ema.update()

    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get("ema", None)
        if ema is not None:
            self.ema.ema_model.load_state_dict(checkpoint["ema"])
        else:
            self._error_loading_ema = True

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.ema_model.state_dict()

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def on_after_backward(self):
        def gradient_norm(model):
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)
            return total_norm
        self.log("grad_norm", gradient_norm(self.model), on_step=True, on_epoch=False, prog_bar=True)


if __name__ == "__main__":
    _ = ImaGenLitModule(None, None, None, None)
