import os
import re
import torch
from torchvision.utils import make_grid, save_image
from lightning.pytorch.callbacks import Callback


class SaveImageGrid(Callback):

    CHECKPOINT_JOIN_CHAR = "-"
    FILE_EXTENSION = ".png"
    STARTING_VERSION = 1

    def __init__(self, dirpath=None, filename=None):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename

    @classmethod
    def _format_checkpoint_name(
        cls, filename, epoch, metrics={}, prefix="", auto_insert_metric_name=True
    ):
        if not filename:
            filename = "{epoch}"

        # check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)
        if len(groups) >= 0:
            metrics.update({"epoch": epoch})
            for group in groups:
                name = group[1:]

                if auto_insert_metric_name:
                    filename = filename.replace(group, name + "={" + name)

                if name not in metrics:
                    metrics[name] = 0
            filename = filename.format(**metrics)

        if prefix:
            filename = cls.CHECKPOINT_JOIN_CHAR.join([prefix, filename])

        return filename

    def on_train_start(self, trainer, pl_module):
        pass

    def on_validation_epoch_end(self, trainer, pl_module):
        pass

        prefix = ""

        num_classes = pl_module.net.num_classes
        # num_classes = trainer.datamodule.num_classes # alternative

        if num_classes is None:
            num_classes = 10

        device = pl_module.model.device

        with torch.no_grad():

            if pl_module.net.cond_type == "label" and pl_module.net.num_classes in [
                None,
                0,
                1,
            ]:
                cond = None
            else:
                cond = torch.arange(100).long()
                cond = torch.remainder(cond, num_classes).to(device)

            x_gen = pl_module.sample(
                cond, num_steps=pl_module.num_steps, cfg_scale=pl_module.cfg_scale
            )

            cfg_scale = 3.5
            x_gen_cfg = pl_module.sample(
                cond, num_steps=pl_module.num_steps, cfg_scale=cfg_scale
            )

            num_steps = 2
            x_gen_steps = pl_module.sample(
                cond, num_steps=num_steps, cfg_scale=pl_module.cfg_scale
            )

        x_gen = pl_module.tensor_to_image(x_gen)
        x_gen_cfg = pl_module.tensor_to_image(x_gen_cfg)
        x_gen_steps = pl_module.tensor_to_image(x_gen_steps)

        grid = make_grid(x_gen.cpu(), nrow=10)
        grid_cfg = make_grid(x_gen_cfg.cpu(), nrow=10)
        grid_steps = make_grid(x_gen_steps.cpu(), nrow=10)

        os.makedirs(self.dirpath, exist_ok=True)

        filepath = self._format_checkpoint_name(
            self.filename,
            trainer.current_epoch,
        )

        extra = f"{prefix}_nfe={pl_module.num_steps}_cfg={pl_module.cfg_scale}"
        save_path = os.path.join(
            self.dirpath, f"{filepath}_{extra}{self.FILE_EXTENSION}"
        )
        save_image(grid, save_path)

        extra = f"{prefix}_nfe={pl_module.num_steps}_cfg={cfg_scale}"
        save_path = os.path.join(
            self.dirpath, f"{filepath}_{extra}{self.FILE_EXTENSION}"
        )
        save_image(grid_cfg, save_path)

        extra = f"{prefix}_nfe={num_steps}_cfg={pl_module.cfg_scale}"
        save_path = os.path.join(
            self.dirpath, f"{filepath}_{extra}{self.FILE_EXTENSION}"
        )
        save_image(grid_steps, save_path)
        # print("Saved image at", save_path)
