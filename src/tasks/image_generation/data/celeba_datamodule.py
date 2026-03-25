import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import datasets
from torchvision.transforms import transforms


class CelebADataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.image_size = 64
        self.num_channels = 3
        # self.num_classes = 8192

        # data transformations
        to_tensor = transforms.ToTensor()
        transforms_test = [to_tensor, transforms.CenterCrop(128), transforms.Resize(64)]
        transforms_train = transforms_test + [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.0,
            ),
        ]
        self.transforms_train = transforms.Compose(transforms_train)
        self.transforms_test = transforms.Compose(transforms_test)

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.batch_size_per_device = batch_size

    def prepare_data(self):
        datasets.CelebA(
            self.hparams.data_dir, split="all", target_type="identity", download=True
        )

    def make_contiguous_labels(self):
        identity_train = self.data_train.identity
        identity_val = self.data_val.identity
        identity_test = self.data_test.identity

        sizes = [len(identity_train), len(identity_val), len(identity_test)]
        identity = torch.cat([identity_train, identity_val, identity_test])

        unique, inverse_map = torch.unique(identity, return_inverse=True)
        identity = torch.arange(len(unique)).long()[inverse_map]

        self.num_classes = len(unique)

        self.data_train.identity = identity[: sizes[0]]
        self.data_val.identity = identity[sizes[0] : sizes[0] + sizes[1]]
        self.data_test.identity = identity[sizes[0] + sizes[1] :]

    def setup(self, stage):
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = datasets.CelebA(
                self.hparams.data_dir,
                split="train",
                target_type="identity",
                transform=self.transforms_train,
            )
            self.data_val = datasets.CelebA(
                self.hparams.data_dir,
                split="valid",
                target_type="identity",
                transform=self.transforms_test,
            )
            self.data_test = datasets.CelebA(
                self.hparams.data_dir,
                split="test",
                target_type="identity",
                transform=self.transforms_test,
            )

            self.make_contiguous_labels()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage=None):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


if __name__ == "__main__":
    _ = CelebADataModule()
