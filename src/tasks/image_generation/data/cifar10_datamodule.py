import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import datasets
from torchvision.transforms import transforms


class CIFAR10DataModule(LightningDataModule):

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

        self.image_size = 32
        self.num_channels = 3
        self.num_classes = 10

        # data transformations
        to_tensor = transforms.ToTensor()
        transforms_test = [to_tensor]
        transforms_train = transforms_test + [transforms.RandomHorizontalFlip()]
        self.transforms_train = transforms.Compose(transforms_train)
        self.transforms_test = transforms.Compose(transforms_test)

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.batch_size_per_device = batch_size

    def prepare_data(self):
        datasets.CIFAR10(self.hparams.data_dir, train=True, download=True)
        datasets.CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None):
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
            trainset = datasets.CIFAR10(
                self.hparams.data_dir, train=True, transform=self.transforms_train
            )
            testset = datasets.CIFAR10(
                self.hparams.data_dir, train=False, transform=self.transforms_test
            )
            self.data_test = testset
            dataset = ConcatDataset(datasets=[trainset])
            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=[45_000, 5_000],
                generator=torch.Generator().manual_seed(42),
            )

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
    _ = CIFAR10DataModule()
