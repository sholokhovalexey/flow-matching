"""VoiceBank + DEMAND (or compatible) parallel clean/noisy folders at 16 kHz."""

from __future__ import annotations

import os
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.tasks.speech_enhancement.data.audio_io import load_audio
from src.tasks.speech_enhancement.data.augment import (
    MusanRirAugment,
    augment_config_from_dict,
    coerce_augment_mapping,
)
from src.tasks.speech_enhancement.data.audio_crop import (
    crop_clean_noisy_waveform_pair,
    normalize_batch_time_align,
    normalize_segment_crop_mode,
    resolve_segment_samples,
)
from src.tasks.speech_enhancement.data.waveform_collate import make_speech_waveform_pair_collate_fn


def list_pairs_parallel_clean_noisy(clean_dir: str, noisy_dir: str) -> List[Tuple[str, str]]:
    """Pair wav files by matching basename under ``clean_dir`` and ``noisy_dir``."""
    clean_files = {}
    for name in os.listdir(clean_dir):
        if name.lower().endswith(".wav"):
            clean_files[name] = os.path.join(clean_dir, name)
    pairs = []
    for name, cp in clean_files.items():
        np = os.path.join(noisy_dir, name)
        if os.path.isfile(np):
            pairs.append((cp, np))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _splits_layout_help(splits_root: str) -> str:
    return (
        f"Expected layout under splits_root (absolute path on your machine):\n"
        f"  {os.path.join(splits_root, '<split>', 'clean', '*.wav')}\n"
        f"  {os.path.join(splits_root, '<split>', 'noisy', '*.wav')}\n"
        f"Example Hydra override: data.splits_root=C:/Users/you/datasets/VoicebankDEMAND/splits\n"
        f"(Do not use documentation placeholders like C:/path/to/... - that path must exist.)\n"
        f"Prepare folders from the VoiceBank+DEMAND release with scripts/speech_enhancement/prepare_voicebank.py "
        f"or your own split manifests; see configs/data/speech_voicebank_splits.yaml."
    )


def list_pairs_from_splits_subdir(splits_root: str, split: str) -> List[Tuple[str, str]]:
    """Load pairs from ``splits_root/<split>/clean`` and ``splits_root/<split>/noisy`` (same basenames).

    Typical layout (e.g. Voicebank+DEMAND repacked under a ``splits`` folder)::

        splits_root/train/clean/*.wav
        splits_root/train/noisy/*.wav
        splits_root/valid/clean/*.wav
        splits_root/valid/noisy/*.wav
        splits_root/test/clean/*.wav
        splits_root/test/noisy/*.wav
    """
    splits_root = os.path.abspath(os.path.expanduser(str(splits_root)))
    clean_dir = os.path.join(splits_root, split, "clean")
    noisy_dir = os.path.join(splits_root, split, "noisy")
    if not os.path.isdir(clean_dir):
        raise FileNotFoundError(
            f"Missing clean dir for split {split!r}:\n  {clean_dir}\n\n{_splits_layout_help(splits_root)}"
        )
    if not os.path.isdir(noisy_dir):
        raise FileNotFoundError(
            f"Missing noisy dir for split {split!r}:\n  {noisy_dir}\n\n{_splits_layout_help(splits_root)}"
        )
    return list_pairs_parallel_clean_noisy(clean_dir, noisy_dir)


class VoicebankDemandParallelDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        sample_rate: int,
        segment_length: int,
        seed: int,
        extra_augment: Optional[MusanRirAugment],
        segment_crop_mode: str = "random",
        batch_time_align: str = "pad_to_segment_length",
    ):
        self.pairs = pairs
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.rng = random.Random(seed)
        self.extra_augment = extra_augment
        self.segment_crop_mode = normalize_segment_crop_mode(segment_crop_mode)
        self.batch_time_align = normalize_batch_time_align(batch_time_align)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        cp, np = self.pairs[idx]
        clean, sr_c = load_audio(cp)
        noisy, sr_n = load_audio(np)
        if clean.shape[0] > 1:
            clean = clean.mean(dim=0, keepdim=True)
        if noisy.shape[0] > 1:
            noisy = noisy.mean(dim=0, keepdim=True)
        if sr_c != self.sample_rate:
            clean = torchaudio.functional.resample(clean, sr_c, self.sample_rate)
        if sr_n != self.sample_rate:
            noisy = torchaudio.functional.resample(noisy, sr_n, self.sample_rate)
        clean, noisy = crop_clean_noisy_waveform_pair(
            clean,
            noisy,
            self.segment_length,
            crop_mode=self.segment_crop_mode,
            batch_time_align=self.batch_time_align,
            rng=self.rng,
        )

        if self.extra_augment is not None and (
            self.extra_augment.has_musan or self.extra_augment.has_rir
        ):
            noisy = self.extra_augment.extra_degrade_noisy(noisy)

        return clean, noisy


class VoiceBankDemandDataModule(LightningDataModule):
    """Expects::

        root/train/clean/*.wav  parallel to  root/train/noisy/*.wav
        root/test/clean/*.wav   parallel to  root/test/noisy/*.wav

    Paths are configurable via ``train_clean_dir``, ``train_noisy_dir``, etc.
    """

    def __init__(
        self,
        train_clean_dir: str,
        train_noisy_dir: str,
        val_clean_dir: str,
        val_noisy_dir: str,
        sample_rate: int = 16000,
        segment_length: int = 16000,
        segment_duration_sec: Optional[float] = None,
        batch_size: int = 8,
        num_workers: int = 0,
        seed: int = 42,
        pin_memory: bool = True,
        augment: Optional[Dict[str, Any]] = None,
        segment_crop_mode: str = "random",
        batch_time_align: str = "pad_to_segment_length",
    ):
        segment_length = resolve_segment_samples(
            sample_rate, segment_length, segment_duration_sec
        )
        normalize_segment_crop_mode(segment_crop_mode)
        normalize_batch_time_align(batch_time_align)
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train = None
        self.data_val = None

    def _extra_aug(self) -> Optional[MusanRirAugment]:
        a = self.hparams.augment
        if not a:
            return None
        cfg = augment_config_from_dict(coerce_augment_mapping(a))
        if not cfg.musan_root and not cfg.rir_root:
            return None
        if cfg.p_musan <= 0 and cfg.p_rir <= 0:
            return None
        return MusanRirAugment(cfg, self.hparams.sample_rate, self.hparams.seed)

    def setup(self, stage: Optional[str] = None):
        train_pairs = list_pairs_parallel_clean_noisy(self.hparams.train_clean_dir, self.hparams.train_noisy_dir)
        val_pairs = list_pairs_parallel_clean_noisy(self.hparams.val_clean_dir, self.hparams.val_noisy_dir)
        if not train_pairs:
            raise FileNotFoundError(
                f"No paired wav files under {self.hparams.train_clean_dir} / {self.hparams.train_noisy_dir}"
            )
        ex = self._extra_aug()
        if stage in ("fit", None):
            self.data_train = VoicebankDemandParallelDataset(
                train_pairs,
                self.hparams.sample_rate,
                self.hparams.segment_length,
                self.hparams.seed,
                ex,
                segment_crop_mode=self.hparams.segment_crop_mode,
                batch_time_align=self.hparams.batch_time_align,
            )
        if stage in ("fit", "validate", None):
            self.data_val = VoicebankDemandParallelDataset(
                val_pairs or train_pairs[: max(1, len(train_pairs) // 10)],
                self.hparams.sample_rate,
                self.hparams.segment_length,
                self.hparams.seed + 1,
                None,
                segment_crop_mode=self.hparams.segment_crop_mode,
                batch_time_align=self.hparams.batch_time_align,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            collate_fn=make_speech_waveform_pair_collate_fn(self.hparams.batch_time_align),
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            collate_fn=make_speech_waveform_pair_collate_fn(self.hparams.batch_time_align),
        )


class VoiceBankDemandSplitsDataModule(LightningDataModule):
    """VoiceBank+DEMAND with a single **splits** root: ``<splits_root>/<split_name>/{clean,noisy}``.

    Matches on-disk layouts such as::

        <splits_root>/train/clean/*.wav   +  train/noisy/*.wav
        <splits_root>/valid/clean/*.wav   +  valid/noisy/*.wav
        <splits_root>/test/clean/*.wav    +  test/noisy/*.wav

    Split folder names default to ``train`` / ``valid``; set ``val_split: val`` if your tree uses ``val``.
    """

    def __init__(
        self,
        splits_root: str,
        train_split: str = "train",
        val_split: str = "valid",
        sample_rate: int = 16000,
        segment_length: int = 16000,
        segment_duration_sec: Optional[float] = None,
        batch_size: int = 8,
        num_workers: int = 0,
        seed: int = 42,
        pin_memory: bool = True,
        augment: Optional[Dict[str, Any]] = None,
        segment_crop_mode: str = "random",
        batch_time_align: str = "pad_to_segment_length",
    ):
        segment_length = resolve_segment_samples(
            sample_rate, segment_length, segment_duration_sec
        )
        normalize_segment_crop_mode(segment_crop_mode)
        normalize_batch_time_align(batch_time_align)
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train = None
        self.data_val = None

    def _extra_aug(self) -> Optional[MusanRirAugment]:
        a = self.hparams.augment
        if not a:
            return None
        cfg = augment_config_from_dict(coerce_augment_mapping(a))
        if not cfg.musan_root and not cfg.rir_root:
            return None
        if cfg.p_musan <= 0 and cfg.p_rir <= 0:
            return None
        return MusanRirAugment(cfg, self.hparams.sample_rate, self.hparams.seed)

    def setup(self, stage: Optional[str] = None):
        root = self.hparams.splits_root
        train_pairs = list_pairs_from_splits_subdir(root, self.hparams.train_split)
        if not train_pairs:
            raise FileNotFoundError(
                f"No paired wav files under {os.path.join(root, self.hparams.train_split, 'clean')} / "
                f"{os.path.join(root, self.hparams.train_split, 'noisy')}"
            )
        val_missing_split = False
        try:
            val_pairs = list_pairs_from_splits_subdir(root, self.hparams.val_split)
        except FileNotFoundError:
            val_pairs = []
            val_missing_split = True
            warnings.warn(
                f"Validation split {self.hparams.val_split!r} not found or incomplete under {root}; "
                "using 10% of training pairs for validation.",
                UserWarning,
                stacklevel=2,
            )
        if not val_pairs:
            val_pairs = train_pairs[: max(1, len(train_pairs) // 10)]
            if not val_missing_split:
                warnings.warn(
                    f"No paired wav files under {os.path.join(root, self.hparams.val_split, 'clean')} / "
                    f"{os.path.join(root, self.hparams.val_split, 'noisy')}; "
                    "using 10% of training pairs for validation.",
                    UserWarning,
                    stacklevel=2,
                )

        ex = self._extra_aug()
        if stage in ("fit", None):
            self.data_train = VoicebankDemandParallelDataset(
                train_pairs,
                self.hparams.sample_rate,
                self.hparams.segment_length,
                self.hparams.seed,
                ex,
                segment_crop_mode=self.hparams.segment_crop_mode,
                batch_time_align=self.hparams.batch_time_align,
            )
        if stage in ("fit", "validate", None):
            self.data_val = VoicebankDemandParallelDataset(
                val_pairs,
                self.hparams.sample_rate,
                self.hparams.segment_length,
                self.hparams.seed + 1,
                None,
                segment_crop_mode=self.hparams.segment_crop_mode,
                batch_time_align=self.hparams.batch_time_align,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            collate_fn=make_speech_waveform_pair_collate_fn(self.hparams.batch_time_align),
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            collate_fn=make_speech_waveform_pair_collate_fn(self.hparams.batch_time_align),
        )
