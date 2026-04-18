"""Lightning callbacks for speech enhancement (validation audio exports, etc.)."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import torch
import torchaudio
from lightning.pytorch.callbacks import Callback

from src.tasks.speech_enhancement.data.audio_io import save_audio
from src.tasks.speech_enhancement.models.speech_module import SpeechEnhancementLitModule
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def _format_checkpoint_name(
    filename: str | None,
    epoch: int,
    metrics: dict[str, Any] | None = None,
    *,
    auto_insert_metric_name: bool = True,
) -> str:
    """Expand ``filename`` with ``str.format`` (e.g. ``epoch_{epoch:03d}`` → ``epoch_000``).

    ``auto_insert_metric_name`` is kept for backward compatibility; formatting uses standard
    ``{field[:format]}`` placeholders only.
    """
    del auto_insert_metric_name  # legacy API; Lightning-style format strings need no rewriting.
    if not filename:
        filename = "epoch_{epoch:03d}"
    m = dict(metrics or {})
    m.setdefault("epoch", epoch)
    return filename.format(**m)


def _get_val_dataloader(trainer) -> Any:
    dm = trainer.datamodule
    if dm is None:
        return None
    dl = dm.val_dataloader()
    if isinstance(dl, (list, tuple)):
        dl = dl[0] if dl else None
    return dl


def _to_mono_file_wav(wav: torch.Tensor) -> torch.Tensor:
    """``(B, C, T)`` or ``(B, 1, T)`` → ``(1, T)`` for :func:`save_audio`."""
    x = wav[0]
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


def _save_spectrogram_png(path: str, wav_mono: torch.Tensor, sample_rate: int) -> None:
    """Save a log-magnitude spectrogram image next to exported audio."""
    x = wav_mono.detach().cpu()
    if x.dim() == 2:
        x = x[0]
    x = x.float()
    spec = torch.stft(
        x,
        n_fft=512,
        hop_length=128,
        win_length=512,
        window=torch.hann_window(512),
        return_complex=True,
    )
    spec_db = 20.0 * torch.log10(spec.abs().clamp_min(1e-6))
    fig, ax = plt.subplots(figsize=(8, 3), dpi=120)
    ax.imshow(spec_db.numpy(), origin="lower", aspect="auto", cmap="magma")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Freq bin")
    ax.set_title(f"Spectrogram @ {sample_rate} Hz")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _flow_enhanced_to_wav(
    pl_module: SpeechEnhancementLitModule,
    enhanced: torch.Tensor,
    clean_wav: torch.Tensor,
) -> torch.Tensor:
    if pl_module.hparams.representation == "stft":
        return pl_module._stft_to_wav(enhanced, length=clean_wav.shape[-1])
    est = pl_module.latent_backend.decode_to_wav(enhanced.squeeze(2))
    sr_lb = int(pl_module.latent_backend.sample_rate)
    sr_data = int(pl_module.hparams.sample_rate)
    if est.dim() == 2:
        est = est.unsqueeze(1)
    if sr_lb != sr_data:
        est = torchaudio.functional.resample(est, sr_lb, sr_data)
    m = min(est.shape[-1], clean_wav.shape[-1])
    return est[..., :m]


class SaveValSpeechSamples(Callback):
    """After each validation epoch, run the model on one val batch (first utterance) and write WAVs.

    **Discriminative** modules (:class:`SpeechEnhancementBaselineLitModule`,
    :class:`SpeechMetricGANLitModule`): saves ``clean``, ``noisy``, ``enhanced_ema``, ``enhanced_online``.

    **Flow** (:class:`SpeechEnhancementLitModule`): saves ``clean``, ``noisy``, plus three enhanced samples
    (default NFE + CFG, alternate CFG, fewer steps - mirroring :class:`~src.tasks.image_generation.callbacks.SaveImageGrid`).
    """

    def __init__(
        self,
        dirpath: str | None = None,
        filename: str = "val_audio_{epoch:03d}",
        every_n_epochs: int = 1,
        val_batch_index: int = 36, #
        flow_alt_cfg_scale: float = 2.0,
        flow_alt_num_steps: int = 2,
    ) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.every_n_epochs = int(every_n_epochs)
        self.val_batch_index = int(val_batch_index)
        self.flow_alt_cfg_scale = float(flow_alt_cfg_scale)
        self.flow_alt_num_steps = int(flow_alt_num_steps)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        if getattr(trainer, "sanity_checking", False):
            return
        if self.every_n_epochs < 1:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if not isinstance(
            pl_module, SpeechEnhancementLitModule):
            return

        dl = _get_val_dataloader(trainer)
        if dl is None:
            return

        try:
            it = iter(dl)
            batch = None
            for _ in range(self.val_batch_index + 1):
                batch = next(it)
        except StopIteration:
            log.warning("SaveValSpeechSamples: val dataloader empty, skipping.")
            return

        if not self.dirpath:
            return

        device = pl_module.device
        sr = int(pl_module.hparams.sample_rate)

        clean, noisy = batch
        clean = clean[:1].to(device)
        noisy = noisy[:1].to(device)

        base = _format_checkpoint_name(self.filename, trainer.current_epoch)
        out_dir = os.path.join(self.dirpath, base)
        os.makedirs(out_dir, exist_ok=True)

        with torch.no_grad():
            if isinstance(pl_module, SpeechEnhancementLitModule):
                assert isinstance(pl_module, SpeechEnhancementLitModule)
                flow_batch, meta = pl_module._batch_to_flow((clean, noisy))
                cond = flow_batch[1]
                n_main = int(pl_module.num_steps)
                cfg0 = float(pl_module.cfg_scale)
                if (
                    pl_module.hparams.representation != "stft"
                    and meta.get("latent_source") == "precomputed"
                ):
                    lb = pl_module.latent_backend

                    def _z_to_wav(z: torch.Tensor) -> torch.Tensor:
                        w = lb.decode_to_wav(z.squeeze(2))
                        if w.dim() == 2:
                            w = w.unsqueeze(1)
                        srl = int(lb.sample_rate)
                        if srl != sr:
                            w = torchaudio.functional.resample(w, srl, sr)
                        return w

                    clean_w = _z_to_wav(flow_batch[0])
                    noisy_w = _z_to_wav(flow_batch[1])
                else:
                    clean_w, noisy_w = clean, noisy

                h1 = pl_module.sample(cond, num_steps=n_main, cfg_scale=cfg0, seed=42)
                h2 = pl_module.sample(cond, num_steps=n_main, cfg_scale=self.flow_alt_cfg_scale, seed=42)
                h3 = pl_module.sample(cond, num_steps=self.flow_alt_num_steps, cfg_scale=cfg0, seed=42)
                e1 = _flow_enhanced_to_wav(pl_module, h1, clean_w)
                e2 = _flow_enhanced_to_wav(pl_module, h2, clean_w)
                e3 = _flow_enhanced_to_wav(pl_module, h3, clean_w)
                versions = [
                    ("clean", clean_w),
                    ("noisy", noisy_w),
                    (f"enhanced_nfe{n_main}_cfg{cfg0}", e1),
                    (f"enhanced_nfe{n_main}_cfg{self.flow_alt_cfg_scale}", e2),
                    (f"enhanced_nfe{self.flow_alt_num_steps}_cfg{cfg0}", e3),
                ]

        for name, w in versions:
            wav_path = os.path.join(out_dir, f"{name}.wav")
            png_path = os.path.join(out_dir, f"{name}.png")
            wav_mono = _to_mono_file_wav(w)
            save_audio(wav_path, wav_mono, sample_rate=sr)
            _save_spectrogram_png(png_path, wav_mono, sample_rate=sr)

        log.info("Saved validation audio samples under %s", out_dir)
