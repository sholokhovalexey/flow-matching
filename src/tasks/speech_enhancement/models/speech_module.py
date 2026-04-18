"""Lightning module: flow matching for speech enhancement (STFT or frozen latent codecs / VAE)."""

import torch
import torch.nn.functional as F
import torchaudio
from ema_pytorch import EMA
from lightning import LightningModule
from torchmetrics import MeanMetric

from src.flows import construct_sampler
from src.tasks.speech_enhancement.data.audio_crop import (
    format_stft_shape_mismatch_message,
    stft_unet_spatial_size,
)
from src.tasks.speech_enhancement.metrics.validation_runner import (
    ValidationMetricSelection,
    compute_validation_metrics,
)
from src.tasks.speech_enhancement.models.stft_io import stft_spec_to_wav, wav_to_stft_spec


class SpeechEnhancementLitModule(LightningModule):
    def __init__(
        self,
        flow_model,
        net,
        solver,
        optimizer,
        scheduler,
        representation: str = "stft",
        latent_source: str = "online",
        latent_backend=None,
        codec=None,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        n_freq_bins: int = 256,
        stft_mag_compression_gamma: float = 1.0,
        max_latent_frames: int = 376,
        ema_decay: float = 0.9999,
        num_steps: int = 32,
        cfg_scale: float = 1.0,
        compile: bool = False,
        val_si_sdr_steps: int = 0,
        val_metrics: dict | None = None,
        lsd_config: dict | None = None,
        causal: bool = False,
    ):
        """
        Args:
            latent_source: ``"online"`` (encode waveforms each step) or ``"precomputed"`` (datamodule yields
                ``(z_clean, z_noisy)`` tensors from ``.pt`` files; still need ``latent_backend`` for decode).
            val_si_sdr_steps: If ``> 0``, number of flow/solver steps when sampling enhanced waveforms for
                validation metrics (STFT/latent flow). **Not** used by supervised baselines - there the
                same field triggers a single deterministic forward; see ``SpeechEnhancementBaselineLitModule``.
        """
        super().__init__()
        backend = latent_backend if latent_backend is not None else codec
        self.save_hyperparameters(logger=False, ignore=["latent_backend", "codec"])
        self.net = net
        self.model = flow_model(self.net)
        self.solver = solver
        self.latent_backend = backend
        self.val_si_sdr_steps = val_si_sdr_steps
        self.val_metric_selection = ValidationMetricSelection.from_dict(val_metrics)
        self.lsd_config = dict(lsd_config or {})

        self.ema_decay = ema_decay
        self.ema = EMA(self.model, beta=ema_decay, update_after_step=0, update_every=1)
        self._error_loading_ema = False

        self.num_steps = num_steps
        self.cfg_scale = cfg_scale

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def _is_latent_rep(self) -> bool:
        return self.hparams.representation in ("codec", "latent")

    @torch.no_grad()
    def sample(
        self,
        cond,
        num_steps=32,
        cfg_scale=1.0,
        seed=42,
        initial_noise: torch.Tensor | None = None,
    ):
        self.model.eval()
        samp = construct_sampler(
            self.ema.ema_model,
            self.solver,
            cond,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            seed=seed,
            device=self.device,
            initial_noise=initial_noise,
        )
        return samp()

    def _wav_to_stft(self, wav: torch.Tensor) -> torch.Tensor:
        return wav_to_stft_spec(
            wav,
            n_fft=int(self.hparams.n_fft),
            hop_length=int(self.hparams.hop_length),
            n_freq_bins=int(self.hparams.n_freq_bins),
            mag_compression_gamma=float(self.hparams.stft_mag_compression_gamma),
        )

    def _stft_to_wav(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return stft_spec_to_wav(
            spec,
            n_fft=int(self.hparams.n_fft),
            hop_length=int(self.hparams.hop_length),
            length=length,
            mag_compression_gamma=float(self.hparams.stft_mag_compression_gamma),
        )

    def _encode_latent(self, wav: torch.Tensor) -> torch.Tensor:
        if self.latent_backend is None:
            raise ValueError("latent_backend is required for latent representation")
        if wav.shape[1] != 1:
            wav = wav.mean(dim=1, keepdim=True)
        sr_in = self.hparams.sample_rate
        sr_lb = int(self.latent_backend.sample_rate)
        if sr_in != sr_lb:
            wav = torchaudio.functional.resample(wav, sr_in, sr_lb)
        z = self.latent_backend.encode_continuous(wav)
        z = self._crop_pad_latent(z)
        return z.unsqueeze(2)

    def _crop_pad_latent(self, z: torch.Tensor) -> torch.Tensor:
        Lm = self.hparams.max_latent_frames
        L = z.shape[-1]
        if L >= Lm:
            return z[..., :Lm]
        return F.pad(z, (0, Lm - L))

    def _batch_to_flow(self, batch):
        clean_wav, noisy_wav = batch
        if self.hparams.representation == "stft":
            x1 = self._wav_to_stft(clean_wav)
            cond = self._wav_to_stft(noisy_wav)
            meta = {"length": clean_wav.shape[-1], "clean_wav": clean_wav, "noisy_wav": noisy_wav}
            return (x1, cond), meta
        if self._is_latent_rep():
            if str(self.hparams.latent_source) == "precomputed":
                # (B, C, 1, T) latents from :class:`~src.tasks.speech_enhancement.data.latent_pair_datamodule`
                c, n = clean_wav, noisy_wav
                return (c, n), {"latent_source": "precomputed"}
            c = self._encode_latent(clean_wav)
            n = self._encode_latent(noisy_wav)
            meta = {"clean_wav": clean_wav, "noisy_wav": noisy_wav, "latent_source": "online"}
            return (c, n), meta
        raise ValueError(self.hparams.representation)

    def forward(self, batch, use_ema=False):
        flow_batch, _ = self._batch_to_flow(batch)
        if use_ema:
            return self.ema.ema_model(flow_batch)
        return self.model(flow_batch)

    def model_step(self, batch, use_ema=False):
        out = self.forward(batch, use_ema=use_ema)
        return out["loss"]

    def training_step(self, batch, batch_idx: int):
        loss = self.model_step(batch, use_ema=False)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.trainer.optimizers:
            self.log(
                "lr",
                self.trainer.optimizers[0].param_groups[0]["lr"],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model_step(batch, use_ema=True)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.val_si_sdr_steps > 0 and batch_idx == 0:
            self._log_val_waveform_metrics(batch)

    def _log_val_waveform_metrics(self, batch):
        flow_batch, meta = self._batch_to_flow(batch)
        clean_z, cond = flow_batch[0], flow_batch[1]

        with torch.no_grad():
            enhanced = self.sample(
                cond,
                num_steps=self.val_si_sdr_steps,
                cfg_scale=self.cfg_scale,
                seed=42,
            )

        if self.hparams.representation == "stft":
            clean_wav = meta["clean_wav"]
            est = self._stft_to_wav(enhanced, length=clean_wav.shape[-1])
            tgt = clean_wav
        else:
            est = self.latent_backend.decode_to_wav(enhanced.squeeze(2))
            sr_lb = int(self.latent_backend.sample_rate)
            sr_data = int(self.hparams.sample_rate)
            if est.dim() == 2:
                est = est.unsqueeze(1)
            if sr_lb != sr_data:
                est = torchaudio.functional.resample(est, sr_lb, sr_data)
            if "clean_wav" in meta:
                tgt = meta["clean_wav"]
            else:
                tgt = self.latent_backend.decode_to_wav(clean_z.squeeze(2))
                if tgt.dim() == 2:
                    tgt = tgt.unsqueeze(1)
                if sr_lb != sr_data:
                    tgt = torchaudio.functional.resample(tgt, sr_lb, sr_data)
            m = min(est.shape[-1], tgt.shape[-1])
            est = est[..., :m]
            tgt = tgt[..., :m]

        compute_validation_metrics(
            self,
            est,
            tgt,
            sample_rate=int(self.hparams.sample_rate),
            n_fft=int(self.hparams.n_fft),
            hop_length=int(self.hparams.hop_length),
            sel=self.val_metric_selection,
            device=self.device,
        )

    def _nested_backbone(self):
        n = self.net
        return n.backbone if hasattr(n, "backbone") else n

    def on_fit_start(self):
        bb = self._nested_backbone()
        if self.hparams.causal:
            if getattr(bb, "supports_causal", True) is False:
                raise ValueError(
                    "Training has causal=True but the chosen backbone does not implement causal "
                    "streaming (e.g. diffusers UNet2D uses symmetric padding). Use "
                    "TFConformerBackbone, SpatialNetBackbone, CausalSTFTStackBackbone, LatentDiTBackbone "
                    "with dit.causal=true, etc."
                )
            if hasattr(bb, "causal") and not bool(getattr(bb, "causal")):
                raise ValueError(
                    "Lightning module causal=True but backbone.causal=False - set the backbone's "
                    "causal flag in the model config."
                )

        if self.hparams.representation != "stft":
            return
        g = float(self.hparams.stft_mag_compression_gamma)
        if g <= 0.0:
            raise ValueError(
                "stft_mag_compression_gamma must be positive "
                f"(got {g}). Use 1.0 for uncompressed (raw) STFT."
            )
        dm = self.trainer.datamodule if self.trainer is not None else None
        if dm is None or not hasattr(dm, "hparams"):
            return
        if getattr(dm.hparams, "batch_time_align", "pad_to_segment_length") == "truncate_to_min":
            # Batches can have time length < ``segment_length``; spectrogram width varies per step.
            return
        seg = getattr(dm.hparams, "segment_length", None)
        if seg is None:
            return
        expected = stft_unet_spatial_size(
            int(seg),
            int(self.hparams.n_fft),
            int(self.hparams.hop_length),
            int(self.hparams.n_freq_bins),
        )
        net_shape = getattr(self.net, "shape", None)
        if net_shape is None or len(net_shape) < 3:
            return
        got = tuple(net_shape[1:])
        if expected != got:
            raise ValueError(
                format_stft_shape_mismatch_message(expected, tuple(net_shape))
            )

    def on_train_start(self):
        if (
            self._is_latent_rep()
            and str(self.hparams.latent_source) == "precomputed"
            and self.trainer is not None
            and self.trainer.datamodule is not None
        ):
            dm = self.trainer.datamodule
            if not dm.__class__.__name__.startswith("LatentPt"):
                raise ValueError(
                    "model.latent_source=precomputed requires a ``.pt`` latent datamodule "
                    "(e.g. Hydra ``data=speech_latent_pt_splits`` with ``data.splits_root``). "
                    f"Got datamodule {type(dm).__name__}."
                )
        lb = self.latent_backend
        if lb is not None and hasattr(lb, "ensure_latent_channels"):
            lb.ensure_latent_channels(self.device)

    def setup(self, stage: str):
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)
            if hasattr(self.ema, "online_model"):
                self.ema.online_model = self.model

    def configure_optimizers(self):
        # Train the flow model only - exclude EMA shadow weights from the optimizer.
        optimizer = self.hparams.optimizer(params=self.model.parameters())
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
