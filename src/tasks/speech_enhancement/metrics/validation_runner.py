"""Runs a configurable subset of validation metrics on enhanced vs clean waveforms."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
from lightning import LightningModule

from src.tasks.speech_enhancement.metrics import audio_quality as aq
from src.tasks.speech_enhancement.metrics.si_sdr import si_sdr


@dataclass
class ValidationMetricSelection:
    """Which metrics to log during validation (subset via Hydra ``model.val_metrics``)."""

    si_sdr: bool = True
    lsd: bool = False
    pesq: bool = False
    csig: bool = False
    cbak: bool = False
    covl: bool = False
    dnsmos: bool = False
    dnsmos_personalized: bool = False
    utmos: bool = False
    scoreq_natural_nr: bool = False
    scoreq_natural_ref: bool = False
    urgentmos: bool = False

    @classmethod
    def from_dict(cls, d: Any | None) -> ValidationMetricSelection:
        if d is None:
            return cls()
        try:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(d, DictConfig):
                d = OmegaConf.to_container(d, resolve=True)
        except ImportError:
            pass
        if not d:
            return cls()
        if not isinstance(d, dict):
            raise TypeError(f"val_metrics must be a dict or DictConfig, got {type(d)}")
        known = {f.name for f in fields(cls)}
        base = cls()
        kwargs = {f.name: getattr(base, f.name) for f in fields(cls)}
        for k, v in d.items():
            if k in known:
                kwargs[k] = bool(v)
        return cls(**kwargs)


class _LazyUtmos:
    _scorer = None

    @classmethod
    def score(cls, estimate: torch.Tensor, sr: int) -> torch.Tensor:
        import utmos

        if cls._scorer is None:
            cls._scorer = utmos.Score()
        if estimate.dim() == 3:
            estimate = estimate.squeeze(1)
        scores = []
        for i in range(estimate.shape[0]):
            scores.append(float(cls._scorer.calculate_wav(estimate[i].detach().cpu(), sr)))
        return torch.tensor(scores, dtype=torch.float32).mean()


def _log_warn(pl_module: LightningModule, msg: str) -> None:
    pl_module.print(msg)


def compute_validation_metrics(
    pl_module: LightningModule,
    estimate: torch.Tensor,
    reference: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    sel: ValidationMetricSelection,
    device: torch.device,
) -> None:
    """Computes enabled metrics and calls ``pl_module.log('val/...')``."""

    def log(name: str, value: torch.Tensor | float, *, prog_bar: bool = False) -> None:
        pl_module.log(name, value, on_epoch=True, prog_bar=prog_bar)

    lsd_cfg = {}
    try:
        cfg_obj = getattr(pl_module.hparams, "lsd_config", None)
        if cfg_obj is not None:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(cfg_obj, DictConfig):
                lsd_cfg = dict(OmegaConf.to_container(cfg_obj, resolve=True))
            elif isinstance(cfg_obj, dict):
                lsd_cfg = dict(cfg_obj)
    except Exception:
        lsd_cfg = {}

    if sel.si_sdr:
        s = si_sdr(estimate, reference).mean()
        log("val/si_sdr", s, prog_bar=True)
    if sel.lsd:
        l = aq.log_spectral_distance_db(
            estimate,
            reference,
            n_fft=n_fft,
            hop_length=hop_length,
            **lsd_cfg,
        ).mean()
        log("val/lsd_db", l)

    if sel.pesq:
        try:
            p = aq.pesq_wb_batch(estimate, reference, sample_rate).mean().to(device)
            log("val/pesq", p, prog_bar=True)
        except Exception as e:
            _log_warn(pl_module, f"[val/pesq skipped] {e!r}")
    if sel.csig or sel.cbak or sel.covl:
        try:
            comp = aq.composite_csig_cbak_covl_batch(estimate, reference, sample_rate)
            if sel.csig:
                log("val/csig", comp["csig"].mean().to(device))
            if sel.cbak:
                log("val/cbak", comp["cbak"].mean().to(device))
            if sel.covl:
                log("val/covl", comp["covl"].mean().to(device), prog_bar=True)
        except Exception as e:
            _log_warn(pl_module, f"[val/composite skipped] {e!r}")

    est_detached = estimate.detach()
    if sel.dnsmos:
        try:
            d = aq.dnsmos_scores(
                est_detached,
                sample_rate=sample_rate,
                personalized=sel.dnsmos_personalized,
                device=device,
            )
            log("val/dnsmos_p808", d["p808"].to(device))
            log("val/dnsmos_sig", d["sig"].to(device))
            log("val/dnsmos_bak", d["bak"].to(device))
            log("val/dnsmos_ovrl", d["ovrl"].to(device))
        except Exception as e:
            _log_warn(pl_module, f"[val/dnsmos skipped] {e!r}")

    if sel.utmos:
        try:
            u = _LazyUtmos.score(estimate, sample_rate)
            log("val/utmos", u.to(device))
        except Exception as e:
            _log_warn(pl_module, f"[val/utmos skipped] {e!r}")

    if sel.scoreq_natural_nr:
        try:
            s = aq.scoreq_natural_nr_mean(estimate, sample_rate)
            log("val/scoreq_natural_nr", s.to(device))
        except Exception as e:
            _log_warn(pl_module, f"[val/scoreq_natural_nr skipped] {e!r}")

    if sel.scoreq_natural_ref:
        try:
            s = aq.scoreq_natural_ref_distance(estimate, reference, sample_rate)
            log("val/scoreq_natural_ref_dist", s.to(device))
        except Exception as e:
            _log_warn(pl_module, f"[val/scoreq_natural_ref skipped] {e!r}")

    if sel.urgentmos:
        try:
            aq.urgentmos_placeholder()
        except NotImplementedError as e:
            _log_warn(pl_module, f"[val/urgentmos] {e}")
