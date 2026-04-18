"""Full-reference and no-reference audio quality metrics."""

from __future__ import annotations

import tempfile
import os
import sys
from glob import glob
from pathlib import Path
import numpy as np
import torch

# --- LSD (log spectral distance, full-reference) ---


def log_spectral_distance_db(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    *,
    n_fft: int = 512,
    hop_length: int = 128,
    db_range: float = 50.0,
    vad_db_threshold: float = -50.0,
    exclude_edge_bins: bool = True,
) -> torch.Tensor:
    """LSD in dB on active frames (lower is better):
    - compute magnitude STFTs and convert to dB with a dynamic floor ``db_range`` below
      each utterance's reference peak;
    - compute per-frame LSD = sqrt(mean_f((ref_db - est_db)^2));
    - average only over active frames determined by reference-frame energy threshold.
    """
    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if reference.dim() == 3:
        reference = reference.squeeze(1)
    device = estimate.device
    w = torch.hann_window(n_fft, device=device)
    est = torch.stft(estimate, n_fft, hop_length, window=w, return_complex=True, center=True)
    ref = torch.stft(reference, n_fft, hop_length, window=w, return_complex=True, center=True)
    mag_e = est.abs()
    mag_r = ref.abs()

    # Dynamic floor relative to per-utterance peak reference magnitude.
    ref_peak = mag_r.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    floor = ref_peak * (10.0 ** (-float(db_range) / 20.0))
    est_db = 20.0 * torch.log10(torch.maximum(mag_e, floor))
    ref_db = 20.0 * torch.log10(torch.maximum(mag_r, floor))

    if exclude_edge_bins and est_db.shape[-2] > 2:
        est_db = est_db[:, 1:-1, :]
        ref_db = ref_db[:, 1:-1, :]
        mag_r = mag_r[:, 1:-1, :]

    # Per-frame LSD over frequency bins.
    lsd_frame = torch.sqrt(torch.mean((ref_db - est_db) ** 2, dim=-2))  # (B, T)

    # Active-frame mask from reference frame energy.
    ref_frame_db = 10.0 * torch.log10(mag_r.pow(2).mean(dim=-2).clamp_min(1e-12))  # (B, T)
    active = ref_frame_db > float(vad_db_threshold)

    # Ensure at least one frame per sample to avoid empty reductions.
    active_counts = active.sum(dim=-1).clamp_min(1)
    lsd_sum = (lsd_frame * active.to(lsd_frame.dtype)).sum(dim=-1)
    return lsd_sum / active_counts


# --- PESQ (optional: pip install pesq) ---


def pesq_wb_batch(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Wideband PESQ per utterance; returns tensor ``(B,)`` on CPU float.

    Requires ``pip install pesq``. Uses 16 kHz wideband mode.
    """
    try:
        from pesq import pesq as _pesq
    except ImportError as e:
        raise ImportError("PESQ requires: pip install pesq") from e

    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if reference.dim() == 3:
        reference = reference.squeeze(1)
    est = estimate.detach().cpu().numpy().astype(np.float32)
    ref = reference.detach().cpu().numpy().astype(np.float32)
    scores = []
    for i in range(est.shape[0]):
        m = min(est.shape[1], ref.shape[1])
        scores.append(float(_pesq(sample_rate, ref[i, :m], est[i, :m], "wb")))
    return torch.tensor(scores, dtype=torch.float32)


# --- Composite objective quality (CSIG/CBAK/COVL) ---


def composite_csig_cbak_covl_batch(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    sample_rate: int = 16000,
) -> dict[str, torch.Tensor]:
    """Batch composite metrics from Hu & Loizou objective quality model.

    Returns per-utterance tensors for ``csig``, ``cbak``, ``covl``.
    Requires ``pysepm`` (GitHub package).
    """
    # pysepm still references ``np.NaN`` (removed in NumPy 2.0); add compatibility alias.
    if not hasattr(np, "NaN"):
        setattr(np, "NaN", np.nan)
    try:
        import pysepm  # type: ignore
    except ImportError as e:
        raise ImportError(
            "CSIG/CBAK/COVL require pysepm. Install via: "
            "pip install git+https://github.com/schmiph2/pysepm.git"
        ) from e

    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if reference.dim() == 3:
        reference = reference.squeeze(1)
    est = estimate.detach().cpu().numpy().astype(np.float64)
    ref = reference.detach().cpu().numpy().astype(np.float64)

    csig: list[float] = []
    cbak: list[float] = []
    covl: list[float] = []
    for i in range(est.shape[0]):
        m = min(est.shape[1], ref.shape[1])
        out = pysepm.composite(ref[i, :m], est[i, :m], int(sample_rate))
        if len(out) >= 3:
            s, b, o = float(out[0]), float(out[1]), float(out[2])
        else:
            raise RuntimeError(f"Unexpected pysepm.composite output: {out!r}")
        csig.append(s)
        cbak.append(b)
        covl.append(o)
    return {
        "csig": torch.tensor(csig, dtype=torch.float32),
        "cbak": torch.tensor(cbak, dtype=torch.float32),
        "covl": torch.tensor(covl, dtype=torch.float32),
    }


# --- DNSMOS via torchmetrics (no-reference; preds = enhanced only) ---

_ONNX_LIBS_PATCHED = False
_SCOREQ_MODELS: dict[tuple[str, str], object] = {}


def _ensure_onnx_cuda_runtime_libs() -> None:
    """Expose pip-installed NVIDIA CUDA libs to ONNX Runtime loader.

    ``onnxruntime-gpu`` needs CUDA/cuDNN shared objects discoverable via
    ``LD_LIBRARY_PATH``. When CUDA runtime wheels are installed under
    ``site-packages/nvidia/*/lib``, this helper prepends those paths.
    """
    global _ONNX_LIBS_PATCHED
    if _ONNX_LIBS_PATCHED:
        return

    base = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia"
    if not base.exists():
        _ONNX_LIBS_PATCHED = True
        return

    lib_dirs = sorted({str(Path(p).parent) for p in glob(str(base / "**" / "lib" / "*.so*"), recursive=True)})
    if not lib_dirs:
        _ONNX_LIBS_PATCHED = True
        return

    cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([cur] if cur else []))
    _ONNX_LIBS_PATCHED = True


def dnsmos_scores(
    estimate: torch.Tensor,
    *,
    sample_rate: int = 16000,
    personalized: bool = False,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Returns dict with ``p808``, ``sig``, ``bak``, ``ovrl`` (batch means as scalars).

    Uses :func:`torchmetrics.functional.audio.dnsmos.deep_noise_suppression_mean_opinion_score`.
    Downloads ONNX weights on first use (``~/.torchmetrics/DNSMOS``).

    Requires: ``pip install librosa onnxruntime requests`` (or ``torchmetrics[audio]``).
    """
    _ensure_onnx_cuda_runtime_libs()
    try:
        from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
    except ImportError as e:
        raise ImportError(
            "DNSMOS requires torchmetrics audio extras: pip install 'torchmetrics[audio]' "
            "or librosa onnxruntime requests"
        ) from e

    if estimate.dim() == 3:
        x = estimate.squeeze(1)
    else:
        x = estimate
    dev = device or x.device
    if isinstance(dev, torch.device):
        if dev.type == "cuda":
            cuda_idx = dev.index if dev.index is not None else torch.cuda.current_device()
            dev_for_tm = f"cuda:{cuda_idx}"
        else:
            dev_for_tm = str(dev)
    else:
        dev_for_tm = dev
    # preds: [..., time]
    out = deep_noise_suppression_mean_opinion_score(
        x.float(),
        fs=sample_rate,
        personalized=personalized,
        device=dev_for_tm,
    )
    # shape (B, 4): p808, sig, bak, ovrl
    if out.dim() == 1:
        out = out.unsqueeze(0)
    return {
        "p808": out[:, 0].mean(),
        "sig": out[:, 1].mean(),
        "bak": out[:, 2].mean(),
        "ovrl": out[:, 3].mean(),
    }


# --- UTMOS (optional: pip install utmos) ---


def utmos_score(estimate: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """Mean UTMOS score (scalar tensor). Requires ``pip install utmos``."""
    try:
        import utmos
    except ImportError as e:
        raise ImportError("UTMOS requires: pip install utmos") from e

    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    scorer = utmos.Score()
    scores = []
    for i in range(estimate.shape[0]):
        wav = estimate[i].detach().cpu()
        scores.append(float(scorer.calculate_wav(wav, sample_rate)))
    return torch.tensor(scores, dtype=torch.float32).mean()


# --- SCOREQ (optional: pip install scoreq) - uses temp wav files ---


def scoreq_natural_nr_mean(estimate: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """No-reference SCOREQ (natural domain). Mean over batch."""
    _ensure_onnx_cuda_runtime_libs()
    try:
        import scoreq
    except ImportError as e:
        raise ImportError("SCOREQ requires: pip install scoreq (Python >= 3.10)") from e

    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    key = ("natural", "nr")
    model = _SCOREQ_MODELS.get(key)
    if model is None:
        model = scoreq.Scoreq(data_domain="natural", mode="nr")
        _SCOREQ_MODELS[key] = model
    scores: list[float] = []
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        for i in range(estimate.shape[0]):
            p = td / f"e_{i}.wav"
            _write_wav_mono16k(p, estimate[i], sample_rate)
            scores.append(float(model.predict(test_path=str(p), ref_path=None)))
    return torch.tensor(float(np.mean(scores)), dtype=torch.float32)


def scoreq_natural_ref_distance(
    estimate: torch.Tensor, reference: torch.Tensor, sample_rate: int = 16000
) -> torch.Tensor:
    """Full-reference natural SCOREQ (distance; lower is better). Mean over batch."""
    _ensure_onnx_cuda_runtime_libs()
    try:
        import scoreq
    except ImportError as e:
        raise ImportError("SCOREQ requires: pip install scoreq") from e

    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if reference.dim() == 3:
        reference = reference.squeeze(1)
    key = ("natural", "ref")
    model = _SCOREQ_MODELS.get(key)
    if model is None:
        model = scoreq.Scoreq(data_domain="natural", mode="ref")
        _SCOREQ_MODELS[key] = model
    scores: list[float] = []
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        for i in range(estimate.shape[0]):
            pe = td / f"e_{i}.wav"
            pr = td / f"r_{i}.wav"
            m = min(estimate.shape[1], reference.shape[1])
            _write_wav_mono16k(pe, estimate[i, :m], sample_rate)
            _write_wav_mono16k(pr, reference[i, :m], sample_rate)
            scores.append(float(model.predict(test_path=str(pe), ref_path=str(pr))))
    return torch.tensor(float(np.mean(scores)), dtype=torch.float32)


def _write_wav_mono16k(path: Path, wav_1d: torch.Tensor, sr: int) -> None:
    import soundfile as sf

    x = wav_1d.detach().cpu().numpy().astype(np.float32)
    if sr != 16000:
        try:
            import librosa

            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            sr = 16000
        except ImportError:
            raise ImportError("Resampling for SCOREQ requires librosa or 16 kHz audio") from None
    sf.write(str(path), x, sr, subtype="PCM_16")


