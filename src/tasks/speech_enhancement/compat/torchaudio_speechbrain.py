"""Compatibility for SpeechBrain with torchaudio 2.9+.

SpeechBrain < develop (e.g. 1.0.x) calls :func:`torchaudio.list_audio_backends`, which was
removed in torchaudio 2.9. Apply this patch **before** importing ``speechbrain`` so the
check in ``speechbrain.utils.torch_audio_backend`` does not raise ``AttributeError``.
"""

from __future__ import annotations


def apply_torchaudio_speechbrain_compat() -> None:
    import torchaudio

    if hasattr(torchaudio, "list_audio_backends"):
        return

    def list_audio_backends() -> list[str]:
        # Match pre-2.9 behavior enough for SpeechBrain's empty-backend warning.
        try:
            import soundfile  # noqa: F401

            return ["soundfile"]
        except ImportError:
            return []

    torchaudio.list_audio_backends = list_audio_backends  # type: ignore[attr-defined, assignment]
