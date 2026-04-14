"""Decoder resolution helpers for baseline and threshold codepaths."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from surface_code_in_stem.decoders.base import DecoderProtocol
from surface_code_in_stem.decoders.mwpm import MWPMDecoder


@dataclass(frozen=True)
class DecoderResolution:
    """Result of a decoder lookup with fallback metadata."""

    requested_name: str | None
    resolved_name: str
    decoder: DecoderProtocol
    fallback_reason: str | None = None


def _default_container_factory() -> Any:
    """Return the global container at call time to support test monkeypatching."""
    from syndrome_net.container import get_container

    return get_container()


def resolve_baseline_decoder(
    requested_name: str | None,
    *,
    container_factory: Callable[[], Any] = _default_container_factory,
    fallback_decoder: Callable[[], DecoderProtocol] = MWPMDecoder,
) -> DecoderResolution:
    """Resolve a baseline decoder with resilient MWPM fallback."""
    if not requested_name:
        return DecoderResolution(
            requested_name=None,
            resolved_name="mwpm",
            decoder=fallback_decoder(),
            fallback_reason=None,
        )

    try:
        container = container_factory()
        decoder = container.get_decoder(requested_name)
        return DecoderResolution(
            requested_name=requested_name,
            resolved_name=decoder.name,
            decoder=decoder,
            fallback_reason=None,
        )
    except Exception:
        return DecoderResolution(
            requested_name=requested_name,
            resolved_name="mwpm",
            decoder=fallback_decoder(),
            fallback_reason=f"fallback_to_mwpm_decoder:{requested_name}",
        )


def resolve_threshold_decoder(
    decoder_name: str,
    *,
    container_factory: Callable[[], Any] = _default_container_factory,
) -> DecoderProtocol:
    """Resolve a threshold decoder and fail fast when requested decoder is unavailable."""
    container = container_factory()
    names = container.decoders.list()
    if decoder_name not in names:
        raise KeyError(f"Unknown threshold decoder '{decoder_name}'. Available: {names}")
    return container.get_decoder(decoder_name)
