"""Accelerator interfaces for optional runtime backends."""

from .base import AccelerationMetadata, AcceleratorResult, NoiseAccelerator
from .qhybrid_backend import (
    apply_kraus_1q,
    apply_kraus_1q_with_metadata,
    apply_pauli_channel,
    apply_pauli_channel_with_metadata,
    get_backend,
    get_backend_metadata,
    is_accelerated,
    probe_capability,
)

__all__ = [
    "AccelerationMetadata",
    "AcceleratorResult",
    "NoiseAccelerator",
    "apply_kraus_1q",
    "apply_kraus_1q_with_metadata",
    "apply_pauli_channel",
    "apply_pauli_channel_with_metadata",
    "get_backend",
    "get_backend_metadata",
    "is_accelerated",
    "probe_capability",
]
