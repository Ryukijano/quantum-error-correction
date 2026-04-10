"""Deep module adapter for optional qhybrid-kernels acceleration.

This module enforces explicit execution metadata so callers can detect
degraded-mode behavior at runtime instead of silently masking it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import AccelerationMetadata, AcceleratorResult, NoiseAccelerator


try:
    from qhybrid_kernels import (
        apply_pauli_channel_statevector,
        apply_kraus_1q_density_matrix,
    )
except ImportError as exc:  # pragma: no cover
    _IMPORT_ERROR: Optional[Exception] = exc
    HAS_QHYBRID = False
else:
    _IMPORT_ERROR = None
    HAS_QHYBRID = True


def _noop_metadata(enabled: bool, reason: str | None) -> AccelerationMetadata:
    return AccelerationMetadata(
        name="qhybrid",
        enabled=enabled,
        degraded=not enabled,
        reason=reason,
        details={
            "module": "surface_code_in_stem.accelerators.qhybrid_backend",
            "import_error": repr(_IMPORT_ERROR) if _IMPORT_ERROR else None,
        },
    )


def _to_float_array(value: list[float] | np.ndarray | float) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class _NoopQhybridAccelerator(NoiseAccelerator):
    """Explicit degraded-mode implementation that returns a structural copy."""

    name: str = "qhybrid"
    metadata: AccelerationMetadata = _noop_metadata(False, "qhybrid_kernels unavailable")

    def apply_pauli_channel(
        self,
        psi: np.ndarray,
        *,
        n_qubits: int,
        target_qubit: int,
        probs: list[float] | np.ndarray,
        seed: int = 42,
    ) -> AcceleratorResult:
        return AcceleratorResult(state=np.asarray(psi).copy(), metadata=self.metadata)

    def apply_kraus_1q(
        self,
        rho: np.ndarray,
        *,
        n_qubits: int,
        target_qubit: int,
        kraus_ops: np.ndarray,
    ) -> AcceleratorResult:
        return AcceleratorResult(state=np.asarray(rho).copy(), metadata=self.metadata)


@dataclass(frozen=True)
class _QhybridNoiseAccelerator(NoiseAccelerator):
    """Accelerator-backed implementation."""

    name: str = "qhybrid"
    metadata: AccelerationMetadata = AccelerationMetadata(name="qhybrid", enabled=True, degraded=False)

    def apply_pauli_channel(
        self,
        psi: np.ndarray,
        *,
        n_qubits: int,
        target_qubit: int,
        probs: list[float] | np.ndarray,
        seed: int = 42,
    ) -> AcceleratorResult:
        result = apply_pauli_channel_statevector(
            np.asarray(psi),
            n_qubits=int(n_qubits),
            target_qubit=int(target_qubit),
            probs=_to_float_array(probs),
            seed=int(seed),
        )
        return AcceleratorResult(state=np.asarray(result), metadata=self.metadata)

    def apply_kraus_1q(
        self,
        rho: np.ndarray,
        *,
        n_qubits: int,
        target_qubit: int,
        kraus_ops: np.ndarray,
    ) -> AcceleratorResult:
        result = apply_kraus_1q_density_matrix(
            np.asarray(rho),
            n_qubits=int(n_qubits),
            target_qubit=int(target_qubit),
            kraus_ops=np.asarray(kraus_ops),
        )
        return AcceleratorResult(state=np.asarray(result), metadata=self.metadata)


_BACKEND: NoiseAccelerator
if HAS_QHYBRID:
    _BACKEND = _QhybridNoiseAccelerator(
        metadata=AccelerationMetadata(
            name="qhybrid",
            enabled=True,
            degraded=False,
            details={"module": "qhybrid_kernels"},
        )
    )
else:
    _BACKEND = _NoopQhybridAccelerator()


def get_backend() -> NoiseAccelerator:
    """Return the configured accelerator backend."""
    return _BACKEND


def get_backend_metadata() -> AccelerationMetadata:
    """Expose execution metadata for diagnostics and UI tooling."""
    return _BACKEND.metadata


def is_accelerated() -> bool:
    """Whether the qhybrid backend is active for this run."""
    return bool(get_backend_metadata().enabled)


def probe_capability() -> dict[str, object]:
    """Return a stable capability snapshot for UI and runtime diagnostics."""
    metadata = get_backend_metadata().as_dict()
    return {
        "name": metadata.get("name", "qhybrid"),
        "enabled": bool(metadata.get("enabled", False)),
        "degraded": bool(metadata.get("degraded", False)),
        "reason": metadata.get("reason"),
        "available": bool(HAS_QHYBRID),
        "details": dict(metadata.get("details") or {}),
        "import_error": repr(_IMPORT_ERROR) if _IMPORT_ERROR else None,
    }


def apply_pauli_channel(
    psi: np.ndarray,
    n_qubits: int,
    target_qubit: int,
    probs: list[float],
    seed: int = 42,
) -> np.ndarray:
    """Apply a single-qubit Pauli channel to a statevector.

    This keeps backwards-compatible ndarray return semantics while routing through
    a backend that records execution metadata.
    """
    return apply_pauli_channel_with_metadata(
        psi,
        n_qubits=n_qubits,
        target_qubit=target_qubit,
        probs=probs,
        seed=seed,
    ).state


def apply_kraus_1q(
    rho: np.ndarray,
    n_qubits: int,
    target_qubit: int,
    kraus_ops: np.ndarray,
) -> np.ndarray:
    """Apply a single-qubit Kraus channel to a density matrix."""
    return apply_kraus_1q_with_metadata(
        rho,
        n_qubits=n_qubits,
        target_qubit=target_qubit,
        kraus_ops=kraus_ops,
    ).state


def apply_pauli_channel_with_metadata(
    psi: np.ndarray,
    n_qubits: int,
    target_qubit: int,
    probs: list[float],
    seed: int = 42,
) -> AcceleratorResult:
    """Apply a single-qubit Pauli channel and return execution metadata."""
    return get_backend().apply_pauli_channel(
        psi,
        n_qubits=int(n_qubits),
        target_qubit=int(target_qubit),
        probs=probs,
        seed=int(seed),
    )


def apply_kraus_1q_with_metadata(
    rho: np.ndarray,
    n_qubits: int,
    target_qubit: int,
    kraus_ops: np.ndarray,
) -> AcceleratorResult:
    """Apply single-qubit Kraus channel and return execution metadata."""
    return get_backend().apply_kraus_1q(
        rho,
        n_qubits=int(n_qubits),
        target_qubit=int(target_qubit),
        kraus_ops=kraus_ops,
    )


__all__ = [
    "HAS_QHYBRID",
    "apply_pauli_channel",
    "apply_pauli_channel_with_metadata",
    "apply_kraus_1q",
    "apply_kraus_1q_with_metadata",
    "get_backend",
    "get_backend_metadata",
    "is_accelerated",
    "probe_capability",
]
