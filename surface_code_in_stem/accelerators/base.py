"""Contracts for acceleration backends and execution metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class AccelerationMetadata:
    """Metadata describing one accelerator invocation."""

    name: str
    enabled: bool
    degraded: bool = False
    reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "degraded": self.degraded,
            "reason": self.reason,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class AcceleratorResult:
    """Return type for an accelerator call with execution metadata."""

    state: np.ndarray
    metadata: AccelerationMetadata


@runtime_checkable
class NoiseAccelerator(Protocol):
    """Protocol for one- and two-ket-path accelerator implementations."""

    name: str
    metadata: AccelerationMetadata

    def apply_pauli_channel(
        self,
        psi: np.ndarray,
        *,
        n_qubits: int,
        target_qubit: int,
        probs: list[float] | np.ndarray,
        seed: int = 42,
    ) -> AcceleratorResult:
        """Apply single-qubit Pauli channel to a statevector."""

    def apply_kraus_1q(
        self,
        rho: np.ndarray,
        *,
        n_qubits: int,
        target_qubit: int,
        kraus_ops: np.ndarray,
    ) -> AcceleratorResult:
        """Apply single-qubit Kraus channel to a density matrix."""
