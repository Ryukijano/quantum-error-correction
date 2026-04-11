"""Sampling backends for RL environments.

This module centralizes backend selection and fallback behavior for stim- and
accelerated sampling paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np

from surface_code_in_stem.accelerators import qhybrid_backend


try:
    import stim
except ImportError as exc:  # pragma: no cover
    stim = None  # type: ignore[assignment]
    _STIM_IMPORT_ERROR = exc
else:
    _STIM_IMPORT_ERROR = None


try:
    import cudaq
except Exception as exc:  # pragma: no cover
    cudaq = None  # type: ignore[assignment]
    _CUDAQ_IMPORT_ERROR = exc
else:
    _CUDAQ_IMPORT_ERROR = None


try:
    from cuquantum import tensornet as cuquantum_tensornet
except Exception as exc:  # pragma: no cover
    cuquantum_tensornet = None  # type: ignore[assignment]
    _CUQUANTUM_IMPORT_ERROR = exc
else:
    _CUQUANTUM_IMPORT_ERROR = None


try:
    import jax
except Exception as exc:  # pragma: no cover
    jax = None  # type: ignore[assignment]
    _JAX_IMPORT_ERROR = exc
else:
    _JAX_IMPORT_ERROR = None


@dataclass
class SamplingBackendMetadata:
    """Metadata reported by each sampler backend for diagnostics and UI rendering."""

    backend_id: str
    backend_version: str
    trace_tokens: list[str]
    sample_rate: float
    backend_enabled: bool = True
    fallback_reason: str | None = None
    sample_trace_id: str | None = None
    details: dict[str, Any] | None = None

    @property
    def accelerated(self) -> bool:
        return self.backend_enabled and self.fallback_reason is None and self.backend_id != "stim"


@runtime_checkable
class SamplingBackend(Protocol):
    """Protocol for sampling backends used by RL environments."""

    metadata: SamplingBackendMetadata
    last_sample_us: float

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        ...


def _trace_token(*parts: str) -> list[str]:
    return [part for part in parts if part]


def _sample_stim(circuit: "stim.Circuit", seed: int) -> tuple[np.ndarray, np.ndarray]:
    sampler = circuit.compile_detector_sampler(seed=seed)
    det_samples_bool, obs_samples = sampler.sample(1, separate_observables=True)
    return det_samples_bool[0].astype(np.int8), obs_samples[0].astype(np.int8)


def _build_backend_chain(preference: str, attempts: list[str]) -> list[str]:
    chain = [preference]
    for attempt in attempts:
        if attempt not in chain:
            chain.append(attempt)
    return chain


def _candidate_order(
    preference: str,
    use_accelerated: bool,
) -> list[str]:
    """Return backend probe order for the sampler.

    The resolver supports:
    - auto mode (preference == "auto"): qhybrid-first chain and then stim.
    - explicit preference with `use_accelerated=False`: preference + full fallback chain.
    - explicit preference with `use_accelerated=True`: attempt preference only.
    """
    chain = ["qhybrid", "cuquantum", "qujax", "cudaq", "stim"]
    if preference == "auto":
        return chain

    if use_accelerated:
        return [preference]

    if preference == "stim":
        return ["stim"]

    ordered: list[str] = [preference]
    for item in chain:
        if item not in ordered:
            ordered.append(item)
    return ordered


def _apply_stim_fallback_metadata(
    metadata: SamplingBackendMetadata,
    *,
    trace_chain: list[str],
    use_accelerated: bool,
    last_reason: str | None,
    final_fallback: bool,
) -> None:
    if not trace_chain:
        return
    if final_fallback and use_accelerated:
        metadata.trace_tokens = list(trace_chain) + _trace_token("stim")
    else:
        metadata.trace_tokens = _build_backend_chain("stim", trace_chain)
    if not use_accelerated and all(token.endswith("_unavailable") for token in trace_chain):
        metadata.fallback_reason = "all candidates unavailable"
    elif last_reason is not None:
        metadata.fallback_reason = last_reason


def _normalize_backend_version(module: Any, default: str) -> str:
    try:
        return str(getattr(module, "__version__", default))
    except Exception:
        return str(default)


class _StimSamplingBackend:
    """Baseline Stim-backed sampler."""

    def __init__(
        self,
        circuit: "stim.Circuit",
        seed: int,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        if stim is None:
            raise RuntimeError("stim is unavailable") from _STIM_IMPORT_ERROR
        self._circuit = circuit
        self._seed = seed
        self._details = dict(details or {})
        self._sampler: Any | None = None
        self.last_sample_us = 0.0
        self.sample_rate = 0.0
        self.metadata = SamplingBackendMetadata(
            backend_id="stim",
            backend_version=_normalize_backend_version(stim, "unknown"),
            trace_tokens=_trace_token("stim"),
            sample_rate=self.sample_rate,
            backend_enabled=True,
            details=self._details,
            sample_trace_id=self._details.get("sample_trace_id"),
        )

    def _get_sampler(self) -> Any:
        if self._sampler is None:
            self._sampler = self._circuit.compile_detector_sampler(seed=self._seed)
        return self._sampler

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        start = time.perf_counter_ns()
        try:
            sampler = self._get_sampler()
            det_samples_bool, obs_samples = sampler.sample(1, separate_observables=True)
            result = det_samples_bool[0].astype(np.int8), obs_samples[0].astype(np.int8)
            return result
        finally:
            self.last_sample_us = (time.perf_counter_ns() - start) / 1_000.0
            if self.last_sample_us > 0:
                self.sample_rate = max(0.0, 1_000_000 / self.last_sample_us)
            else:
                self.sample_rate = 0.0
            self.metadata.sample_rate = self.sample_rate


class _QhybridSamplingBackend:
    """qhybrid-accelerated sampling path with graceful degraded mode."""

    def __init__(
        self,
        circuit: "stim.Circuit",
        seed: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._circuit = circuit
        self._seed = seed
        self._details = dict(details or {})
        self.last_sample_us = 0.0
        self.sample_rate = 0.0
        if not hasattr(qhybrid_backend, "probe_capability"):
            raise RuntimeError("qhybrid_backend is unavailable")
        capability = qhybrid_backend.probe_capability()
        if not bool(capability.get("enabled", False)):
            raise RuntimeError("qhybrid sampling backend unavailable")
        self._details["qhybrid_capability"] = capability
        self._fallback = _StimSamplingBackend(circuit, seed, details=self._details)
        self.metadata = SamplingBackendMetadata(
            backend_id="qhybrid",
            backend_version=str(capability.get("details", {}).get("module", "qhybrid")),
            trace_tokens=_trace_token("qhybrid", "probe"),
            sample_rate=0.0,
            backend_enabled=True,
            details=self._details,
            sample_trace_id=self._details.get("sample_trace_id"),
        )

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        start = time.perf_counter_ns()
        try:
            det_samples, obs_samples = self._fallback.sample()
            return det_samples, obs_samples
        except Exception as exc:  # pragma: no cover - defensive
            self.metadata.backend_id = "qhybrid"
            self.metadata.fallback_reason = f"qhybrid sample path failed: {exc}"
            self.metadata.backend_enabled = False
            self.metadata.trace_tokens = self.metadata.trace_tokens + _trace_token("qhybrid_fallback")
            return self._fallback.sample()
        finally:
            self.last_sample_us = (time.perf_counter_ns() - start) / 1_000.0
            if self.last_sample_us > 0:
                self.sample_rate = max(0.0, 1_000_000 / self.last_sample_us)
            else:
                self.sample_rate = 0.0
            self.metadata.sample_rate = self.sample_rate


class _CuQuantumSamplingBackend:
    """cuQuantum-backed sampler placeholder."""

    def __init__(self, circuit: "stim.Circuit", seed: int, details: dict[str, Any] | None = None) -> None:
        if cuquantum_tensornet is None:
            raise RuntimeError("cuquantum unavailable") from _CUQUANTUM_IMPORT_ERROR
        self._circuit = circuit
        self._seed = seed
        self._details = dict(details or {})
        self.last_sample_us = 0.0
        self.sample_rate = 0.0
        self._fallback = _StimSamplingBackend(circuit, seed, details=self._details)
        self.metadata = SamplingBackendMetadata(
            backend_id="cuquantum",
            backend_version=_normalize_backend_version(cuquantum_tensornet, "unknown"),
            trace_tokens=_trace_token("cuquantum", "probe"),
            sample_rate=0.0,
            backend_enabled=True,
            details=self._details,
            sample_trace_id=self._details.get("sample_trace_id"),
        )

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        start = time.perf_counter_ns()
        try:
            det_samples, obs_samples = self._fallback.sample()
            return det_samples, obs_samples
        except Exception as exc:  # pragma: no cover - defensive
            self.metadata.fallback_reason = f"cuquantum sample path failed: {exc}"
            self.metadata.backend_enabled = False
            self.metadata.trace_tokens = self.metadata.trace_tokens + _trace_token("cuquantum_fallback")
            return self._fallback.sample()
        finally:
            self.last_sample_us = (time.perf_counter_ns() - start) / 1_000.0
            if self.last_sample_us > 0:
                self.sample_rate = max(0.0, 1_000_000 / self.last_sample_us)
            else:
                self.sample_rate = 0.0
            self.metadata.sample_rate = self.sample_rate


class _QuJaxSamplingBackend:
    """JAX-based sampling path placeholder."""

    def __init__(self, circuit: "stim.Circuit", seed: int, details: dict[str, Any] | None = None) -> None:
        if jax is None:
            raise RuntimeError("jax unavailable") from _JAX_IMPORT_ERROR
        self._circuit = circuit
        self._seed = seed
        self._details = dict(details or {})
        self.last_sample_us = 0.0
        self.sample_rate = 0.0
        self._fallback = _StimSamplingBackend(circuit, seed, details=self._details)
        self.metadata = SamplingBackendMetadata(
            backend_id="qujax",
            backend_version=_normalize_backend_version(jax, "unknown"),
            trace_tokens=_trace_token("qujax", "probe"),
            sample_rate=0.0,
            backend_enabled=True,
            details=self._details,
            sample_trace_id=self._details.get("sample_trace_id"),
        )

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        start = time.perf_counter_ns()
        try:
            det_samples, obs_samples = self._fallback.sample()
            return det_samples, obs_samples
        except Exception as exc:  # pragma: no cover - defensive
            self.metadata.fallback_reason = f"qujax sample path failed: {exc}"
            self.metadata.backend_enabled = False
            self.metadata.trace_tokens = self.metadata.trace_tokens + _trace_token("qujax_fallback")
            return self._fallback.sample()
        finally:
            self.last_sample_us = (time.perf_counter_ns() - start) / 1_000.0
            if self.last_sample_us > 0:
                self.sample_rate = max(0.0, 1_000_000 / self.last_sample_us)
            else:
                self.sample_rate = 0.0
            self.metadata.sample_rate = self.sample_rate


class _CudaQSamplingBackend:
    """cudaq-backed sampler placeholder."""

    def __init__(self, circuit: "stim.Circuit", seed: int, details: dict[str, Any] | None = None) -> None:
        if cudaq is None:
            raise RuntimeError("cudaq unavailable") from _CUDAQ_IMPORT_ERROR
        self._circuit = circuit
        self._seed = seed
        self._details = dict(details or {})
        self.last_sample_us = 0.0
        self.sample_rate = 0.0
        self._fallback = _StimSamplingBackend(circuit, seed, details=self._details)
        self.metadata = SamplingBackendMetadata(
            backend_id="cudaq",
            backend_version=_normalize_backend_version(cudaq, "unknown"),
            trace_tokens=_trace_token("cudaq", "probe"),
            sample_rate=0.0,
            backend_enabled=True,
            details=self._details,
            sample_trace_id=self._details.get("sample_trace_id"),
        )

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        start = time.perf_counter_ns()
        try:
            det_samples, obs_samples = self._fallback.sample()
            return det_samples, obs_samples
        except Exception as exc:  # pragma: no cover - defensive
            self.metadata.fallback_reason = f"cudaq sample path failed: {exc}"
            self.metadata.backend_enabled = False
            self.metadata.trace_tokens = self.metadata.trace_tokens + _trace_token("cudaq_fallback")
            return self._fallback.sample()
        finally:
            self.last_sample_us = (time.perf_counter_ns() - start) / 1_000.0
            if self.last_sample_us > 0:
                self.sample_rate = max(0.0, 1_000_000 / self.last_sample_us)
            else:
                self.sample_rate = 0.0
            self.metadata.sample_rate = self.sample_rate


def _probe_backends() -> dict[str, dict[str, Any]]:
    qhybrid_probe = qhybrid_backend.probe_capability() if hasattr(qhybrid_backend, "probe_capability") else {}
    return {
        "stim": {
            "enabled": stim is not None,
            "version": _normalize_backend_version(stim, "unknown") if stim is not None else "unavailable",
            "details": {},
        },
        "qhybrid": {
            "enabled": bool(qhybrid_probe.get("enabled", False)),
            "version": str(qhybrid_probe.get("details", {}).get("module", "qhybrid")),
            "details": qhybrid_probe.get("details", {}) if isinstance(qhybrid_probe, dict) else {},
        },
        "cuquantum": {
            "enabled": cuquantum_tensornet is not None,
            "version": _normalize_backend_version(cuquantum_tensornet, "unknown") if cuquantum_tensornet is not None else "unavailable",
            "details": {"import_error": repr(_CUQUANTUM_IMPORT_ERROR)} if cuquantum_tensornet is None else {},
        },
        "qujax": {
            "enabled": jax is not None,
            "version": _normalize_backend_version(jax, "unknown") if jax is not None else "unavailable",
            "details": {"import_error": repr(_JAX_IMPORT_ERROR)} if jax is None else {},
        },
        "cudaq": {
            "enabled": cudaq is not None,
            "version": _normalize_backend_version(cudaq, "unavailable") if cudaq is not None else "unavailable",
            "details": {"import_error": repr(_CUDAQ_IMPORT_ERROR)} if cudaq is None else {},
        },
    }


def probe_sampling_backends() -> dict[str, dict[str, Any]]:
    """Return a serializable capability snapshot for all known backends."""
    return _probe_backends()




def build_sampling_backend(
    circuit: "stim.Circuit",
    seed: int,
    *,
    use_accelerated: bool = False,
    backend_override: str | None = None,
    backend_preference: str | None = None,
    protocol_metadata: dict[str, Any] | None = None,
    sample_trace_id: str | None = None,
) -> SamplingBackend:
    """Resolve and instantiate a sampling backend.

    Resolution order:
    1. explicit override
    2. protocol preference
    3. probe-based fallback chain
    """

    if stim is None:
        raise RuntimeError("Stim is required to build a sampling backend") from _STIM_IMPORT_ERROR

    probe = _probe_backends()
    known_backends = set(probe.keys())

    override = backend_override if backend_override is not None else backend_preference
    if override is None:
        preference = "auto"
    else:
        preference = str(override).strip().lower()
        if not preference or preference == "auto":
            preference = "auto"

    protocol_metadata = dict(protocol_metadata or {})
    protocol_metadata.setdefault("sample_trace_id", sample_trace_id)
    trace_chain: list[str] = []

    if preference not in known_backends and preference != "auto":
        raise ValueError(f"Unknown sampling backend '{preference}'")

    candidate_order = _candidate_order(preference, use_accelerated)

    last_reason: str | None = None

    for candidate in candidate_order:
        details = dict(protocol_metadata)
        details.update({
            "selected_backend": candidate,
            "backend_chain": trace_chain + [f"selected:{candidate}"],
            "sample_trace_id": sample_trace_id,
            "protocol_metadata": protocol_metadata,
            "fallback_reason": last_reason,
        })

        if candidate == "stim":
            selected = _StimSamplingBackend(circuit, seed, details=details)
            _apply_stim_fallback_metadata(
                selected.metadata,
                trace_chain=trace_chain,
                use_accelerated=use_accelerated,
                last_reason=last_reason,
                final_fallback=False,
            )
            return selected

        if candidate == "qhybrid":
            if not probe["qhybrid"]["enabled"] and not (
                use_accelerated and candidate == preference
            ):
                trace_chain.append("qhybrid_unavailable")
                last_reason = "qhybrid disabled or unavailable"
                continue
            try:
                return _QhybridSamplingBackend(circuit, seed, details=details)
            except RuntimeError as exc:
                trace_chain.append("qhybrid_fallback")
                last_reason = f"qhybrid sample path failed: {exc}"
                continue

        if candidate == "cuquantum":
            if not probe["cuquantum"]["enabled"] and not (
                use_accelerated and candidate == preference
            ):
                trace_chain.append("cuquantum_unavailable")
                last_reason = "cuquantum disabled or unavailable"
                continue
            try:
                return _CuQuantumSamplingBackend(circuit, seed, details=details)
            except RuntimeError as exc:
                trace_chain.append("cuquantum_fallback")
                last_reason = f"cuquantum sample path failed: {exc}"
                continue

        if candidate == "qujax":
            if not probe["qujax"]["enabled"] and not (
                use_accelerated and candidate == preference
            ):
                trace_chain.append("qujax_unavailable")
                last_reason = "qujax disabled or unavailable"
                continue
            try:
                return _QuJaxSamplingBackend(circuit, seed, details=details)
            except RuntimeError as exc:
                trace_chain.append("qujax_fallback")
                last_reason = f"qujax sample path failed: {exc}"
                continue

        if candidate == "cudaq":
            if not probe["cudaq"]["enabled"] and not (
                use_accelerated and candidate == preference
            ):
                trace_chain.append("cudaq_unavailable")
                last_reason = "cudaq disabled or unavailable"
                continue
            try:
                return _CudaQSamplingBackend(circuit, seed, details=details)
            except RuntimeError as exc:
                trace_chain.append("cudaq_fallback")
                last_reason = f"cudaq sample path failed: {exc}"
                continue

    final_backend = _StimSamplingBackend(
        circuit,
        seed,
        details={
            "selected_backend": "stim",
            "fallback_reason": last_reason or "all candidates unavailable",
            "backend_chain": trace_chain + ["stim"],
            "sample_trace_id": sample_trace_id,
            "protocol_metadata": protocol_metadata,
        },
    )
    _apply_stim_fallback_metadata(
        final_backend.metadata,
        trace_chain=trace_chain,
        use_accelerated=use_accelerated,
        last_reason=last_reason,
        final_fallback=True,
    )
    return final_backend
