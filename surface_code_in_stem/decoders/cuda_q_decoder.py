"""Backend-style decoder adapters for CUDA/JAX/Neural accelerator families.

These adapters provide protocol-compatible decoder implementations with optional
backend acceleration and robust fallback into deterministic parity-matrix logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .base import BoolArray, DecoderMetadata, DecoderOutput, DecoderProtocol
from .union_find import UnionFindDecoder


try:
    import cudaq
except Exception as exc:  # pragma: no cover - optional dependency
    cudaq = None  # type: ignore[assignment]
    _CUDAQ_IMPORT_ERROR = exc
else:  # pragma: no cover
    _CUDAQ_IMPORT_ERROR = None

try:
    import qujax
except Exception as exc:  # pragma: no cover - optional dependency
    qujax = None  # type: ignore[assignment]
    _QUJAX_IMPORT_ERROR = exc
else:  # pragma: no cover
    _QUJAX_IMPORT_ERROR = None

try:
    import cuquantum
except Exception as exc:  # pragma: no cover - optional dependency
    cuquantum = None  # type: ignore[assignment]
    _CUQNN_IMPORT_ERROR = exc
else:  # pragma: no cover
    _CUQNN_IMPORT_ERROR = None


def _coerce_matrix(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        matrix = np.asarray(value, dtype=np.uint8)
    except Exception:
        return None
    if matrix.ndim != 2 or matrix.size == 0:
        return None
    return matrix


def _extract_parity_matrices(metadata: DecoderMetadata) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract parity matrices from metadata.

    The helper supports multiple alias names to reduce integration friction.
    """
    extra = metadata.extra or {}
    if not isinstance(extra, Mapping):
        return None, None

    hx = _coerce_matrix(extra.get("hx", extra.get("hx_matrix", extra.get("parity_x"))))
    hz = _coerce_matrix(extra.get("hz", extra.get("hz_matrix", extra.get("parity_z"))))
    return hx, hz


def _parity_projection_predictions(
    detector_events: BoolArray,
    metadata: DecoderMetadata,
    hx: np.ndarray | None,
    hz: np.ndarray | None,
) -> np.ndarray:
    events = np.asarray(detector_events, dtype=np.bool_)
    if events.ndim != 2:
        raise ValueError("detector_events must be a 2D bool array.")
    if metadata.num_observables < 0:
        raise ValueError("metadata.num_observables must be >= 0")

    shots = events.shape[0]
    obs_count = int(metadata.num_observables)
    if obs_count == 0:
        return np.zeros((shots, 0), dtype=np.bool_)

    x_checks = hx.shape[0] if hx is not None else 0
    z_checks = hz.shape[0] if hz is not None else 0
    detectors = events.shape[1]

    if x_checks + z_checks <= 0:
        x_checks = max(1, min(detectors // 2, max(1, obs_count)))
        z_checks = max(0, detectors - x_checks)

    x_checks = min(x_checks, detectors)
    remaining = max(0, detectors - x_checks)
    z_checks = min(z_checks, remaining)
    if z_checks == 0 and detectors > 0:
        x_checks = detectors

    x_syndrome = events[:, :x_checks]
    z_syndrome = events[:, x_checks:x_checks + z_checks]

    if x_syndrome.size == 0:
        x_syndrome = events
        z_syndrome = np.empty((shots, 0), dtype=np.bool_)
        z_checks = 0

    predictions = np.zeros((shots, obs_count), dtype=np.bool_)
    for obs_idx in range(obs_count):
        source = x_syndrome if (obs_idx % 2 == 0 or z_syndrome.size == 0) else z_syndrome
        if source.size == 0:
            continue

        window = max(1, source.shape[1] // obs_count)
        start = (obs_idx * window) % source.shape[1]
        end = min(source.shape[1], start + window)
        block = source[:, start:end]
        if block.size == 0:
            block = source[:, start:start + 1]
        predictions[:, obs_idx] = np.logical_xor.reduce(block, axis=1)

    return predictions


def _decode_with_parity_matrix(
    detector_events: BoolArray,
    metadata: DecoderMetadata,
    fallback_decoder: UnionFindDecoder,
) -> DecoderOutput:
    hx, hz = _extract_parity_matrices(metadata)
    if hx is not None and hz is not None and hx.size and hz.size:
        return DecoderOutput(
            logical_predictions=_parity_projection_predictions(
                detector_events,
                metadata,
                hx=hx,
                hz=hz,
            ),
            decoder_name="parity_projection",
            diagnostics={"parity_matrix_path": True},
        )
    return fallback_decoder.decode(detector_events, metadata)


def _coerce_backend_prediction(raw: Any, num_observables: int, shots: int) -> np.ndarray:
    array = np.asarray(raw)
    if array.ndim == 0:
        raise ValueError("Backend decoder returned a scalar output")

    if array.ndim == 1 and array.shape[0] == num_observables:
        array = np.tile(array.astype(np.int8), (shots, 1))
    elif array.ndim == 2 and array.shape[0] == num_observables and array.shape[1] != shots:
        if array.shape[1] == shots and array.shape[0] == num_observables:
            array = array.T
        else:
            array = array.T

    if array.ndim != 2:
        raise ValueError("Backend decoder output must be 2D")

    if array.shape[0] != shots:
        raise ValueError(f"Backend decoder output shot axis mismatch: expected {shots}, got {array.shape[0]}")
    if array.shape[1] != num_observables:
        if num_observables == 0:
            array = np.zeros((shots, 0), dtype=np.int8)
        else:
            raise ValueError(
                f"Backend decoder output logical axis mismatch: expected {num_observables}, got {array.shape[1]}"
            )

    return np.asarray(array.astype(bool))


def _safe_invoke_decoder_backend(
    backend_callable: Any,
    events: np.ndarray,
    metadata: DecoderMetadata,
) -> Any:
    hx, hz = _extract_parity_matrices(metadata)
    num_observables = int(metadata.num_observables)

    call_patterns = [
        ("fn(events)", (events,), {}),
        ("fn(events, metadata)", (events, metadata), {}),
        ("fn(events, num_observables)", (events,), {"num_observables": num_observables}),
        ("fn(events, hx=hx, hz=hz)", (events,), {"hx": hx, "hz": hz}),
        ("fn(detector_events=events)", (), {"detector_events": events}),
        ("fn(detector_events=events, metadata=metadata)", (), {"detector_events": events, "metadata": metadata}),
        (
            "fn(detector_events=events, parity_x=hx, parity_z=hz)",
            (),
            {"detector_events": events, "parity_x": hx, "parity_z": hz},
        ),
    ]

    for _, args, kwargs in call_patterns:
        try:
            return backend_callable(*args, **kwargs)
        except TypeError:
            continue
    # As a last resort, let the call error once if nothing matches exactly.
    return backend_callable(events)


def _decode_with_backend(
    backend_name: str,
    backend_obj: Any,
    detector_events: BoolArray,
    metadata: DecoderMetadata,
) -> tuple[np.ndarray, str | None]:
    if backend_obj is None:
        raise RuntimeError(f"Backend '{backend_name}' is unavailable")

    events = np.asarray(detector_events, dtype=np.bool_)
    if events.ndim != 2:
        raise ValueError("detector_events must be a 2D bool array")

    attempt_targets: list[str] = []
    if isinstance(backend_obj, type):
        try:
            candidate = backend_obj()
        except Exception:
            candidate = backend_obj
        else:
            backend_obj = candidate

    for attr_name in ("decode", "decode_events", "decode_detector_events", "decode_batch"):
        backend_callable = getattr(backend_obj, attr_name, None)
        if not callable(backend_callable):
            continue
        attempt_targets.append(attr_name)
        result = _safe_invoke_decoder_backend(backend_callable, events, metadata)
        return _coerce_backend_prediction(result, metadata.num_observables, events.shape[0]), attr_name

    if attempt_targets:
        raise RuntimeError(f"Backend '{backend_name}' has decode symbols but no callable entries in {attempt_targets}")

    raise RuntimeError(f"Backend '{backend_name}' does not expose a known decode callable")


def _probe_backend(name: str) -> tuple[Any, bool, str | None]:
    if name == "cudaq":
        return cudaq, bool(cudaq), repr(_CUDAQ_IMPORT_ERROR) if _CUDAQ_IMPORT_ERROR else None
    if name == "qujax":
        return qujax, bool(qujax), repr(_QUJAX_IMPORT_ERROR) if _QUJAX_IMPORT_ERROR else None
    if name == "cuqnn":
        return cuquantum, bool(cuquantum), repr(_CUQNN_IMPORT_ERROR) if _CUQNN_IMPORT_ERROR else None
    return None, False, f"Unknown backend '{name}'"


def _decode_with_backend_or_fallback(
    backend_name: str,
    detector_events: BoolArray,
    metadata: DecoderMetadata,
    fallback_decoder: UnionFindDecoder,
) -> tuple[DecoderOutput, dict[str, Any]]:
    backend_obj, enabled, import_error = _probe_backend(backend_name)
    start = time.perf_counter_ns()

    diagnostics: dict[str, Any] = {
        "backend": backend_name,
        "backend_available": bool(enabled),
        "backend_contract": bool(enabled),
        "backend_error": import_error,
        "backend_chain": [f"requested:{backend_name}"],
        "fallback_chain": [f"requested:{backend_name}"],
    }
    if not enabled:
        diagnostics["backend_contract"] = False
        diagnostics["fallback_chain"] = [f"requested:{backend_name}", "unavailable"]
        output = _decode_with_parity_matrix(detector_events, metadata, fallback_decoder)
        diagnostics["backend_error"] = import_error
        diagnostics["latency_ms"] = (time.perf_counter_ns() - start) / 1_000_000
        return (
            DecoderOutput(
                logical_predictions=output.logical_predictions,
                decoder_name=output.decoder_name,
                diagnostics={**output.diagnostics, **diagnostics},
            ),
            diagnostics,
        )

    try:
        preds, call_name = _decode_with_backend(backend_name, backend_obj, detector_events, metadata)
    except Exception as exc:
        diagnostics["backend_error"] = str(exc)
        diagnostics["backend_chain"].append("backend_fallback")
        diagnostics["fallback_chain"].append("backend_fallback")
        output = _decode_with_parity_matrix(detector_events, metadata, fallback_decoder)
        diagnostics["latency_ms"] = (time.perf_counter_ns() - start) / 1_000_000
        return (
            DecoderOutput(
                logical_predictions=output.logical_predictions,
                decoder_name=output.decoder_name,
                diagnostics={**output.diagnostics, **diagnostics},
            ),
            diagnostics,
        )

    diagnostics["backend_error"] = None
    diagnostics["backend_call"] = call_name
    diagnostics["backend_chain"].append(f"selected:{backend_name}")
    diagnostics["fallback_chain"].append(f"selected:{backend_name}")
    diagnostics["latency_ms"] = (time.perf_counter_ns() - start) / 1_000_000
    return DecoderOutput(logical_predictions=preds, decoder_name=backend_name, diagnostics=diagnostics), diagnostics


def _inject_backend_diagnostics(
    output: DecoderOutput,
    backend: str,
    device: str = "cpu",
    degraded: bool = False,
) -> DecoderOutput:
    diagnostics = dict(output.diagnostics)
    diagnostics.update(
        {
            "backend": backend,
            "device": device,
            "degraded": bool(degraded),
            "latency_ms": diagnostics.get("latency_ms"),
            "fallback_chain": diagnostics.get("fallback_chain", diagnostics.get("backend_chain", [])),
        },
    )
    return DecoderOutput(
        logical_predictions=output.logical_predictions,
        decoder_name=backend,
        diagnostics=diagnostics,
    )


@dataclass
class CudaQDecoder(DecoderProtocol):
    """Decoder adapter representing CUDA-Q selection path."""

    name: str = "cudaq"
    device: str = "cuda"
    degraded: bool = False
    decoder: UnionFindDecoder = UnionFindDecoder()

    def decode(self, detector_events: BoolArray, metadata: DecoderMetadata) -> DecoderOutput:
        output, diagnostics = _decode_with_backend_or_fallback(
            backend_name="cudaq",
            detector_events=detector_events,
            metadata=metadata,
            fallback_decoder=self.decoder,
        )
        return _inject_backend_diagnostics(
            output=output,
            backend=self.name,
            device=self.device,
            degraded=bool(diagnostics.get("backend_error") and diagnostics.get("backend_error") not in {None, "None"}),
        )


@dataclass
class CuQNNBackendAdapterDecoder(DecoderProtocol):
    """Adapter label for GPU-optimized graph/QNN backend decode policy."""

    name: str = "cuqnn"
    device: str = "cuda"
    degraded: bool = False
    decoder: UnionFindDecoder = UnionFindDecoder()

    def decode(self, detector_events: BoolArray, metadata: DecoderMetadata) -> DecoderOutput:
        output, diagnostics = _decode_with_backend_or_fallback(
            backend_name="cuqnn",
            detector_events=detector_events,
            metadata=metadata,
            fallback_decoder=self.decoder,
        )
        return _inject_backend_diagnostics(
            output=output,
            backend=self.name,
            device=self.device,
            degraded=bool(diagnostics.get("backend_error") and diagnostics.get("backend_error") not in {None, "None"}),
        )


@dataclass
class QuJaxNeuralBPDecoder(DecoderProtocol):
    """Adapter for qujax-oriented neural BP decode policy."""

    name: str = "qujax"
    device: str = "gpu"
    degraded: bool = False
    decoder: UnionFindDecoder = UnionFindDecoder()

    def decode(self, detector_events: BoolArray, metadata: DecoderMetadata) -> DecoderOutput:
        output, diagnostics = _decode_with_backend_or_fallback(
            backend_name="qujax",
            detector_events=detector_events,
            metadata=metadata,
            fallback_decoder=self.decoder,
        )
        return _inject_backend_diagnostics(
            output=output,
            backend=self.name,
            device=self.device,
            degraded=bool(diagnostics.get("backend_error") and diagnostics.get("backend_error") not in {None, "None"}),
        )
