"""Ising pre-decoder adapter for surface-code decoders.

The adapter preserves the shared decoder protocol while executing an optional
pre-decoding stage and then deferring to an existing deterministic fallback
decoder (MWPM). When pre-decoding is unavailable, unsupported, or fails, the
adapter falls back cleanly to MWPM and emits deterministic diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from .base import BoolArray, DecoderMetadata, DecoderOutput, DecoderProtocol
from .mwpm import MWPMDecoder


def _coerce_non_negative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed < 0:
        return None
    return parsed


def _coerce_bool_array(detector_events: BoolArray) -> np.ndarray:
    events = np.asarray(detector_events, dtype=np.bool_)
    if events.ndim != 2:
        raise ValueError("detector_events must be a 2D bool array.")
    return events


def _extract_geometry(metadata: DecoderMetadata) -> tuple[int | None, int | None]:
    extra = metadata.extra or {}
    if not isinstance(extra, Mapping):
        return None, None

    distance = _coerce_non_negative_int(extra.get("distance", extra.get("d")))
    rounds = _coerce_non_negative_int(extra.get("rounds", extra.get("num_rounds")))
    if distance is None or rounds is None:
        distance = _coerce_non_negative_int(extra.get("surface_distance"))
        rounds = _coerce_non_negative_int(extra.get("surface_rounds"))
    return distance, rounds


def _expected_surface_detectors(distance: int | None, rounds: int | None) -> int | None:
    if distance is None or rounds is None:
        return None
    if distance < 2 or rounds < 1:
        return None
    return int(2 * rounds * ((distance * distance - 1) // 2))


def _coerce_predecode_output(raw: Any, num_detectors: int, shots: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(raw, Mapping):
        if "residual" in raw:
            residual = np.asarray(raw["residual"], dtype=np.bool_)
        elif "syndrome" in raw:
            residual = np.asarray(raw["syndrome"], dtype=np.bool_)
        elif "residual_detectors" in raw:
            residual = np.asarray(raw["residual_detectors"], dtype=np.bool_)
        else:
            pre_l = np.asarray(raw.get("pre_l"), dtype=np.bool_)  # type: ignore[call-arg]
            residual = np.asarray(raw.get("residual"), dtype=np.bool_)
            if pre_l.shape == () or residual.shape == ():
                raise ValueError("predecode mapping-style output missing required fields.")
            pre_l = pre_l.astype(np.bool_).reshape(shots)
            residual = residual.astype(np.bool_)
            if residual.shape != (shots, num_detectors):
                residual = residual[:shots, :num_detectors]
            return pre_l, residual

        pre_l = np.asarray(raw.get("pre_l"), dtype=np.bool_) if "pre_l" in raw else np.zeros(shots, dtype=np.bool_)  # type: ignore[call-arg]
        residual = np.asarray(residual, dtype=np.bool_)
        if residual.ndim != 2 or residual.shape[0] != shots:
            raise ValueError(
                "predecode mapping-style output must be 2D with shape (shots, num_detectors)."
            )
        if residual.shape[1] != num_detectors:
            raise ValueError(
                f"predecode residual width mismatch: expected {num_detectors}, got {residual.shape[1]}"
            )
        if pre_l.ndim == 0:
            pre_l = np.full(shots, bool(pre_l), dtype=np.bool_)
        pre_l = np.asarray(pre_l, dtype=np.bool_).reshape(shots)
        return pre_l, residual

    if isinstance(raw, (tuple, list)):
        if len(raw) == 0:
            raise ValueError("predecode output is empty.")
        pre_l = np.asarray(raw[0], dtype=np.bool_)
        residual = np.asarray(raw[1], dtype=np.bool_) if len(raw) > 1 else np.asarray(raw[0], dtype=np.bool_)
        if pre_l.shape == (shots, num_detectors + 1):
            pre_l, residual = pre_l[:, 0], pre_l[:, 1:]
        if pre_l.ndim == 0:
            pre_l = np.full(shots, bool(pre_l), dtype=np.bool_)
        if residual.ndim != 2:
            residual = np.asarray(residual, dtype=np.bool_).reshape(shots, -1)
        pre_l = np.asarray(pre_l, dtype=np.bool_).reshape(shots)
        residual = np.asarray(residual, dtype=np.bool_)
    else:
        residual = np.asarray(raw, dtype=np.bool_)
        pre_l = np.zeros(shots, dtype=np.bool_)

    if residual.ndim != 2:
        raise ValueError("predecode residual output must be 2D.")
    if residual.shape[0] != shots:
        raise ValueError(
            f"predecode residual shot-axis mismatch: expected {shots}, got {residual.shape[0]}"
        )
    if residual.shape[1] == num_detectors + 1:
        residual = residual[:, :num_detectors]
    elif residual.shape[1] != num_detectors:
        raise ValueError(
            f"predecode residual width mismatch: expected {num_detectors}, got {residual.shape[1]}"
        )
    if pre_l.size == 1:
        pre_l = np.full(shots, bool(pre_l), dtype=np.bool_)
    return pre_l[:shots], residual[:, :num_detectors]


def _identity_predecode(events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shots = events.shape[0]
    return np.zeros(shots, dtype=np.bool_), np.asarray(events, dtype=np.bool_)


def _safe_fingerprint(path: Path) -> str:
    try:
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return ""


def _as_numpy(raw: Any) -> Any:
    if hasattr(raw, "detach"):
        try:
            return raw.detach().cpu().numpy()
        except Exception:
            pass
    if hasattr(raw, "numpy"):
        try:
            return raw.numpy()
        except Exception:
            pass
    return raw


def _predecoder_backend_is_disabled(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"disabled", "none", "off", "false"}


def _candidate_predecode_inputs(events: np.ndarray, metadata: DecoderMetadata) -> list[np.ndarray]:
    distance, rounds = _extract_geometry(metadata)
    variants: list[np.ndarray] = [np.asarray(events, dtype=np.float32)]

    # Flattened detector stream as a simple sequence feature map.
    if events.size > 0:
        variants.append(np.asarray(events, dtype=np.float32).reshape(events.shape[0], -1, 1))

    if distance is not None and rounds is not None and distance >= 1 and rounds >= 1:
        half = (distance * distance - 1) // 2
        if half > 0 and events.shape[1] == 2 * rounds * half:
            xz = np.asarray(events, dtype=np.float32).reshape(events.shape[0], 2, rounds, half)
            variants.append(xz)

            # Optional 4D shape when checkpoints expect spatial layout.
            side = int(distance)
            if side > 0:
                flat = xz.reshape(events.shape[0], -1)
                max_dims = side * side * 2 * rounds
                resized = np.zeros((events.shape[0], max_dims), dtype=np.float32)
                if flat.shape[1] >= max_dims:
                    resized[:, :max_dims] = flat[:, :max_dims]
                else:
                    resized[:, : flat.shape[1]] = flat
                variants.append(resized.reshape(events.shape[0], 2, rounds, side, side))

    unique: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in variants:
        key = candidate.shape
        if key not in seen:
            seen.add(key)
            unique.append(candidate.astype(np.float32))
    return unique


def _invoke_candidate_signatures(
    predecoder: Callable[..., Any],
    events: np.ndarray,
    metadata: DecoderMetadata,
    num_detectors: int,
) -> Any:
    last_error: Exception | None = None
    candidates = _candidate_predecode_inputs(events, metadata)
    signature_args: list[tuple[Any, ...]] = []
    for candidate in candidates:
        signature_args.extend(
            [
                (candidate,),
                (candidate, metadata),
                (candidate, metadata.extra),
                (candidate, num_detectors),
            ]
        )
    if not signature_args:
        signature_args.append((np.asarray(events, dtype=np.float32), metadata))

    for args in signature_args:
        try:
            raw = predecoder(*args)  # type: ignore[misc]
            return _as_numpy(raw)
        except TypeError:
            continue
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("predecoder callable rejected all candidate signatures")


@dataclass
class IsingDecoder(DecoderProtocol):
    """Decoder adapter that performs optional Ising pre-decoding then MWPM fallback."""

    name: str = "ising"
    fallback_decoder: MWPMDecoder = field(default_factory=MWPMDecoder)
    predecoder_backend: str = "identity"
    predecoder_callable: Callable[..., Any] | None = None
    predecoder_artifact: str | os.PathLike[str] | None = None
    predecoder_seed: int | None = None

    _backend_resolution_error: str | None = field(default=None, init=False, repr=False)
    _resolved_backend: str | None = field(default=None, init=False, repr=False)
    _predecoder: Callable[..., Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._resolved_backend = None
        if self.predecoder_callable is not None and callable(self.predecoder_callable):
            self._predecoder = self.predecoder_callable
            self._resolved_backend = "callable"
            return

        if _predecoder_backend_is_disabled(self.predecoder_backend):
            self._resolved_backend = "disabled"
            return

        artifact_path = None
        if self.predecoder_artifact is not None:
            artifact_path = Path(self.predecoder_artifact)
            if not artifact_path.exists():
                self._backend_resolution_error = f"artifact does not exist: {artifact_path}"
                self._resolved_backend = "artifact-missing"
                return
            self._resolved_backend = f"artifact:{artifact_path.name}"
            return

        self._resolved_backend = "identity"

    def _apply_metadata_config(self, metadata: DecoderMetadata) -> None:
        extra = metadata.extra
        if not isinstance(extra, Mapping):
            return

        requested_backend = extra.get("predecoder_backend")
        requested_artifact = extra.get("predecoder_artifact")
        requested_seed = extra.get("predecoder_seed")

        changed = False
        if isinstance(requested_backend, str) and requested_backend != self.predecoder_backend:
            self.predecoder_backend = requested_backend
            changed = True
        if "predecoder_artifact" in extra:
            if requested_artifact is not None and requested_artifact != self.predecoder_artifact:
                self.predecoder_artifact = requested_artifact
                changed = True
            elif requested_artifact is None and self.predecoder_artifact is not None:
                self.predecoder_artifact = None
                changed = True
        if isinstance(requested_seed, int) and requested_seed != self.predecoder_seed:
            self.predecoder_seed = requested_seed
            changed = True

        if changed:
            self._backend_resolution_error = None
            self._resolved_backend = None
            self._predecoder = None
            self.__post_init__()

    def _load_artifact_predecoder(self, num_detectors: int, metadata: DecoderMetadata) -> Callable[[np.ndarray], Any] | None:
        artifact = self.predecoder_artifact
        if artifact is None:
            return None
        artifact_path = Path(artifact)
        if not artifact_path.exists():
            self._backend_resolution_error = f"artifact does not exist: {artifact_path}"
            return None

        requested_backend = (self.predecoder_backend or "auto").strip().lower()
        suffix = artifact_path.suffix.lower()
        if requested_backend in {"auto", "torch", "torchscript", "pytorch", "pt"} or suffix in {".pt", ".pth", ".ckpt"}:
            try:
                import torch
            except Exception:
                self._backend_resolution_error = "torch is required for torch-based predecoder backends"
                return None

            try:
                loaded = torch.jit.load(str(artifact_path), map_location="cpu")
                if callable(loaded):
                    return lambda events: loaded(torch.as_tensor(events))
            except Exception:
                pass

            try:
                loaded = torch.load(str(artifact_path), map_location="cpu")
            except Exception as exc:
                self._backend_resolution_error = f"Unable to load torch artifact '{artifact_path}': {exc}"
                return None

            if callable(loaded):
                return lambda events: loaded(events)
            if isinstance(loaded, Mapping):
                for key in ("predecoder", "model", "module", "decoder"):
                    candidate = loaded.get(key)
                    if callable(candidate):
                        return lambda events: candidate(events)

            self._backend_resolution_error = (
                f"torch artifact '{artifact_path}' does not expose a callable predecoder module."
            )
            return None

        if requested_backend in {"safetensor", "safetensors", "numpy", "np"} or suffix in {".npy", ".npz", ".safetensors"}:
            try:
                if suffix == ".npz":
                    loaded = np.load(str(artifact_path), allow_pickle=False)
                    if isinstance(loaded, np.lib.npyio.NpzFile):
                        first = None
                        for value in loaded.values():
                            candidate = np.asarray(value, dtype=np.float32)
                            if candidate.size > 0:
                                first = candidate
                                break
                        if first is None:
                            self._backend_resolution_error = f"npz artifact '{artifact_path}' is empty"
                            return None
                    else:
                        first = np.asarray(loaded, dtype=np.float32)
                elif suffix == ".safetensors":
                    try:
                        from safetensors.numpy import load as load_safetensor_array
                    except Exception as exc:
                        self._backend_resolution_error = (
                            f"Failed to import safetensors for '{artifact_path}': {exc}"
                        )
                        return None
                    tensors = load_safetensor_array(str(artifact_path))
                    first_values = list(tensors.values())
                    if not first_values:
                        self._backend_resolution_error = f"No tensors in safe tensor artifact '{artifact_path}'"
                        return None
                    first = np.asarray(first_values[0], dtype=np.float32)
                else:
                    first = np.asarray(np.load(str(artifact_path), allow_pickle=False), dtype=np.float32)
            except Exception as exc:
                self._backend_resolution_error = f"Unable to read artifact '{artifact_path}': {exc}"
                return None

            if first.ndim == 1:
                first = first.reshape(1, -1)
            if first.ndim != 2:
                self._backend_resolution_error = (
                    f"Unsupported artifact shape {first.shape}; expected matrix-like transform."
                )
                return None

            weight = first
            if weight.shape[1] != num_detectors:
                self._backend_resolution_error = (
                    f"Artifact width mismatch: expected {num_detectors}, got {weight.shape[1]}"
                )
                return None

            def _matrix_predecode(events: np.ndarray) -> np.ndarray:
                flat = np.asarray(events, dtype=np.float32).reshape(events.shape[0], -1)
                return flat @ weight.T

            return _matrix_predecode

        self._backend_resolution_error = f"Unsupported predecoder backend '{requested_backend}' for artifact '{artifact_path}'"
        return None

    def _run_predecoder(
        self,
        events: np.ndarray,
        metadata: DecoderMetadata,
        num_detectors: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str | None]:
        self._apply_metadata_config(metadata)
        if self._resolved_backend in {None, "disabled", "artifact-missing"}:
            reason = self._backend_resolution_error or "predecoder disabled"
            pre_l, residual = _identity_predecode(events)
            details: dict[str, Any] = {
                "predecoder_backend": self._resolved_backend or "disabled",
                "predecoder_fallback_reason": reason,
            }
            return pre_l, residual, details, reason

        if self._predecoder is None:
            pre_l, residual = _identity_predecode(events)
            if self._resolved_backend == "identity":
                details = {"predecoder_backend": "identity", "predecoder_fallback_reason": None}
                return pre_l, residual, details, None

            if self.predecoder_artifact is None:
                details = {"predecoder_backend": self._resolved_backend, "predecoder_fallback_reason": None}
                return pre_l, residual, details, None

            self._predecoder = self._load_artifact_predecoder(num_detectors, metadata)
            if self._predecoder is None:
                details = {
                    "predecoder_backend": self._resolved_backend,
                    "predecoder_fallback_reason": self._backend_resolution_error,
                    "predecoder_artifact": str(self.predecoder_artifact),
                    "predecoder_artifact_available": True,
                }
                pre_l, residual = _identity_predecode(events)
                return pre_l, residual, details, self._backend_resolution_error

        predecoder_start = time.perf_counter_ns()
        try:
            raw = _invoke_candidate_signatures(
                self._predecoder,  # type: ignore[arg-type]
                events,
                metadata,
                num_detectors=num_detectors,
            )
            pre_l, residual = _coerce_predecode_output(raw, num_detectors=num_detectors, shots=events.shape[0])
            details = {
                "predecoder_backend": self._resolved_backend or self.predecoder_backend,
                "predecoder_fallback_reason": None,
                "predecoder_latency_ms": (time.perf_counter_ns() - predecoder_start) / 1_000_000,
            }
            return pre_l.astype(np.bool_), np.asarray(residual, dtype=np.bool_), details, None
        except Exception as exc:
            details = {
                "predecoder_backend": self._resolved_backend or self.predecoder_backend,
                "predecoder_fallback_reason": str(exc),
            }
            pre_l, residual = _identity_predecode(events)
            return pre_l, residual, details, str(exc)

    def _artifact_info(self) -> dict[str, Any]:
        if self.predecoder_artifact is None:
            return {}
        path = Path(self.predecoder_artifact)
        if not path.exists():
            return {
                "predecoder_artifact": str(path),
                "predecoder_artifact_exists": False,
            }
        return {
            "predecoder_artifact": str(path),
            "predecoder_artifact_exists": True,
            "predecoder_artifact_size": path.stat().st_size,
            "predecoder_artifact_fingerprint": _safe_fingerprint(path),
            "predecoder_artifact_backend": self._resolved_backend or "identity",
        }

    def decode(self, detector_events: BoolArray, metadata: DecoderMetadata) -> DecoderOutput:
        events = _coerce_bool_array(detector_events)
        shots, num_detectors = events.shape
        if metadata.num_observables <= 0:
            raise ValueError("metadata.num_observables must be > 0")

        start_ns = time.perf_counter_ns()
        diagnostics: dict[str, Any] = {
            "backend": self.name,
            "backend_id": self.name,
            "backend_enabled": True,
            "backend_available": True,
            "backend_contract": True,
            "backend_error": None,
            "backend_chain": [f"requested:{self.name}"],
            "fallback_chain": [f"requested:{self.name}"],
            "contract_flags": "backend_enabled,contract_met",
            "degraded": False,
            "num_shots": int(shots),
            "num_detectors": int(num_detectors),
            "num_observables": int(metadata.num_observables),
            "predecoder_seed": self.predecoder_seed,
        }
        diagnostics.update(self._artifact_info())
        diagnostics["predecoder_available"] = self._predecoder is not None or self._resolved_backend == "identity"

        distance, rounds = _extract_geometry(metadata)
        expected = _expected_surface_detectors(distance, rounds)
        if expected is not None and expected != num_detectors:
            reason = (
                f"Detector width mismatch: expected {expected} for distance={distance}, rounds={rounds}, "
                f"got {num_detectors}"
            )
            diagnostics.update(
                {
                    "backend_contract": False,
                    "predecoder_fallback_reason": reason,
                    "backend_error": reason,
                    "contract_flags": "backend_disabled,contract_fallback",
                    "degraded": True,
                    "backend_chain": diagnostics["backend_chain"] + ["predecoder_contract_failed"],
                    "fallback_chain": diagnostics["fallback_chain"] + ["predecoder_contract_failed"],
                    "predecoder_backend": self._resolved_backend or self.predecoder_backend,
                    "predecoder_available": False,
                    "predecoder_latency_ms": 0.0,
                }
            )
        pre_l, residual, predecoder_details, predecode_error = self._run_predecoder(
            events=events,
            metadata=metadata,
            num_detectors=num_detectors,
        )
        diagnostics["predecoder_backend"] = predecoder_details.get("predecoder_backend")
        diagnostics["predecoder_fallback_reason"] = predecoder_details.get("predecoder_fallback_reason")
        if "predecoder_latency_ms" in predecoder_details:
            diagnostics["predecoder_latency_ms"] = predecoder_details["predecoder_latency_ms"]

        if predecode_error is not None:
            diagnostics["backend_chain"].append("predecoder_fallback")
            diagnostics["fallback_chain"].append("predecoder_fallback")
            diagnostics["backend_error"] = predecode_error
            diagnostics["contract_flags"] = "backend_disabled,contract_fallback"
            diagnostics["degraded"] = True
            diagnostics["predecoder_available"] = False
        else:
            diagnostics["backend_chain"].append("selected:ising_predecoder")
            diagnostics["fallback_chain"].append("selected:ising_predecoder")
            diagnostics["predecoder_available"] = True

        if pre_l.shape[0] == shots:
            diagnostics["predecoder_logical_bit"] = int(np.sum(pre_l.astype(np.uint8)))

        if residual.shape != (shots, num_detectors):
            diagnostics.update(
                {
                    "backend_error": f"Residual shape malformed: {residual.shape}",
                    "backend_chain": diagnostics["backend_chain"] + ["predecoder_shape_fallback"],
                    "fallback_chain": diagnostics["fallback_chain"] + ["predecoder_shape_fallback"],
                    "contract_flags": "backend_disabled,contract_fallback",
                    "degraded": True,
                }
            )
            residual = events

        try:
            fallback_output = self.fallback_decoder.decode(residual, metadata)
            logicals = np.asarray(fallback_output.logical_predictions, dtype=np.bool_)
            diagnostics["fallback_decoder"] = fallback_output.decoder_name
            diagnostics["fallback_decoder_diagnostics"] = dict(fallback_output.diagnostics)
            diagnostics["backend_chain"].append(f"selected:{fallback_output.decoder_name}")
            diagnostics["fallback_chain"].append(f"selected:{fallback_output.decoder_name}")
            diagnostics["contract_flags"] = "backend_enabled,contract_met"
        except Exception as exc:
            diagnostics.update(
                {
                    "backend_error": str(exc),
                    "backend_chain": diagnostics["backend_chain"] + ["fallback_decoding_failed"],
                    "fallback_chain": diagnostics["fallback_chain"] + ["fallback_decoding_failed"],
                    "contract_flags": "backend_disabled,contract_fallback",
                    "degraded": True,
                }
            )
            logicals = np.zeros((shots, int(metadata.num_observables)), dtype=np.bool_)

        if logicals.shape != (shots, int(metadata.num_observables)):
            raise ValueError(
                f"Decoder returned predictions with invalid shape {logicals.shape}, expected ({shots}, {metadata.num_observables})"
            )

        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        diagnostics["latency_ms"] = latency_ms
        diagnostics["sample_us"] = latency_ms * 1000.0
        diagnostics["degraded"] = bool(diagnostics.get("backend_error") is not None)

        if "predecoder_latency_ms" not in diagnostics:
            diagnostics["predecoder_latency_ms"] = 0.0

        return DecoderOutput(
            logical_predictions=logicals,
            decoder_name=self.name,
            diagnostics=diagnostics,
        )
