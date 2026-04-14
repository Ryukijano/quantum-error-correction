"""Tests for Ising pre-decoder integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from surface_code_in_stem.decoders import DecoderMetadata, DecoderOutput, IsingDecoder
from surface_code_in_stem.decoders.ising_predecoder import (
    _candidate_predecode_inputs,
    _invoke_candidate_signatures,
)


def test_candidate_predecode_inputs_include_geometry_variants() -> None:
    events = np.zeros((4, 16), dtype=np.float32)
    metadata = DecoderMetadata(num_observables=1, extra={"distance": 3, "rounds": 2})
    candidates = _candidate_predecode_inputs(events, metadata)
    candidate_shapes = {candidate.shape for candidate in candidates}

    assert (4, 16) in candidate_shapes
    assert (4, 16, 1) in candidate_shapes
    assert (4, 2, 2, 4) in candidate_shapes
    assert (4, 2, 2, 3, 3) in candidate_shapes


def test_invoke_candidate_signatures_matches_metadata_aware_signature() -> None:
    metadata = DecoderMetadata(num_observables=1, extra={"distance": 3, "rounds": 2})
    events = np.zeros((2, 2), dtype=np.float32)

    def predecoder(payload: np.ndarray, context: DecoderMetadata) -> np.ndarray:
        assert context is metadata
        assert payload.shape == events.shape
        return np.ones((payload.shape[0], 2), dtype=np.float32)

    output = _invoke_candidate_signatures(predecoder, events, metadata, num_detectors=2)
    assert output.shape == (2, 2)
    assert np.all(output == 1)


def test_ising_decoder_loads_numpy_artifact_and_runs(tmp_path: Path) -> None:
    artifact_path = tmp_path / "ising_matrix.npy"
    np.save(artifact_path, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    class FallbackDecoder:
        name = "fallback_decoder"

        def decode(self, detector_events: np.ndarray, metadata: DecoderMetadata) -> DecoderOutput:
            predictions = np.zeros((detector_events.shape[0], metadata.num_observables), dtype=np.uint8)
            return DecoderOutput(logical_predictions=predictions, decoder_name=self.name, diagnostics={})

    decoder = IsingDecoder(
        predecoder_backend="numpy",
        predecoder_artifact=str(artifact_path),
        fallback_decoder=FallbackDecoder(),  # type: ignore[arg-type]
    )
    events = np.array([[True, False], [False, True], [True, True]], dtype=bool)
    metadata = DecoderMetadata(num_observables=1)
    output = decoder.decode(events, metadata)

    assert output.decoder_name == "ising"
    assert output.logical_predictions.shape == (3, 1)
    assert output.diagnostics["predecoder_backend"] == f"artifact:{artifact_path.name}"
    assert output.diagnostics["predecoder_fallback_reason"] is None


def test_ising_decoder_uses_metadata_runtime_overrides(tmp_path: Path) -> None:
    artifact_path = tmp_path / "ising_runtime_matrix.npy"
    np.save(artifact_path, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    class FallbackDecoder:
        name = "fallback_decoder"

        def decode(self, detector_events: np.ndarray, metadata: DecoderMetadata) -> DecoderOutput:
            predictions = np.zeros((detector_events.shape[0], metadata.num_observables), dtype=np.uint8)
            return DecoderOutput(logical_predictions=predictions, decoder_name=self.name, diagnostics={})

    decoder = IsingDecoder(
        predecoder_backend="identity",
        predecoder_artifact=None,
        fallback_decoder=FallbackDecoder(),  # type: ignore[arg-type]
    )
    events = np.array([[True, False], [False, True]], dtype=bool)
    metadata = DecoderMetadata(
        num_observables=1,
        extra={
            "predecoder_backend": "numpy",
            "predecoder_artifact": str(artifact_path),
        },
    )
    output = decoder.decode(events, metadata)

    assert output.diagnostics["predecoder_backend"] == f"artifact:{artifact_path.name}"
    assert output.diagnostics["predecoder_fallback_reason"] is None


@pytest.mark.parametrize("backend", ["disabled", "identity"])
def test_ising_decoder_disabled_backend_short_circuits(tmp_path: Path, backend: str) -> None:
    class FallbackDecoder:
        name = "fallback_decoder"

        def decode(self, detector_events: np.ndarray, metadata: DecoderMetadata) -> DecoderOutput:
            predictions = np.zeros((detector_events.shape[0], metadata.num_observables), dtype=np.uint8)
            return DecoderOutput(logical_predictions=predictions, decoder_name=self.name, diagnostics={})

    decoder_kwargs = {
        "predecoder_backend": backend,
        "fallback_decoder": FallbackDecoder(),  # type: ignore[arg-type]
    }
    if backend == "identity":
        decoder_kwargs["predecoder_artifact"] = None
    else:
        decoder_kwargs["predecoder_artifact"] = tmp_path / "missing.npy"
    decoder = IsingDecoder(**decoder_kwargs)
    events = np.array([[True, False]], dtype=bool)
    metadata = DecoderMetadata(num_observables=1)
    output = decoder.decode(events, metadata)

    if backend == "disabled":
        assert output.diagnostics["predecoder_backend"] == "disabled"
        assert output.diagnostics["predecoder_fallback_reason"] == "predecoder disabled"
    else:
        assert output.diagnostics["predecoder_backend"] == "identity"
        assert output.diagnostics["predecoder_fallback_reason"] is None
