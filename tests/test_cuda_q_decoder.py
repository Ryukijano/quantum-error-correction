from __future__ import annotations

import numpy as np

from surface_code_in_stem.decoders.base import DecoderMetadata, DecoderOutput
from surface_code_in_stem.decoders.cuda_q_decoder import (
    CudaQDecoder,
    CuQNNBackendAdapterDecoder,
    QuJaxNeuralBPDecoder,
)


class _StubFallbackDecoder:
    name = "stub"

    def __init__(self, output: np.ndarray):
        self._output = np.asarray(output, dtype=np.bool_)

    def decode(self, detector_events: np.ndarray, metadata: DecoderMetadata) -> DecoderOutput:  # noqa: ARG002
        return DecoderOutput(
            logical_predictions=self._output.copy(),
            decoder_name=self.name,
            diagnostics={"fallback": True},
        )


class _CallableBackend:
    def __init__(self, result):
        self._result = result

    def decode(self, detector_events, metadata):  # noqa: ARG002
        return self._result


class _FailingBackend:
    def decode(self, detector_events, metadata):  # noqa: ARG002
        raise RuntimeError("backend decode failed")


def test_cudaq_decoder_uses_fallback_when_backend_unavailable(monkeypatch):
    events = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    expected = np.array([[True, True], [False, False]])
    fallback = np.array([[True, True], [False, False]], dtype=bool)

    # No parity metadata so this falls back directly to union-find replacement decoder.
    metadata = DecoderMetadata(num_observables=2, seed=1)

    monkeypatch.setattr("surface_code_in_stem.decoders.cuda_q_decoder._probe_backend", lambda name: (None, False, "missing"))
    decoder = CudaQDecoder(decoder=_StubFallbackDecoder(fallback))

    output = decoder.decode(events, metadata)
    assert output.decoder_name == "cudaq"
    assert np.array_equal(output.logical_predictions, expected)
    assert output.diagnostics["degraded"] is True
    assert "requested:cudaq" in output.diagnostics["backend_chain"]
    assert output.diagnostics["backend"] == "cudaq"


def test_qujax_decoder_uses_mock_backend_and_bypasses_fallback(monkeypatch):
    events = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    backend_result = np.array([[1, 0], [0, 1]], dtype=np.bool_)
    metadata = DecoderMetadata(num_observables=2, seed=1)

    monkeypatch.setattr(
        "surface_code_in_stem.decoders.cuda_q_decoder._probe_backend",
        lambda name: (_CallableBackend(backend_result), True, None),
    )
    decoder = QuJaxNeuralBPDecoder(decoder=_StubFallbackDecoder(np.zeros((2, 2), dtype=bool)))

    output = decoder.decode(events, metadata)
    assert output.decoder_name == "qujax"
    assert np.array_equal(output.logical_predictions, backend_result)
    assert output.diagnostics["degraded"] is False
    assert output.diagnostics["backend_error"] is None


def test_cuqnn_decoder_fallback_on_backend_exception(monkeypatch):
    events = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    metadata = DecoderMetadata(num_observables=2, seed=2)
    fallback = np.array([[False, False], [False, False]], dtype=bool)

    monkeypatch.setattr(
        "surface_code_in_stem.decoders.cuda_q_decoder._probe_backend",
        lambda name: (_FailingBackend(), True, None),
    )
    decoder = CuQNNBackendAdapterDecoder(decoder=_StubFallbackDecoder(fallback))

    output = decoder.decode(events, metadata)
    assert output.decoder_name == "cuqnn"
    assert np.array_equal(output.logical_predictions, fallback)
    assert output.diagnostics["degraded"] is True
    assert output.diagnostics["backend_error"] == "backend decode failed"
    assert "backend_fallback" in output.diagnostics["backend_chain"]


def test_parity_matrix_fallback_projection(monkeypatch):
    events = np.array(
        [
            [1, 0, 1, 1],
            [0, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    metadata = DecoderMetadata(
        num_observables=2,
        seed=3,
        extra={
            "hx": [[1, 0], [0, 1]],
            "hz": [[1, 1], [0, 1]],
        },
    )

    monkeypatch.setattr(
        "surface_code_in_stem.decoders.cuda_q_decoder._probe_backend",
        lambda name: (None, False, "missing"),
    )

    decoder = CudaQDecoder(decoder=_StubFallbackDecoder(np.zeros((2, 2), dtype=bool)))
    output = decoder.decode(events, metadata)

    assert output.decoder_name == "cudaq"
    assert np.array_equal(output.logical_predictions, np.array([[True, True], [False, True]], dtype=bool))
    assert output.diagnostics["parity_matrix_path"] is True
    assert output.diagnostics["degraded"] is True
