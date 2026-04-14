"""Decoder interfaces and adapters for surface-code experiments."""

from __future__ import annotations

from typing import Any

from .base import DecoderInput, DecoderMetadata, DecoderOutput, DecoderProtocol
from .mwpm import MWPMDecoder
from .sparse_blossom import SparseBlossomDecoder
from .union_find import ConfidenceAwareUnionFindDecoder, UnionFindDecoder


def _unavailable_decoder(name: str, import_error: BaseException) -> type:
    message = f"{name} is unavailable because required optional dependency import failed: {import_error!r}"

    class _UnavailableDecoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(message)

    _UnavailableDecoder.name = name
    _UnavailableDecoder.__name__ = name
    _UnavailableDecoder.__qualname__ = name
    return _UnavailableDecoder


HAS_JAX_NEURAL_BP_DECODER = False
HAS_JAX_CONFIDENCE_DECODER = False
HAS_CUDAQ_DECODERS = False
HAS_ISING_DECODER = False

try:
    from .jax_gnn_decoder import JAXNeuralBPDecoder
except Exception as exc:  # pragma: no cover - optional dependency
    JAXNeuralBPDecoder = _unavailable_decoder("JAXNeuralBPDecoder", exc)
else:
    HAS_JAX_NEURAL_BP_DECODER = True

try:
    from .jax_confidence import JAXConfidenceDecoder, JAXConfidenceDecoderAdapter
except Exception as exc:  # pragma: no cover - optional dependency
    JAXConfidenceDecoder = _unavailable_decoder("JAXConfidenceDecoder", exc)
    JAXConfidenceDecoderAdapter = _unavailable_decoder("JAXConfidenceDecoderAdapter", exc)
    HAS_JAX_CONFIDENCE_DECODER = False
else:
    HAS_JAX_CONFIDENCE_DECODER = True

try:
    from .cuda_q_decoder import CudaQDecoder, CuQNNBackendAdapterDecoder, QuJaxNeuralBPDecoder
except Exception as exc:  # pragma: no cover - optional dependency
    CudaQDecoder = _unavailable_decoder("CudaQDecoder", exc)
    CuQNNBackendAdapterDecoder = _unavailable_decoder("CuQNNBackendAdapterDecoder", exc)
    QuJaxNeuralBPDecoder = _unavailable_decoder("QuJaxNeuralBPDecoder", exc)
    HAS_CUDAQ_DECODERS = False
else:
    HAS_CUDAQ_DECODERS = True

try:
    from .ising_predecoder import IsingDecoder
except Exception as exc:  # pragma: no cover - optional dependency
    IsingDecoder = _unavailable_decoder("IsingDecoder", exc)
else:
    HAS_ISING_DECODER = True

__all__ = [
    "DecoderInput",
    "DecoderMetadata",
    "DecoderOutput",
    "DecoderProtocol",
    "MWPMDecoder",
    "UnionFindDecoder",
    "ConfidenceAwareUnionFindDecoder",
    "SparseBlossomDecoder",
    "JAXNeuralBPDecoder",
    "JAXConfidenceDecoder",
    "JAXConfidenceDecoderAdapter",
    "IsingDecoder",
    "CudaQDecoder",
    "CuQNNBackendAdapterDecoder",
    "QuJaxNeuralBPDecoder",
    "HAS_ISING_DECODER",
]
