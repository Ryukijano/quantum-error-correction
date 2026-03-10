"""Confidence-aware decoding utilities.

This module introduces a small data container for syndrome batches and a
weighted-MWPM decoder that can consume optional per-detector confidence values.
When confidence data is omitted, decoding behavior falls back to standard
hard-decision MWPM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class SyndromeBatch:
    """Container for batched detector outcomes.

    Attributes:
        hard_bits: Binary detector outcomes with shape ``(shots, num_detectors)``.
        confidence: Optional confidence values in ``[0, 1]`` with the same shape
            as ``hard_bits``. Larger values indicate higher trust in the
            corresponding hard detector outcome.
    """

    hard_bits: np.ndarray
    confidence: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        hard_bits = np.asarray(self.hard_bits, dtype=np.uint8)
        if hard_bits.ndim != 2:
            raise ValueError("hard_bits must be a 2D array with shape (shots, num_detectors).")
        if np.any((hard_bits != 0) & (hard_bits != 1)):
            raise ValueError("hard_bits must contain only 0/1 values.")
        object.__setattr__(self, "hard_bits", hard_bits)

        if self.confidence is None:
            return

        confidence = np.asarray(self.confidence, dtype=np.float64)
        if confidence.shape != hard_bits.shape:
            raise ValueError("confidence must have the same shape as hard_bits.")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("confidence entries must lie in [0, 1].")
        object.__setattr__(self, "confidence", confidence)


class SyndromeDecoder(Protocol):
    """Decoder interface accepting hard syndromes with optional soft features."""

    def decode_batch(self, syndromes: SyndromeBatch) -> np.ndarray:
        """Decode a batch and return logical predictions per shot."""


class WeightedMWPMDecoder:
    """Weighted MWPM decoder with per-shot confidence reweighting.

    The decoder starts from a base matching graph extracted from a
    detector-error model. If confidence is available, edge weights are adjusted
    by endpoint confidence before running MWPM:

    ``w' = w * (1 + confidence_scale * (1 - mean_endpoint_confidence))``

    so low-confidence measurements get larger effective costs.
    """

    def __init__(self, detector_error_model: object, *, confidence_scale: float = 1.0) -> None:
        if confidence_scale < 0:
            raise ValueError("confidence_scale must be non-negative.")

        try:
            import pymatching
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pymatching is required for WeightedMWPMDecoder.") from exc

        self._pymatching = pymatching
        self._confidence_scale = float(confidence_scale)
        self._base_matching = pymatching.Matching.from_detector_error_model(detector_error_model)
        self._base_graph = self._base_matching.to_networkx()

    def _confidence_adjusted_graph(self, shot_confidence: np.ndarray) -> nx.Graph:
        """Create a per-shot graph with edge weights adjusted by confidence."""

        graph = self._base_graph.copy()
        boundary_node = self._base_matching.num_detectors

        for u, v, data in graph.edges(data=True):
            base_weight = float(data["weight"])
            conf_terms = []
            if u != boundary_node:
                conf_terms.append(float(shot_confidence[u]))
            if v != boundary_node:
                conf_terms.append(float(shot_confidence[v]))

            mean_confidence = float(np.mean(conf_terms)) if conf_terms else 1.0
            scale = 1.0 + self._confidence_scale * (1.0 - mean_confidence)
            data["weight"] = base_weight * scale

        return graph

    def decode_batch(self, syndromes: SyndromeBatch) -> np.ndarray:
        """Decode each shot, optionally consuming confidence values."""

        hard = syndromes.hard_bits
        confidence = syndromes.confidence
        predictions = np.zeros((hard.shape[0],), dtype=np.uint8)

        if confidence is None:
            for shot_ix, shot in enumerate(hard):
                predictions[shot_ix] = self._base_matching.decode(shot)[0]
            return predictions

        for shot_ix, (shot, shot_confidence) in enumerate(zip(hard, confidence, strict=True)):
            weighted_graph = self._confidence_adjusted_graph(shot_confidence)
            matching = self._pymatching.Matching(weighted_graph)
            predictions[shot_ix] = matching.decode(shot)[0]

        return predictions
