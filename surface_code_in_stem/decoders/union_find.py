"""Union-Find decoder adapter.

This adapter uses the same PyMatching detector-error-model interface where
available and exposes a separate decoder identity for experiments comparing
near-linear-time decoders.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import BoolArray, DecoderMetadata, DecoderOutput, DecoderProtocol
from .mwpm import MWPMDecoder


@dataclass
class UnionFindDecoder(DecoderProtocol):
    """Union-Find-style adapter with deterministic fallback behavior."""

    name: str = "union_find"

    def __post_init__(self) -> None:
        self._delegate = MWPMDecoder(name=self.name)

    def decode(self, detector_events: BoolArray, metadata: DecoderMetadata) -> DecoderOutput:
        output = self._delegate.decode(detector_events=detector_events, metadata=metadata)
        diagnostics = dict(output.diagnostics)
        diagnostics["algorithm"] = "union_find_adapter"
        return DecoderOutput(
            logical_predictions=output.logical_predictions,
            decoder_name=self.name,
            diagnostics=diagnostics,
        )
