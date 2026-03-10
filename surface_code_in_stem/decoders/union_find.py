"""Union-Find decoder adapter.

This is currently an experiment-facing placeholder that delegates to the MWPM
implementation while preserving a distinct decoder identity and diagnostics.
It exists so comparison code can wire in a future union-find decoder without
changing the surrounding interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import BoolArray, DecoderMetadata, DecoderOutput, DecoderProtocol
from .mwpm import MWPMDecoder


@dataclass
class UnionFindDecoder(DecoderProtocol):
    """Placeholder union-find adapter that currently delegates to MWPM."""

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
