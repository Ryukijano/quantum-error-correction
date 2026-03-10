"""qLDPC plugin placeholder with explicit parity-check schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..interfaces import (
    CircuitGenerationConfig,
    CodeFamilyPlugin,
    DecoderCompatibilityMetadata,
    SyndromeExtractionSpec,
)


@dataclass(frozen=True)
class QLDPCParityCheckInput:
    """Concrete schema for qLDPC parity-check definitions."""

    hx: Any
    hz: Any


@dataclass(frozen=True)
class QLDPCCodePlugin(CodeFamilyPlugin):
    """Placeholder plugin for qLDPC-family circuit generation."""

    family: str = "qldpc"

    def build_circuit(self, config: CircuitGenerationConfig) -> str:
        _ = config
        raise NotImplementedError(
            "qLDPC circuit generation is not implemented. Required parity-check inputs: "
            "'hx' and 'hz' binary parity-check matrices provided via "
            "CircuitGenerationConfig.extra_params['parity_checks'] as QLDPCParityCheckInput."
        )

    def syndrome_spec(self) -> SyndromeExtractionSpec:
        return SyndromeExtractionSpec(
            format_name="parity_check_syndrome",
            detector_axis="check_index",
            observable_axis="logical_operator_index",
            supports_separate_observables=False,
        )

    def decoder_metadata(self) -> DecoderCompatibilityMetadata:
        return DecoderCompatibilityMetadata(
            family=self.family,
            compatible_decoders=("belief_propagation", "osd", "pymatching"),
            required_inputs=("hx", "hz"),
            syndrome_spec=self.syndrome_spec(),
        )
