"""Dual-rail erasure-code placeholder with explicit input schema."""

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
class DualRailErasureParityInput:
    """Concrete schema for dual-rail erasure parity-check inputs."""

    parity_check: Any
    erasure_map: Any


@dataclass(frozen=True)
class DualRailErasureCodePlugin(CodeFamilyPlugin):
    """Placeholder plugin for dual-rail erasure-code circuit generation."""

    family: str = "dual_rail_erasure"

    def build_circuit(self, config: CircuitGenerationConfig) -> str:
        _ = config
        raise NotImplementedError(
            "Dual-rail erasure circuit generation is not implemented. Required parity-check "
            "inputs: 'parity_check' and 'erasure_map' provided via "
            "CircuitGenerationConfig.extra_params['parity_inputs'] as DualRailErasureParityInput."
        )

    def syndrome_spec(self) -> SyndromeExtractionSpec:
        return SyndromeExtractionSpec(
            format_name="erasure_aware_syndrome",
            detector_axis="parity_check_index",
            observable_axis="logical_rail_index",
            supports_separate_observables=False,
        )

    def decoder_metadata(self) -> DecoderCompatibilityMetadata:
        return DecoderCompatibilityMetadata(
            family=self.family,
            compatible_decoders=("erasure_matching", "belief_propagation_erasure"),
            required_inputs=("parity_check", "erasure_map"),
            syndrome_spec=self.syndrome_spec(),
        )
