"""Dual-rail erasure-code plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..interfaces import (
    CircuitGenerationConfig,
    CodeFamilyPlugin,
    DecoderCompatibilityMetadata,
    SyndromeExtractionSpec,
)
from .erasure_surface import erasure_surface_code


@dataclass(frozen=True)
class DualRailErasureParityInput:
    """Concrete schema for dual-rail erasure parity-check inputs."""
    parity_check: Any
    erasure_map: Any


@dataclass(frozen=True)
class DualRailErasureCodePlugin(CodeFamilyPlugin):
    """Plugin for dual-rail erasure-code circuit generation."""

    family: str = "dual_rail_erasure"

    def build_circuit(self, config: CircuitGenerationConfig) -> str:
        variant = str(config.extra_params.get("variant", "erasure_surface")).lower()
        
        if variant == "erasure_surface":
            erasure_prob = float(config.extra_params.get("erasure_probability", 0.0))
            return erasure_surface_code(
                distance=config.distance,
                rounds=config.rounds,
                p=config.physical_error_rate,
                erasure_prob=erasure_prob
            ).__str__()

        raise NotImplementedError(
            f"Dual-rail variant '{variant}' is not implemented. "
            "Supported variants: 'erasure_surface'."
        )

    def syndrome_spec(self) -> SyndromeExtractionSpec:
        return SyndromeExtractionSpec(
            format_name="stim_detector_sampler",
            detector_axis="parity_check_index",
            observable_axis="logical_rail_index",
            supports_separate_observables=True,
        )

    def decoder_metadata(self) -> DecoderCompatibilityMetadata:
        return DecoderCompatibilityMetadata(
            family=self.family,
            compatible_decoders=("mwpm", "pymatching"),
            required_inputs=(),
            syndrome_spec=self.syndrome_spec(),
        )
