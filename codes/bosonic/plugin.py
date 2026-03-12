"""Bosonic-code plugin with GKP surface code support."""

from __future__ import annotations

from dataclasses import dataclass

from ..interfaces import (
    CircuitGenerationConfig,
    CodeFamilyPlugin,
    DecoderCompatibilityMetadata,
    SyndromeExtractionSpec,
)
from .gkp_surface import gkp_surface_code


@dataclass(frozen=True)
class BosonicCodePlugin(CodeFamilyPlugin):
    """Plugin for bosonic code families, supporting GKP surface codes."""

    family: str = "bosonic"

    def build_circuit(self, config: CircuitGenerationConfig) -> str:
        variant = str(config.extra_params.get("variant", "gkp_surface")).lower()
        
        if variant == "gkp_surface":
            # Extract GKP specific parameters or default to standard surface code
            # with specific noise characteristics if provided.
            # For now, we map it to a standard surface code structure but
            # conceptually it represents a GKP layer.
            sigma = float(config.extra_params.get("sigma", 0.1)) # GKP squeezing parameter
            
            # Approximate logical error rate from GKP sigma for physical error rate
            # p ~ exp(-pi/sigma^2) ... simplified mapping
            p = config.physical_error_rate
            
            return gkp_surface_code(
                distance=config.distance,
                rounds=config.rounds,
                p=p,
                sigma=sigma
            ).__str__()

        raise NotImplementedError(
            f"Bosonic variant '{variant}' is not implemented. "
            "Supported variants: 'gkp_surface'."
        )

    def syndrome_spec(self) -> SyndromeExtractionSpec:
        return SyndromeExtractionSpec(
            format_name="stim_detector_sampler", # Standard Stim format
            detector_axis="measurement_index",
            observable_axis="logical_subspace_index",
            supports_separate_observables=True,
        )

    def decoder_metadata(self) -> DecoderCompatibilityMetadata:
        return DecoderCompatibilityMetadata(
            family=self.family,
            compatible_decoders=("mwpm", "pymatching"), # Compatible with standard decoders
            required_inputs=(),
            syndrome_spec=self.syndrome_spec(),
        )
