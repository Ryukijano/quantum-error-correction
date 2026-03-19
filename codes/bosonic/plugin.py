"""Bosonic-code plugin with GKP, cat code, and squeezed state support."""

from __future__ import annotations

from dataclasses import dataclass

from ..interfaces import (
    CircuitGenerationConfig,
    CodeFamilyPlugin,
    DecoderCompatibilityMetadata,
    SyndromeExtractionSpec,
)
from .gkp_surface import gkp_surface_code
from .cat_code import cat_surface_code, cat_biased_circuit
from .squeezed_state import squeezed_surface_code, squeezed_circuit_string


@dataclass(frozen=True)
class BosonicCodePlugin(CodeFamilyPlugin):
    """Plugin for bosonic code families, supporting GKP, cat, and squeezed state codes."""

    family: str = "bosonic"

    def build_circuit(self, config: CircuitGenerationConfig) -> str:
        variant = str(config.extra_params.get("variant", "gkp_surface")).lower()

        if variant == "gkp_surface":
            sigma = float(config.extra_params.get("sigma", 0.1))
            p = config.physical_error_rate

            circuit = gkp_surface_code(
                distance=config.distance,
                rounds=config.rounds,
                p=p,
                sigma=sigma
            )
            return circuit.__str__()

        elif variant == "cat_code":
            alpha = float(config.extra_params.get("alpha", 2.0))
            bias_ratio = float(config.extra_params.get("bias_ratio", 10.0))

            return cat_biased_circuit(
                distance=config.distance,
                rounds=config.rounds,
                p=config.physical_error_rate,
                alpha=alpha,
            )

        elif variant == "squeezed_state":
            squeezing_db = float(config.extra_params.get("squeezing_db", 10.0))
            squeezed_quadrature = str(config.extra_params.get("squeezed_quadrature", "p"))

            return squeezed_circuit_string(
                distance=config.distance,
                rounds=config.rounds,
                p=config.physical_error_rate,
                squeezing_db=squeezing_db,
                squeezed_quadrature=squeezed_quadrature,
            )

        raise NotImplementedError(
            f"Bosonic variant '{variant}' is not implemented. "
            "Supported variants: 'gkp_surface', 'cat_code', 'squeezed_state'."
        )

    def syndrome_spec(self) -> SyndromeExtractionSpec:
        return SyndromeExtractionSpec(
            format_name="stim_detector_sampler",
            detector_axis="measurement_index",
            observable_axis="logical_subspace_index",
            supports_separate_observables=True,
        )

    def decoder_metadata(self) -> DecoderCompatibilityMetadata:
        return DecoderCompatibilityMetadata(
            family=self.family,
            compatible_decoders=("mwpm", "pymatching", "union_find"),
            required_inputs=(),
            syndrome_spec=self.syndrome_spec(),
        )
