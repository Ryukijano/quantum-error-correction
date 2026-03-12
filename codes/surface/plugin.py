"""Surface-code plugin wrappers around existing builders."""

from __future__ import annotations

from dataclasses import dataclass

from surface_code_in_stem.dynamic import (
    hexagonal_surface_code,
    iswap_surface_code,
    walking_surface_code,
    xyz2_hexagonal_code,
)
from surface_code_in_stem.surface_code import surface_code_circuit_string

from ..interfaces import (
    CircuitGenerationConfig,
    CodeFamilyPlugin,
    DecoderCompatibilityMetadata,
    SyndromeExtractionSpec,
)


@dataclass(frozen=True)
class SurfaceCodePlugin(CodeFamilyPlugin):
    """Plugin providing access to static and dynamic surface-code builders."""

    family: str = "surface"

    def build_circuit(self, config: CircuitGenerationConfig) -> str:
        variant = str(config.extra_params.get("variant", "static")).lower()
        builders = {
            "static": surface_code_circuit_string,
            "hexagonal": hexagonal_surface_code,
            "walking": walking_surface_code,
            "iswap": iswap_surface_code,
            "xyz2": xyz2_hexagonal_code,
        }
        try:
            builder = builders[variant]
        except KeyError as exc:
            valid = ", ".join(sorted(builders))
            raise ValueError(f"Unknown surface variant '{variant}'. Expected one of: {valid}.") from exc

        circuit = builder(config.distance, config.rounds, config.physical_error_rate)
        return circuit if isinstance(circuit, str) else str(circuit)

    def syndrome_spec(self) -> SyndromeExtractionSpec:
        return SyndromeExtractionSpec(
            format_name="stim_detector_sampler",
            detector_axis="detector_index",
            observable_axis="observable_index",
            supports_separate_observables=True,
        )

    def decoder_metadata(self) -> DecoderCompatibilityMetadata:
        return DecoderCompatibilityMetadata(
            family=self.family,
            compatible_decoders=("pymatching", "union_find", "custom_stim_decoder"),
            required_inputs=("stim_circuit",),
            syndrome_spec=self.syndrome_spec(),
        )
