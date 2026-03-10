"""Bosonic-code plugin placeholder."""

from __future__ import annotations

from dataclasses import dataclass

from ..interfaces import (
    CircuitGenerationConfig,
    CodeFamilyPlugin,
    DecoderCompatibilityMetadata,
    SyndromeExtractionSpec,
)


@dataclass(frozen=True)
class BosonicCodePlugin(CodeFamilyPlugin):
    """Minimal plugin scaffold for bosonic code families."""

    family: str = "bosonic"

    def build_circuit(self, config: CircuitGenerationConfig) -> str:
        _ = config
        raise NotImplementedError(
            "Bosonic circuit generation is not implemented. Required inputs depend on the code "
            "variant and should include a mode Hamiltonian specification in extra_params."
        )

    def syndrome_spec(self) -> SyndromeExtractionSpec:
        return SyndromeExtractionSpec(
            format_name="bosonic_syndrome",
            detector_axis="measurement_index",
            observable_axis="logical_subspace_index",
            supports_separate_observables=False,
        )

    def decoder_metadata(self) -> DecoderCompatibilityMetadata:
        return DecoderCompatibilityMetadata(
            family=self.family,
            compatible_decoders=("max_likelihood", "particle_filter"),
            required_inputs=("mode_hamiltonian",),
            syndrome_spec=self.syndrome_spec(),
        )
