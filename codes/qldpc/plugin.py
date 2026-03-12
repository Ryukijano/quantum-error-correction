"""qLDPC plugin placeholder with explicit parity-check schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .clustered_cyclic import ClusteredCyclicCode
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
        variant = str(config.extra_params.get("variant", "")).lower()
        if variant == "clustered_cyclic":
            num_clusters = config.extra_params.get("num_clusters")
            cluster_size = config.extra_params.get("cluster_size")
            seed = config.extra_params.get("seed")
            code = ClusteredCyclicCode(
                distance=config.distance,
                rounds=config.rounds,
                physical_error_rate=config.physical_error_rate,
                num_clusters=int(num_clusters) if num_clusters is not None else None,
                cluster_size=int(cluster_size) if cluster_size is not None else None,
                check_weight=int(config.extra_params.get("check_weight", 3)),
                seed=int(seed) if seed is not None else None,
                parallel_product_surgery=bool(config.extra_params.get("parallel_product_surgery", True)),
            )
            return code.build_circuit_string()

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
