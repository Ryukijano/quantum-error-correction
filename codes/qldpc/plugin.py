"""qLDPC plugin with parity-check matrix support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .clustered_cyclic import ClusteredCyclicCode
from .parity_builder import (
    qldpc_from_parity_matrices,
    toric_code_parity,
    surface_code_parity,
    hypergraph_product,
)
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
    """qLDPC-family circuit generation with parity matrix support."""

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

        elif variant == "toric":
            size = int(config.extra_params.get("size", config.distance))
            hx, hz = toric_code_parity(size)
            circuit = qldpc_from_parity_matrices(
                hx=hx,
                hz=hz,
                rounds=config.rounds,
                p=config.physical_error_rate,
            )
            return str(circuit)

        elif variant == "surface_from_parity":
            distance = config.distance
            hx, hz = surface_code_parity(distance)
            circuit = qldpc_from_parity_matrices(
                hx=hx,
                hz=hz,
                rounds=config.rounds,
                p=config.physical_error_rate,
            )
            return str(circuit)

        elif variant == "hypergraph_product":
            # Get classical code matrices
            h1 = config.extra_params.get("h1")
            h2 = config.extra_params.get("h2")

            if h1 is None or h2 is None:
                # Default to Hamming codes
                r1 = int(config.extra_params.get("r1", 3))
                r2 = int(config.extra_params.get("r2", 3))
                from .parity_builder import hamming_code_parity
                h1 = hamming_code_parity(r1)
                h2 = hamming_code_parity(r2)

            hx, hz = hypergraph_product(h1, h2)
            circuit = qldpc_from_parity_matrices(
                hx=hx,
                hz=hz,
                rounds=config.rounds,
                p=config.physical_error_rate,
            )
            return str(circuit)

        elif variant == "custom_parity":
            # Get custom parity matrices from config
            hx = config.extra_params.get("hx")
            hz = config.extra_params.get("hz")

            if hx is None or hz is None:
                raise ValueError(
                    "Custom parity variant requires 'hx' and 'hz' matrices in extra_params"
                )

            hx = np.array(hx, dtype=np.uint8)
            hz = np.array(hz, dtype=np.uint8)

            circuit = qldpc_from_parity_matrices(
                hx=hx,
                hz=hz,
                rounds=config.rounds,
                p=config.physical_error_rate,
            )
            return str(circuit)

        raise NotImplementedError(
            f"qLDPC variant '{variant}' is not implemented. Supported variants: "
            "'clustered_cyclic', 'toric', 'surface_from_parity', 'hypergraph_product', 'custom_parity'. "
            "For custom_parity, provide 'hx' and 'hz' matrices via extra_params."
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
            compatible_decoders=("belief_propagation", "osd", "pymatching", "union_find"),
            required_inputs=("hx", "hz"),
            syndrome_spec=self.syndrome_spec(),
        )
