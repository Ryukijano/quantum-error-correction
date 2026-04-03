"""QEC circuit builder implementations.

Provides concrete implementations of the CircuitBuilder protocol
for various quantum error correction codes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import stim

if TYPE_CHECKING:
    from syndrome_net import CircuitSpec

from syndrome_net import CircuitBuilder, CircuitSpec, InvalidSpecError


class SurfaceCodeBuilder(CircuitBuilder):
    """Standard surface code circuit builder.
    
    Implements the standard rotated surface code with X and Z stabilizers
    on a square lattice.
    """
    
    @property
    def name(self) -> str:
        return "surface"
    
    @property
    def supported_distances(self) -> list[int]:
        return [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    
    def build(self, spec: CircuitSpec) -> stim.Circuit:
        """Build a standard surface code circuit.
        
        Args:
            spec: Circuit specification (distance, rounds, error_probability)
            
        Returns:
            Stim circuit with detectors and observables
        """
        if spec.distance not in self.supported_distances:
            raise InvalidSpecError(
                spec,
                f"Distance {spec.distance} not supported. "
                f"Supported: {self.supported_distances}"
            )
        
        # Use existing surface code implementation
        from surface_code_in_stem.surface_code import surface_code_circuit_string
        
        circuit_str = surface_code_circuit_string(
            distance=spec.distance,
            rounds=spec.rounds,
            p=spec.error_probability
        )
        return stim.Circuit(circuit_str)


class HexagonalCodeBuilder(CircuitBuilder):
    """Hexagonal dynamic surface code builder.
    
    Implements the degree-3 connectivity hexagonal surface code
    inspired by Morvan et al.'s architecture.
    """
    
    @property
    def name(self) -> str:
        return "hexagonal"
    
    @property
    def supported_distances(self) -> list[int]:
        return [3, 5, 7, 9, 11]
    
    @property
    def is_dynamic(self) -> bool:
        return True
    
    def build(self, spec: CircuitSpec) -> stim.Circuit:
        """Build a hexagonal surface code circuit.
        
        Uses alternating stabilizer footprints to model degree-3 connectivity.
        """
        if spec.distance not in self.supported_distances:
            raise InvalidSpecError(
                spec,
                f"Distance {spec.distance} not supported"
            )
        
        from surface_code_in_stem.dynamic import hexagonal_surface_code
        
        circuit_str = hexagonal_surface_code(
            distance=spec.distance,
            rounds=spec.rounds,
            p=spec.error_probability
        )
        return stim.Circuit(circuit_str)


class WalkingCodeBuilder(CircuitBuilder):
    """Walking dynamic surface code builder.
    
    Implements the walking code where each cycle swaps which sublattice
    receives a reset, refreshing every physical qubit every other round.
    """
    
    @property
    def name(self) -> str:
        return "walking"
    
    @property
    def supported_distances(self) -> list[int]:
        return [3, 5, 7, 9, 11]
    
    @property
    def is_dynamic(self) -> bool:
        return True
    
    def build(self, spec: CircuitSpec) -> stim.Circuit:
        """Build a walking surface code circuit."""
        if spec.distance not in self.supported_distances:
            raise InvalidSpecError(
                spec,
                f"Distance {spec.distance} not supported"
            )
        
        from surface_code_in_stem.dynamic import walking_surface_code
        
        circuit_str = walking_surface_code(
            distance=spec.distance,
            rounds=spec.rounds,
            p=spec.error_probability
        )
        return stim.Circuit(circuit_str)


class ISwapCodeBuilder(CircuitBuilder):
    """iSwap-based dynamic surface code builder."""
    
    @property
    def name(self) -> str:
        return "iswap"
    
    @property
    def supported_distances(self) -> list[int]:
        return [3, 5, 7, 9, 11]
    
    @property
    def is_dynamic(self) -> bool:
        return True
    
    def build(self, spec: CircuitSpec) -> stim.Circuit:
        """Build an iSwap surface code circuit."""
        if spec.distance not in self.supported_distances:
            raise InvalidSpecError(
                spec,
                f"Distance {spec.distance} not supported"
            )
        
        from surface_code_in_stem.dynamic import iswap_surface_code
        
        circuit_str = iswap_surface_code(
            distance=spec.distance,
            rounds=spec.rounds,
            p=spec.error_probability
        )
        return stim.Circuit(circuit_str)


class XYZ2HexagonalBuilder(CircuitBuilder):
    """XYZ2-inspired concatenated hexagonal code builder.
    
    Combines an inner YZZY-like hexagonal surface code with an outer
    phase-flip repetition layer.
    """
    
    @property
    def name(self) -> str:
        return "xyz2"
    
    @property
    def supported_distances(self) -> list[int]:
        return [3, 5, 7, 9, 11]
    
    @property
    def is_dynamic(self) -> bool:
        return True
    
    def build(self, spec: CircuitSpec) -> stim.Circuit:
        """Build an XYZ2 hexagonal code circuit."""
        if spec.distance not in self.supported_distances:
            raise InvalidSpecError(
                spec,
                f"Distance {spec.distance} not supported"
            )
        
        from surface_code_in_stem.dynamic import xyz2_hexagonal_code
        
        return xyz2_hexagonal_code(
            distance=spec.distance,
            rounds=spec.rounds,
            p=spec.error_probability
        )


class FloquetCodeBuilder(CircuitBuilder):
    """Floquet topological code builder (placeholder for future implementation).
    
    Floquet codes use dynamic stabilizer measurements that change
    in time, offering potential advantages for certain hardware.
    """
    
    @property
    def name(self) -> str:
        return "floquet"
    
    @property
    def supported_distances(self) -> list[int]:
        return [3, 5, 7]
    
    @property
    def is_dynamic(self) -> bool:
        return True
    
    def build(self, spec: CircuitSpec) -> stim.Circuit:
        """Build a Floquet code circuit.
        
        Raises:
            NotImplementedError: Floquet codes are not yet implemented
        """
        raise NotImplementedError(
            "Floquet code builder is a placeholder. "
            "Implementation will follow the Hastings-Haah construction."
        )
