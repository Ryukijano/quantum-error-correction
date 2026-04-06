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


class ColorCodeStimBuilder(CircuitBuilder):
    """Colour code builder using the color-code-stim library.
    
    Supports triangular, rectangular, growing, and cult+growing patch types
    with optional superdense syndrome extraction. Uses concatenated MWPM
    decoding (6-matching: 2 per color R/G/B).
    
    Reference: Lee & Brown, Quantum 9, 1609 (2025)
    """
    
    @property
    def name(self) -> str:
        return "color_code"
    
    @property
    def supported_distances(self) -> list[int]:
        return [3, 5, 7, 9, 11, 13]
    
    def build(self, spec: CircuitSpec) -> stim.Circuit:
        """Build a colour code Stim circuit.
        
        Args:
            spec: Circuit specification. Use spec.circuit_type to select
                  patch geometry (default "tri"). Use spec.superdense for
                  superdense syndrome extraction.
        """
        try:
            from color_code_stim import ColorCode, NoiseModel as CCNoiseModel
        except ImportError as exc:
            raise ImportError(
                "color-code-stim is required for ColorCodeStimBuilder. "
                "Install with: pip install color-code-stim"
            ) from exc
        
        circuit_type = spec.circuit_type or "tri"
        
        # Validate distance constraints per circuit type
        if circuit_type in ("tri",) and spec.distance % 2 == 0:
            raise InvalidSpecError(
                spec, f"Triangular colour code requires odd distance, got {spec.distance}"
            )
        if circuit_type in ("rec", "rec_stability") and spec.distance % 2 != 0:
            raise InvalidSpecError(
                spec, f"Rectangular colour code requires even distance, got {spec.distance}"
            )
        
        noise = CCNoiseModel.uniform_circuit_noise(spec.error_probability)
        
        kwargs: dict = dict(
            d=spec.distance,
            rounds=spec.rounds,
            circuit_type=circuit_type,
            noise_model=noise,
            superdense_circuit=spec.superdense,
        )
        if spec.d2 is not None:
            kwargs["d2"] = spec.d2
        
        cc = ColorCode(**kwargs)
        return cc.circuit


class LoomColorCodeBuilder(CircuitBuilder):
    """Colour code builder using Entropica Labs' el-loom library.
    
    Creates colour codes via Loom's Eka/Block/Lattice abstractions,
    supporting custom lattice geometries, lattice surgery operations,
    and interpretation to Stim circuits.
    
    Reference: Entropica Labs, el-loom v0.4.0
    """
    
    @property
    def name(self) -> str:
        return "loom_color_code"
    
    @property
    def supported_distances(self) -> list[int]:
        return [3, 5, 7, 9]
    
    def build(self, spec: CircuitSpec) -> stim.Circuit:
        """Build a colour code via Loom's Eka pipeline.
        
        Constructs a hexagonal-lattice colour code on a triangular patch,
        interprets it to a Stim circuit with detectors and observables.
        """
        try:
            from loom.eka import Lattice, Block, Eka
            from loom.eka.operations import (
                MeasureBlockSyndromes,
                MeasureLogicalZ,
                ResetAllDataQubits,
            )
            from loom.interpreter import interpret_eka
        except ImportError as exc:
            raise ImportError(
                "el-loom is required for LoomColorCodeBuilder. "
                "Install with: pip install el-loom"
            ) from exc
        
        import numpy as _np
        
        d = spec.distance
        if d % 2 == 0:
            raise InvalidSpecError(spec, "Loom colour code requires odd distance")
        
        # Build hexagonal lattice for colour code
        sqrt3 = _np.sqrt(3)
        basis_vectors = [
            [0, 0], [0.5, -sqrt3 / 2], [1.5, -sqrt3 / 2], [2.0, 0],
            [1.0, 0],            # X ancilla
            [1.0, 0],            # Z ancilla (co-located)
            [2.5, -sqrt3 / 2],   # X ancilla
            [2.5, -sqrt3 / 2],   # Z ancilla (co-located)
        ]
        lattice_vectors = [[3, 0], [0, -sqrt3]]
        
        Lx = (d - 1) / 2 * 3
        dx = (d - 1) / 2 + 1
        dz = _np.round(_np.sqrt(3) * Lx / 2 / sqrt3 + 1)
        
        lattice = Lattice(basis_vectors, lattice_vectors, (dx, dz))
        
        # Identify data vs ancilla qubits inside the triangular patch
        coord_dict = {
            (x, y, b): (
                basis_vectors[b][0] + x * lattice_vectors[0][0] + y * lattice_vectors[1][0],
                basis_vectors[b][1] + x * lattice_vectors[0][1] + y * lattice_vectors[1][1],
            )
            for x, y, b in lattice.all_qubits()
        }
        
        def _inside_patch(q, eps=0.01):
            if q not in lattice.all_qubits():
                return False
            x, y = coord_dict[q]
            m1 = (basis_vectors[1][1]) / (basis_vectors[1][0] + 1e-12)
            m2 = (basis_vectors[3][1]) / (basis_vectors[3][0] - Lx + 1e-12)
            return y <= eps and y >= m1 * x - eps and y >= m2 * (x - Lx) - eps
        
        data_qubits = [q for q in lattice.all_qubits() if q[2] < 4 and _inside_patch(q)]
        
        if len(data_qubits) == 0:
            raise InvalidSpecError(spec, f"Loom lattice produced 0 data qubits for d={d}")
        
        # Use Loom's rotated surface code factory for syndrome circuit generation,
        # then build a colour-code Block with the lattice
        try:
            from loom_rotated_surface_code.code_factory.rotated_surface_code import RotatedSurfaceCode
            _has_rsc_factory = True
        except ImportError:
            _has_rsc_factory = False
        
        # Build Block with minimal stabilizer structure
        # For the colour code, we define X and Z stabilizers on plaquettes
        from loom.eka.stabilizer import Stabilizer
        
        stabilizers = []
        logical_x_operators = []
        logical_z_operators = []
        
        # Generate plaquette stabilizers from the lattice geometry
        # Each hexagonal face gives one X and one Z stabilizer
        inside_qubits = [q for q in lattice.all_qubits() if _inside_patch(q)]
        ancilla_qubits = [q for q in inside_qubits if q[2] >= 4]
        
        for aq in ancilla_qubits:
            pauli_type = "X" if aq[2] in (4, 6) else "Z"
            # Find neighboring data qubits
            ax, ay = coord_dict[aq]
            support = []
            for dq in data_qubits:
                dx_c, dy_c = coord_dict[dq]
                dist = _np.sqrt((ax - dx_c) ** 2 + (ay - dy_c) ** 2)
                if dist < 1.5:
                    support.append(dq)
            if support:
                pauli_str = pauli_type * len(support)
                stabilizers.append(Stabilizer(
                    support=tuple(support),
                    pauli_string=pauli_str,
                ))
        
        # Logical operators along boundaries
        if data_qubits:
            top_row = sorted(data_qubits, key=lambda q: coord_dict[q][1], reverse=True)[:d]
            left_col = sorted(data_qubits, key=lambda q: coord_dict[q][0])[:d]
            logical_x_operators.append(
                Stabilizer(support=tuple(top_row), pauli_string="X" * len(top_row))
            )
            logical_z_operators.append(
                Stabilizer(support=tuple(left_col), pauli_string="Z" * len(left_col))
            )
        
        block = Block(
            unique_label="cc1",
            stabilizers=tuple(stabilizers),
            logical_x_operators=tuple(logical_x_operators),
            logical_z_operators=tuple(logical_z_operators),
        )
        
        n_cycles = max(1, spec.rounds)
        operations = [
            [ResetAllDataQubits("cc1", state="0")],
            [MeasureBlockSyndromes("cc1", n_cycles=n_cycles)],
            [MeasureLogicalZ("cc1")],
        ]
        
        eka = Eka(lattice=lattice, blocks=[block], operations=operations)
        interpreted = interpret_eka(eka)
        
        # interpret_eka returns a Stim circuit string or object
        circuit_str = interpreted.final_circuit
        if isinstance(circuit_str, str):
            return stim.Circuit(circuit_str)
        return circuit_str
