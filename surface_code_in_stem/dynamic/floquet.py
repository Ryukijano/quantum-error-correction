"""Floquet (honeycomb) dynamic code circuit builder.

Implements the Hastings-Haah honeycomb Floquet code using a three-round
schedule of XX, YY, ZZ paired measurements on a honeycomb lattice.

Each round measures one color of edges on the honeycomb, creating a set
of dynamically generated stabilizers (check operators that span 6 qubits
around each hexagon) and a logical qubit whose effective stabilizers
rotate in a well-defined way between rounds.

Reference: Hastings and Haah (2021), "Dynamically Generated Logical Qubits"
https://arxiv.org/abs/2107.02194
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import stim

from surface_code_in_stem.noise_models import NoiseModel, resolve_noise_model

Coord = Tuple[float, float]


@dataclass
class HoneycombLayout:
    """Coordinate layout for a honeycomb Floquet code.

    Qubits sit on vertices of a honeycomb lattice. Edges are grouped into
    three colors (0=red, 1=green, 2=blue) corresponding to XX, YY, ZZ
    measurement rounds respectively.
    """
    qubits: List[Coord]
    coord_to_index: Dict[Coord, int]
    edge_pairs: List[List[Tuple[Coord, Coord]]]  # [round_type][pair_index]
    ancilla_coords: List[Coord]  # Measurement ancillas per pair
    logical_x_qubits: List[Coord]
    logical_z_qubits: List[Coord]

    @classmethod
    def build(cls, distance: int) -> "HoneycombLayout":
        """Build a honeycomb layout for given distance.

        Creates a d×d patch of the honeycomb lattice with boundary conditions
        supporting one logical qubit.

        Args:
            distance: Code distance (must be even for standard honeycomb)
        """
        # Honeycomb qubit positions on a triangular sublattice
        # We use a width-distance, height-distance rhombus
        # Each unit cell has 2 qubits: (2i, 2j) and (2i+1, 2j)
        # Edges: type-0 (vertical), type-1 (diagonal /), type-2 (diagonal \)

        qubits = []
        coord_to_index: Dict[Coord, int] = {}

        # Create a rectangular patch of d columns × d rows (2 qubits per unit)
        rows = distance
        cols = distance

        for row in range(rows):
            for col in range(cols):
                # Two qubits per unit cell
                q0 = (float(2 * col), float(2 * row))
                q1 = (float(2 * col + 1), float(2 * row))
                for q in [q0, q1]:
                    if q not in coord_to_index:
                        coord_to_index[q] = len(qubits)
                        qubits.append(q)

        # Three types of edges (3-coloring of honeycomb)
        # Type 0 (XX): vertical edges between (2c, 2r) and (2c, 2r+1) offset pairs
        # Type 1 (YY): diagonal edges
        # Type 2 (ZZ): horizontal edges within unit cell

        edge_pairs: List[List[Tuple[Coord, Coord]]] = [[], [], []]

        for row in range(rows):
            for col in range(cols):
                q0 = (float(2 * col), float(2 * row))
                q1 = (float(2 * col + 1), float(2 * row))

                # Type 2 (ZZ): within-cell horizontal edge
                if q0 in coord_to_index and q1 in coord_to_index:
                    edge_pairs[2].append((q0, q1))

                # Type 0 (XX): vertical edge to next row's left qubit
                q_down = (float(2 * col + 1), float(2 * (row + 1)))
                if q1 in coord_to_index and q_down in coord_to_index:
                    edge_pairs[0].append((q1, q_down))

                # Type 1 (YY): cross-cell diagonal
                q_right_down = (float(2 * (col + 1)), float(2 * row))
                if q1 in coord_to_index and q_right_down in coord_to_index:
                    edge_pairs[1].append((q1, q_right_down))

        # Ancilla qubits for each edge measurement
        ancilla_coords = []

        # Logical operators: X-string along left boundary, Z-string along top
        logical_z_qubits = [(float(0), float(2 * r)) for r in range(rows) if (0.0, float(2 * r)) in coord_to_index]
        logical_x_qubits = [(float(2 * c), float(0)) for c in range(cols) if (float(2 * c), 0.0) in coord_to_index]

        return cls(
            qubits=qubits,
            coord_to_index=coord_to_index,
            edge_pairs=edge_pairs,
            ancilla_coords=ancilla_coords,
            logical_x_qubits=logical_x_qubits,
            logical_z_qubits=logical_z_qubits,
        )


def floquet_honeycomb_code(
    distance: int,
    rounds: int,
    p: float,
    noise_model: Optional[NoiseModel] = None,
) -> stim.Circuit:
    """Build a Floquet honeycomb code circuit.

    Uses a three-round schedule of paired measurements:
    - Round 0 mod 3: Measure XX on type-0 edges
    - Round 1 mod 3: Measure YY on type-1 edges
    - Round 2 mod 3: Measure ZZ on type-2 edges

    Detectors are formed by comparing each edge measurement to its
    previous appearance (every 3 rounds).

    Args:
        distance: Code distance
        rounds: Number of measurement rounds (should be multiple of 3 for clean boundaries)
        p: Physical error rate
        noise_model: Optional noise model (defaults to IID depolarizing with rate p)

    Returns:
        stim.Circuit with detectors and logical observable
    """
    noise_model = resolve_noise_model(p, noise_model)
    layout = HoneycombLayout.build(distance)

    c2i = layout.coord_to_index
    all_data = list(c2i.keys())
    n_data = len(all_data)

    # Assign ancilla indices after data qubits
    ancilla_offset = n_data
    ancilla_map: Dict[Tuple[Coord, Coord], int] = {}

    all_edge_pairs: List[Tuple[Coord, Coord]] = []
    for group in layout.edge_pairs:
        all_edge_pairs.extend(group)

    for i, pair in enumerate(all_edge_pairs):
        ancilla_map[pair] = ancilla_offset + i

    n_ancilla = len(ancilla_map)
    total_qubits = n_data + n_ancilla

    # Build the Stim circuit
    lines: List[str] = []

    # Qubit coordinates
    for coord, idx in c2i.items():
        lines.append(f"QUBIT_COORDS({coord[0]},{coord[1]}) {idx}")
    for pair, anc_idx in ancilla_map.items():
        mid_x = (pair[0][0] + pair[1][0]) / 2
        mid_y = (pair[0][1] + pair[1][1]) / 2
        lines.append(f"QUBIT_COORDS({mid_x},{mid_y}) {anc_idx}")

    # All qubit indices
    all_data_idx = list(c2i.values())
    all_anc_idx = list(ancilla_map.values())
    all_qubit_idx = all_data_idx + all_anc_idx

    # Pauli for each round type
    round_paulis = ["XX", "YY", "ZZ"]

    # Track previous ancilla measurements for detectors
    prev_anc_meas: Dict[Tuple[Coord, Coord], int] = {}
    meas_count = 0

    def add_lines(new_lines: List[str]) -> None:
        lines.extend(new_lines)

    # Initial reset
    lines.append(f"R {' '.join(map(str, all_qubit_idx))}")
    for ln in noise_model.reset_noise(qubits=list(all_qubit_idx), layer_id="init_reset"):
        lines.append(ln)
    lines.append("TICK")

    for r in range(rounds):
        round_type = r % 3
        pauli = round_paulis[round_type]
        edge_group = layout.edge_pairs[round_type]

        if not edge_group:
            continue

        # Get ancilla qubits for this round
        anc_qubits = [ancilla_map[pair] for pair in edge_group if pair in ancilla_map]

        # Reset ancillas
        if anc_qubits:
            lines.append(f"R {' '.join(map(str, anc_qubits))}")
            for ln in noise_model.reset_noise(qubits=anc_qubits, layer_id=f"round_{r}_reset"):
                lines.append(ln)
            lines.append("TICK")

        # Apply measurement circuit for this Pauli type
        if pauli == "XX":
            # XX measurement: H-CX-CX-H + measure ancilla
            for pair in edge_group:
                if pair not in ancilla_map:
                    continue
                q0, q1 = c2i[pair[0]], c2i[pair[1]]
                anc = ancilla_map[pair]
                active = {q0, q1, anc}
                idle = [q for q in all_qubit_idx if q not in active]
                lines.append(f"H {anc}")
                for ln in noise_model.gate_noise(gate="H", pair_targets=[], idle_targets=list(idle), layer_id=f"round_{r}_h_pre"):
                    lines.append(ln)
                lines.append("TICK")
                lines.append(f"CX {anc} {q0} {anc} {q1}")
                for ln in noise_model.gate_noise(gate="CX", pair_targets=[anc, q0, anc, q1], idle_targets=list(idle), layer_id=f"round_{r}_cx"):
                    lines.append(ln)
                lines.append("TICK")
                lines.append(f"H {anc}")
                for ln in noise_model.gate_noise(gate="H", pair_targets=[], idle_targets=list(idle), layer_id=f"round_{r}_h_post"):
                    lines.append(ln)
                lines.append("TICK")

        elif pauli == "ZZ":
            # ZZ measurement: CX-CX + measure ancilla
            for pair in edge_group:
                if pair not in ancilla_map:
                    continue
                q0, q1 = c2i[pair[0]], c2i[pair[1]]
                anc = ancilla_map[pair]
                active = {q0, q1, anc}
                idle = [q for q in all_qubit_idx if q not in active]
                lines.append(f"CX {q0} {anc} {q1} {anc}")
                for ln in noise_model.gate_noise(gate="CX", pair_targets=[q0, anc, q1, anc], idle_targets=list(idle), layer_id=f"round_{r}_cx"):
                    lines.append(ln)
                lines.append("TICK")

        elif pauli == "YY":
            # YY measurement: S†-H-CX-CX-H-S + measure ancilla (or via CY)
            for pair in edge_group:
                if pair not in ancilla_map:
                    continue
                q0, q1 = c2i[pair[0]], c2i[pair[1]]
                anc = ancilla_map[pair]
                active = {q0, q1, anc}
                idle = [q for q in all_qubit_idx if q not in active]
                # Rotate Y → X via S†
                lines.append(f"S_DAG {q0} {q1}")
                lines.append("TICK")
                lines.append(f"H {anc}")
                lines.append("TICK")
                lines.append(f"CX {anc} {q0} {anc} {q1}")
                for ln in noise_model.gate_noise(gate="CX", pair_targets=[anc, q0, anc, q1], idle_targets=list(idle), layer_id=f"round_{r}_cx"):
                    lines.append(ln)
                lines.append("TICK")
                lines.append(f"H {anc}")
                lines.append("TICK")
                # Rotate back X → Y via S
                lines.append(f"S {q0} {q1}")
                lines.append("TICK")

        # Measure all ancillas for this round
        if anc_qubits:
            for ln in noise_model.measurement_noise(qubits=list(anc_qubits), layer_id=f"round_{r}_meas"):
                lines.append(ln)
            lines.append(f"M {' '.join(map(str, anc_qubits))}")

            # Build detectors (compare to previous same-type measurement)
            curr_indices = list(range(meas_count, meas_count + len(anc_qubits)))
            meas_count += len(anc_qubits)

            for pair, curr_idx in zip(edge_group, curr_indices):
                if pair in ancilla_map:
                    curr_offset = meas_count - curr_idx
                    mid_x = (pair[0][0] + pair[1][0]) / 2
                    mid_y = (pair[0][1] + pair[1][1]) / 2
                    if pair in prev_anc_meas:
                        prev_idx = prev_anc_meas[pair]
                        prev_offset = meas_count - prev_idx
                        lines.append(
                            f"DETECTOR({mid_x},{mid_y}) rec[-{curr_offset}] rec[-{prev_offset}]"
                        )
                    elif r >= 3:
                        lines.append(
                            f"DETECTOR({mid_x},{mid_y}) rec[-{curr_offset}]"
                        )
                    prev_anc_meas[pair] = curr_idx

    # Final logical observable: Z-type string along left boundary
    logical_z = layout.logical_z_qubits
    if logical_z:
        final_meas_targets = [c2i[q] for q in logical_z if q in c2i]
        if final_meas_targets:
            lines.append(f"M {' '.join(map(str, final_meas_targets))}")
            obs_offsets = " ".join(
                f"rec[-{meas_count + len(final_meas_targets) - i}]"
                for i in range(len(final_meas_targets))
            )
            meas_count += len(final_meas_targets)
            lines.append(f"OBSERVABLE_INCLUDE(0) {obs_offsets}")

    return stim.Circuit("\n".join(lines) + "\n")
