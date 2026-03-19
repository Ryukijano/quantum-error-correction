"""General qLDPC code builder from parity-check matrices.

This module provides functionality to build Stim circuits from arbitrary
qLDPC parity-check matrices, supporting both CSS and non-CSS codes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import stim


def qldpc_from_parity_matrices(
    hx: np.ndarray,
    hz: np.ndarray,
    rounds: int,
    p: float,
    logical_x: Optional[np.ndarray] = None,
    logical_z: Optional[np.ndarray] = None,
) -> stim.Circuit:
    """Build a qLDPC circuit from X and Z parity-check matrices.

    Args:
        hx: X-parity check matrix (num_x_checks, num_qubits)
        hz: Z-parity check matrix (num_z_checks, num_qubits)
        rounds: Number of syndrome measurement rounds
        p: Physical error rate
        logical_x: Optional X logical operators (num_logical, num_qubits)
        logical_z: Optional Z logical operators (num_logical, num_qubits)

    Returns:
        Stim circuit implementing the qLDPC code
    """
    num_x_checks, num_qubits = hx.shape
    num_z_checks, num_qubits_z = hz.shape

    if num_qubits != num_qubits_z:
        raise ValueError(f"hx and hz must have same number of qubits: {num_qubits} vs {num_qubits_z}")

    # Validate CSS condition: hx @ hz.T == 0 (mod 2)
    css_check = (hx @ hz.T) % 2
    if not np.all(css_check == 0):
        raise ValueError("Parity matrices do not satisfy CSS condition (hx @ hz.T != 0)")

    circuit = stim.Circuit()

    # Qubit coordinates
    for i in range(num_qubits):
        circuit.append("QUBIT_COORDS", [i], [i % 10, i // 10])

    # Initialize data qubits
    circuit.append("R", range(num_qubits))
    circuit.append("TICK")

    # Syndrome measurement rounds
    for _ in range(rounds):
        # X syndrome measurements
        for check_idx, check in enumerate(hx):
            # Find qubits participating in this X check
            involved_qubits = np.where(check == 1)[0].tolist()

            if len(involved_qubits) > 0:
                # Ancilla for measurement
                ancilla = num_qubits + check_idx

                # Reset ancilla
                circuit.append("R", [ancilla])

                # Hadamard on ancilla
                circuit.append("H", [ancilla])

                # CNOT from ancilla to each data qubit (X-type stabilizer)
                for qubit in involved_qubits:
                    circuit.append("CNOT", [ancilla, qubit])

                # Hadamard on ancilla
                circuit.append("H", [ancilla])

                # Measure ancilla
                circuit.append("M", [ancilla])

            # Add noise
            for qubit in involved_qubits:
                circuit.append("DEPOLARIZE1", [qubit], p)

        circuit.append("TICK")

        # Z syndrome measurements
        for check_idx, check in enumerate(hz):
            involved_qubits = np.where(check == 1)[0].tolist()

            if len(involved_qubits) > 0:
                ancilla = num_qubits + num_x_checks + check_idx

                circuit.append("R", [ancilla])

                # CNOT from each data qubit to ancilla (Z-type stabilizer)
                for qubit in involved_qubits:
                    circuit.append("CNOT", [qubit, ancilla])

                circuit.append("M", [ancilla])

            for qubit in involved_qubits:
                circuit.append("DEPOLARIZE1", [qubit], p)

        circuit.append("TICK")

    # Add detectors for syndrome bits
    # (Simplified - in practice would need to track measurement indices)
    for i in range(num_x_checks + num_z_checks):
        circuit.append("DETECTOR", [stim.target_rec(-1 - i)])

    # Add observables if provided
    if logical_z is not None:
        for log_idx, logical in enumerate(logical_z):
            involved_qubits = np.where(logical == 1)[0].tolist()
            targets = [stim.target_rec(-1 - i) for i in involved_qubits]
            circuit.append("OBSERVABLE_INCLUDE", targets, log_idx)

    return circuit


def toric_code_parity(
    size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate parity matrices for toric code.

    Args:
        size: Size of toric code lattice (size x size)

    Returns:
        Tuple of (hx, hz) parity matrices
    """
    num_qubits = 2 * size * size
    num_stabilizers = size * size

    hx = np.zeros((num_stabilizers, num_qubits), dtype=np.uint8)
    hz = np.zeros((num_stabilizers, num_qubits), dtype=np.uint8)

    # X-stabilizers on vertices (stars)
    for row in range(size):
        for col in range(size):
            check_idx = row * size + col
            
            # Edges meeting at vertex (row, col):
            # Horizontal edge right: (row, col)
            # Horizontal edge left: (row, col-1)
            # Vertical edge down: (row, col)
            # Vertical edge up: (row-1, col)
            
            h_right = 2 * (row * size + col)
            h_left = 2 * (row * size + ((col - 1) % size))
            v_down = 2 * (row * size + col) + 1
            v_up = 2 * (((row - 1) % size) * size + col) + 1

            hx[check_idx, h_right] = 1
            hx[check_idx, h_left] = 1
            hx[check_idx, v_down] = 1
            hx[check_idx, v_up] = 1

    # Z-stabilizers on faces (plaquettes)
    for row in range(size):
        for col in range(size):
            check_idx = row * size + col

            # Edges bordering face (row, col):
            # Horizontal edge top: (row, col)
            # Horizontal edge bottom: (row+1, col)
            # Vertical edge left: (row, col)
            # Vertical edge right: (row, col+1)
            
            h_top = 2 * (row * size + col)
            h_bottom = 2 * (((row + 1) % size) * size + col)
            v_left = 2 * (row * size + col) + 1
            v_right = 2 * (row * size + ((col + 1) % size)) + 1

            hz[check_idx, h_top] = 1
            hz[check_idx, h_bottom] = 1
            hz[check_idx, v_left] = 1
            hz[check_idx, v_right] = 1

    return hx, hz


def surface_code_parity(
    distance: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate parity matrices for surface code.

    Args:
        distance: Code distance

    Returns:
        Tuple of (hx, hz) parity matrices
    """
    num_data_qubits = distance * distance
    num_x_ancilla = (distance - 1) * distance // 2
    num_z_ancilla = (distance - 1) * distance // 2

    # For surface code, the standard construction uses a different layout
    # This is a simplified version

    num_qubits = num_data_qubits
    num_x_checks = num_x_ancilla
    num_z_checks = num_z_ancilla

    hx = np.zeros((num_x_checks, num_qubits), dtype=np.uint8)
    hz = np.zeros((num_z_checks, num_qubits), dtype=np.uint8)

    # X-checks (star operators) on alternating vertices
    x_check_idx = 0
    for row in range(distance - 1):
        for col in range(distance - 1):
            if (row + col) % 2 == 0:  # X-checks on even parity vertices
                # Involve data qubits around this vertex
                top_left = row * distance + col
                top_right = row * distance + (col + 1)
                bottom_left = (row + 1) * distance + col
                bottom_right = (row + 1) * distance + (col + 1)

                if x_check_idx < num_x_checks:
                    hx[x_check_idx, [top_left, top_right, bottom_left, bottom_right]] = 1
                    x_check_idx += 1

    # Z-checks (plaquette operators) on alternating faces
    z_check_idx = 0
    for row in range(distance - 1):
        for col in range(distance - 1):
            if (row + col) % 2 == 1:  # Z-checks on odd parity faces
                top_left = row * distance + col
                top_right = row * distance + (col + 1)
                bottom_left = (row + 1) * distance + col
                bottom_right = (row + 1) * distance + (col + 1)

                if z_check_idx < num_z_checks:
                    hz[z_check_idx, [top_left, top_right, bottom_left, bottom_right]] = 1
                    z_check_idx += 1

    return hx, hz


def hamming_code_parity(
    r: int,
) -> np.ndarray:
    """Generate parity matrix for classical Hamming code.

    Args:
        r: Number of parity bits (code length = 2^r - 1)

    Returns:
        Parity check matrix
    """
    n = 2 ** r - 1
    k = n - r

    # Hamming code parity matrix
    # Columns are all non-zero binary vectors of length r
    h = np.zeros((r, n), dtype=np.uint8)

    for i in range(n):
        # Binary representation of (i + 1)
        binary = format(i + 1, f'0{r}b')
        for j, bit in enumerate(binary):
            h[j, i] = int(bit)

    return h


def hypergraph_product(
    h1: np.ndarray,
    h2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute hypergraph product of two parity matrices.

    The hypergraph product construction creates a quantum CSS code from
    two classical codes.

    Args:
        h1: First parity matrix (r1 x n1)
        h2: Second parity matrix (r2 x n2)

    Returns:
        Tuple of (hx, hz) for the quantum code
    """
    r1, n1 = h1.shape
    r2, n2 = h2.shape

    # Number of qubits = n1 * n2 + r1 * r2
    num_qubits = n1 * n2 + r1 * r2

    # hx has dimensions (r1 * n2 + r2 * n1) x num_qubits
    num_x_checks = r1 * n2 + r2 * n1
    hx = np.zeros((num_x_checks, num_qubits), dtype=np.uint8)

    # hz has dimensions (n1 * r2 + n2 * r1) x num_qubits
    num_z_checks = n1 * r2 + n2 * r1
    hz = np.zeros((num_z_checks, num_qubits), dtype=np.uint8)

    # Construct hx
    # First block: h1 ⊗ I_n2 on qubits 0..n1*n2-1
    for i in range(r1):
        for j in range(n2):
            check_idx = i * n2 + j
            for k in range(n1):
                if h1[i, k] == 1:
                    qubit_idx = k * n2 + j
                    hx[check_idx, qubit_idx] = 1

    # Second block: I_n1 ⊗ h2^T on qubits n1*n2..n1*n2+n1*r2-1
    for i in range(n1):
        for j in range(r2):
            check_idx = i * r2 + j
            for k in range(n2):
                if h2[j, k] == 1:
                    qubit_idx = n1 * n2 + i * r2 + j
                    pass

    # The correct dimensions for hypergraph product:
    # Qubits are split into two blocks: V = V1 x V2  and C = C1 x C2
    # So num_qubits = n1 * n2 + r1 * r2
    # X checks are defined by: (H1 x I_n2 , I_r1 x H2^T)
    # Z checks are defined by: (I_n1 x H2 , H1^T x I_r2)

    hx = np.hstack([np.kron(h1, np.eye(n2, dtype=np.uint8)), np.kron(np.eye(r1, dtype=np.uint8), h2.T)])
    hz = np.hstack([np.kron(np.eye(n1, dtype=np.uint8), h2), np.kron(h1.T, np.eye(r2, dtype=np.uint8))])

    return hx, hz
