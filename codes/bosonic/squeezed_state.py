"""Squeezed state surface code variant.

Implements surface code using squeezed state bosonic encoding.
Squeezed states reduce noise in one quadrature at the expense of
increased noise in the conjugate quadrature.
"""

from __future__ import annotations

import stim

from surface_code_in_stem.surface_code import surface_code_circuit_string
from surface_code_in_stem.noise_models import BiasedNoiseModel


def squeezed_surface_code(
    distance: int,
    rounds: int,
    p: float,
    squeezing_db: float = 10.0,
    squeezed_quadrature: str = "p",
) -> stim.Circuit:
    """Generate surface code with squeezed state qubits.

    Squeezed states have asymmetric noise:
    - One quadrature has reduced noise (anti-squeezed)
    - The conjugate quadrature has amplified noise (squeezed)

    This translates to biased Pauli noise on the encoded qubit.

    Args:
        distance: Code distance.
        rounds: Number of syndrome measurement rounds.
        p: Base physical error rate.
        squeezing_db: Squeezing level in decibels (higher = more squeezing).
        squeezed_quadrature: Which quadrature is squeezed ('p' or 'q').
            'p' squeezing -> X-biased noise
            'q' squeezing -> Z-biased noise

    Returns:
        Stim circuit with quadrature-appropriate biased noise.
    """
    # Convert dB to linear scale
    # squeezing_factor = 10^(squeezing_db / 10)
    import math
    squeezing_factor = 10 ** (squeezing_db / 10)

    # Asymmetric noise based on squeezing
    # Squeezed quadrature: less noise
    # Anti-squeezed quadrature: more noise
    if squeezed_quadrature == "p":
        # P-squeezed -> X errors suppressed, Z errors enhanced
        bias_ratio = squeezing_factor
        biased_pauli = "Z"
    else:  # "q"
        # Q-squeezed -> Z errors suppressed, X errors enhanced
        bias_ratio = squeezing_factor
        biased_pauli = "X"

    total_bias = 1.0 + bias_ratio

    if biased_pauli == "Z":
        p_x = p / total_bias
        p_z = p * bias_ratio / total_bias
    else:  # X-biased
        p_x = p * bias_ratio / total_bias
        p_z = p / total_bias

    p_y = p * 0.1  # Moderate Y error rate

    noise_model = BiasedNoiseModel(
        p_x=p_x,
        p_y=p_y,
        p_z=p_z,
        biased_pauli=biased_pauli,
        bias_ratio=bias_ratio,
    )

    circuit_str = surface_code_circuit_string(distance, rounds, p, noise_model=noise_model)
    circuit = stim.Circuit(circuit_str)

    return circuit


def squeezed_circuit_string(
    distance: int,
    rounds: int,
    p: float,
    squeezing_db: float = 10.0,
    squeezed_quadrature: str = "p",
) -> str:
    """Generate squeezed state circuit string with annotations.

    Args:
        distance: Code distance.
        rounds: Number of syndrome measurement rounds.
        p: Physical error rate.
        squeezing_db: Squeezing level in dB.
        squeezed_quadrature: Which quadrature is squeezed ('p' or 'q').

    Returns:
        Annotated Stim circuit string.
    """
    circuit = squeezed_surface_code(distance, rounds, p, squeezing_db, squeezed_quadrature)

    circuit_str = str(circuit)

    header = f"# Squeezed State Surface Code\n"
    header += f"# Parameters: distance={distance}, rounds={rounds}, p={p}\n"
    header += f"# Squeezing: {squeezing_db} dB in {squeezed_quadrature} quadrature\n"
    header += f"# Bias: {squeezed_quadrature}-squeezed -> {('Z' if squeezed_quadrature == 'p' else 'X')}-biased noise\n"

    return header + circuit_str
