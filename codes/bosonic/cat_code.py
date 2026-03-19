"""Cat code surface code variant.

Implements a surface code using bosonic cat codes as the physical qubits.
Cat codes are continuous-variable codes that encode a qubit in superpositions
of coherent states (|α⟩ + |-α⟩)/√2 for logical |0⟩ and (|α⟩ - |-α⟩)/√2 for |1⟩.
"""

from __future__ import annotations

import stim

from surface_code_in_stem.surface_code import surface_code_circuit_string
from surface_code_in_stem.noise_models import BiasedNoiseModel


def cat_surface_code(
    distance: int,
    rounds: int,
    p: float,
    alpha: float = 2.0,
    bias_ratio: float = 10.0,
) -> stim.Circuit:
    """Generate a surface code circuit with cat code qubits.

    Cat codes naturally suppress bit-flip errors (X errors) due to the
    large overlap between |α⟩ and |-α⟩ for small α, but are more susceptible
    to phase-flip errors (Z errors).

    Args:
        distance: Code distance.
        rounds: Number of syndrome measurement rounds.
        p: Base physical error rate.
        alpha: Cat code coherent state amplitude. Larger α = better phase protection
               but more bit-flip susceptibility.
        bias_ratio: Ratio of Z to X error rates (default 10:1 biased toward Z).

    Returns:
        Stim circuit with biased noise appropriate for cat codes.
    """
    # Cat codes have intrinsic bias: phase errors dominate
    # Error model: p_Z = p * bias_ratio / (1 + bias_ratio)
    #              p_X = p / (1 + bias_ratio)
    #              p_Y = negligible for large alpha

    total_bias = 1.0 + bias_ratio
    p_x = p / total_bias
    p_z = p * bias_ratio / total_bias
    p_y = p * 0.01  # Very small Y error rate

    # Create biased noise model
    noise_model = BiasedNoiseModel(
        p_x=p_x,
        p_y=p_y,
        p_z=p_z,
        biased_pauli="Z",
        bias_ratio=bias_ratio,
    )

    # Generate circuit with biased noise
    circuit_str = surface_code_circuit_string(distance, rounds, p, noise_model=noise_model)
    circuit = stim.Circuit(circuit_str)

    return circuit


def cat_biased_circuit(
    distance: int,
    rounds: int,
    p: float,
    alpha: float = 2.0,
) -> str:
    """Generate cat code circuit string with explicit bias annotations.

    Args:
        distance: Code distance.
        rounds: Number of syndrome measurement rounds.
        p: Physical error rate.
        alpha: Cat code amplitude parameter.

    Returns:
        Stim circuit string with comments indicating cat code parameters.
    """
    # For large alpha, the code becomes very biased toward phase flips
    effective_bias = max(10.0, alpha ** 2)

    circuit = cat_surface_code(distance, rounds, p, alpha, effective_bias)

    # Convert to string and add metadata comment
    circuit_str = str(circuit)

    # Prepend metadata as Stim comment
    header = f"# Cat Code Surface Code\n"
    header += f"# Parameters: distance={distance}, rounds={rounds}, p={p}, alpha={alpha}\n"
    header += f"# Bias ratio: {effective_bias:.1f} (Z-biased)\n"

    return header + circuit_str
