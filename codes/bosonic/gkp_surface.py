"""GKP Surface Code Builder.

This module generates a surface code circuit that models GKP qubits.
Since Stim is a Pauli stabilizer simulator, we approximate GKP noise
as Pauli noise. GKP states are primarily afflicted by displacement errors,
which translate to Pauli errors.
"""

import stim
from surface_code_in_stem.surface_code import surface_code_circuit

def gkp_surface_code(
    distance: int,
    rounds: int,
    p: float,
    sigma: float = 0.1
) -> stim.Circuit:
    """
    Generate a surface code circuit representing GKP qubits.
    
    Args:
        distance: Code distance.
        rounds: Number of rounds.
        p: Physical error rate (depolarizing).
        sigma: GKP squeezing parameter (used to bias noise if we were doing detailed modeling).
               For this placeholder implementation, we largely use the standard surface code
               but could add specific GKP-like noise channels.
    """
    # For now, we wrap the standard surface code.
    # In a full implementation, we would insert specific analog information 
    # or biased noise models derived from 'sigma'.
    
    # We can add a comment or metadata to the circuit
    circuit = surface_code_circuit(distance, rounds, p)
    
    # Prepend some comments about GKP parameters
    header = stim.Circuit()
    header.append("QUBIT_COORDS", [], [0, 0]) # Dummy coord to attach comment? No, just comments.
    # Stim doesn't support arbitrary comments in the object structure easily preserved 
    # except via tags, but we can just return the circuit.
    
    return circuit
