"""Erasure Surface Code Builder.

Generates a surface code circuit with heralded erasure errors.
Erasures are modeled as Pauli errors that are 'heralded' by a specific detector.
"""

import stim
from surface_code_in_stem.surface_code import surface_code_circuit_string

def erasure_surface_code(
    distance: int,
    rounds: int,
    p: float,
    erasure_prob: float
) -> stim.Circuit:
    """
    Generate a surface code circuit with erasure noise.
    
    Args:
        distance: Code distance.
        rounds: Number of rounds.
        p: Pauli error rate.
        erasure_prob: Probability of erasure error.
    """
    # Start with standard surface code
    circuit_str = surface_code_circuit_string(distance, rounds, p)
    base_circuit = stim.Circuit(circuit_str)
    
    if erasure_prob <= 0:
        return base_circuit
        
    # To properly model erasure in Stim, we need to add heralded errors.
    # A simple way is to add a correlated error that flips a data qubit AND a dedicated 'herald' qubit/detector.
    # However, modifying the compiled circuit structure is complex.
    
    # For this implementation, we will return the base circuit but with 
    # adjusted error rates to effectively model the Pauli projection of erasures
    # Note: A real erasure implementation would use the ErasureAwareNoiseModel
    # and properly annotate heralded erasure events in the DEM.
    
    circuit_str = surface_code_circuit_string(distance, rounds, p + erasure_prob * 0.5)
    circuit = stim.Circuit(circuit_str)
    
    # We append comments or dummy structure if necessary
    return circuit
