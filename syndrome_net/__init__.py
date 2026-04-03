"""Core protocols and abstractions for syndrome-net QEC framework.

This module defines the fundamental interfaces that all components must implement,
enabling a plugin-based architecture that supports multiple QEC codes, decoders,
noise models, and visualization backends.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, Any
from dataclasses import dataclass
from abc import abstractmethod

import stim
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CircuitSpec:
    """Specification for building a quantum error correction circuit.
    
    Attributes:
        distance: Code distance (number of data qubits along logical operator)
        rounds: Number of syndrome measurement rounds
        error_probability: Physical error rate
    """
    distance: int
    rounds: int
    error_probability: float


@dataclass(frozen=True)
class Correction:
    """Result of a decoding operation.
    
    Attributes:
        x_flips: Boolean array indicating X corrections on data qubits
        z_flips: Boolean array indicating Z corrections on data qubits
        confidence: Confidence score for the correction (0-1)
    """
    x_flips: NDArray[np.bool_]
    z_flips: NDArray[np.bool_]
    confidence: float = 1.0


@dataclass(frozen=True)
class Syndrome:
    """Syndrome measurement result.
    
    Attributes:
        x_syndrome: Measurements from X-stabilizers
        z_syndrome: Measurements from Z-stabilizers
        time: Round index (for temporal correlations)
    """
    x_syndrome: NDArray[np.bool_]
    z_syndrome: NDArray[np.bool_]
    time: int = 0


@runtime_checkable
class CircuitBuilder(Protocol):
    """Protocol for quantum error correction circuit builders.
    
    Implementations generate stim.Circuit objects for specific QEC codes
    (surface code, hexagonal code, Floquet code, etc.)
    """
    
    @abstractmethod
    def build(self, spec: CircuitSpec) -> stim.Circuit:
        """Build a Stim circuit for the specified parameters.
        
        Args:
            spec: Circuit specification (distance, rounds, error probability)
            
        Returns:
            A Stim circuit with detectors and observables defined
            
        Raises:
            ValueError: If the specification is invalid for this code
        """
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this circuit builder."""
        ...
    
    @property
    @abstractmethod
    def supported_distances(self) -> list[int]:
        """List of valid distance parameters for this code."""
        ...
    
    @property
    def is_dynamic(self) -> bool:
        """Whether this code uses dynamic stabilizer schedules (e.g., Floquet)."""
        return False


@runtime_checkable
class Decoder(Protocol):
    """Protocol for syndrome decoders.
    
    Implementations convert syndrome measurements into error corrections.
    """
    
    @abstractmethod
    def decode(self, syndrome: Syndrome) -> Correction:
        """Decode syndrome measurements into error corrections.
        
        Args:
            syndrome: Measured stabilizer outcomes
            
        Returns:
            Correction indicating which data qubits to flip
        """
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this decoder."""
        ...
    
    @abstractmethod
    def reset(self) -> None:
        """Reset decoder state for a new decoding session."""
        ...


@runtime_checkable
class NoiseModel(Protocol):
    """Protocol for quantum noise models.
    
    Implementations apply realistic noise channels to circuits.
    """
    
    @abstractmethod
    def apply(self, circuit: stim.Circuit, p: float) -> stim.Circuit:
        """Apply noise to a circuit.
        
        Args:
            circuit: Ideal circuit without noise
            p: Base error probability
            
        Returns:
            Circuit with noise channels inserted
        """
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this noise model."""
        ...


@runtime_checkable
class Visualizer(Protocol):
    """Protocol for QEC circuit visualizers.
    
    Implementations render circuits, errors, and syndromes for display.
    """
    
    @abstractmethod
    def render_circuit(self, circuit: stim.Circuit) -> Any:
        """Render a circuit diagram.
        
        Args:
            circuit: Circuit to visualize
            
        Returns:
            Visualization object (e.g., Plotly figure, SVG string)
        """
        ...
    
    @abstractmethod
    def render_syndrome(self, syndrome: Syndrome, layout: Any) -> Any:
        """Render syndrome measurements on the code layout.
        
        Args:
            syndrome: Syndrome to visualize
            layout: Code layout information
            
        Returns:
            Visualization object
        """
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this visualizer."""
        ...


@runtime_checkable
class ThresholdEstimator(Protocol):
    """Protocol for threshold estimation algorithms.
    
    Implementations estimate the error threshold for a QEC code.
    """
    
    @abstractmethod
    def estimate(
        self,
        builder: CircuitBuilder,
        distances: list[int],
        ps: list[float],
        shots: int = 10000
    ) -> ThresholdResult:
        """Estimate the error threshold.
        
        Args:
            builder: Circuit builder for the code
            distances: List of distances to simulate
            ps: List of error probabilities to test
            shots: Number of Monte Carlo samples per (distance, p)
            
        Returns:
            Threshold estimation results
        """
        ...


@dataclass(frozen=True)
class ThresholdResult:
    """Result of threshold estimation.
    
    Attributes:
        threshold: Estimated threshold value
        confidence_interval: (lower, upper) bounds
        crossing_points: List of (distance, p_cross) where curves cross
        logical_error_rates: Dict mapping (distance, p) -> error_rate
    """
    threshold: float
    confidence_interval: tuple[float, float]
    crossing_points: list[tuple[int, float]]
    logical_error_rates: dict[tuple[int, float], float]


class SyndromeNetError(Exception):
    """Base exception for syndrome-net errors."""
    pass


class UnknownBuilderError(SyndromeNetError):
    """Raised when requesting an unregistered circuit builder."""
    
    def __init__(self, name: str):
        super().__init__(f"Unknown circuit builder: {name}")
        self.name = name


class InvalidSpecError(SyndromeNetError):
    """Raised when circuit specification is invalid."""
    
    def __init__(self, spec: CircuitSpec, reason: str):
        super().__init__(f"Invalid circuit spec {spec}: {reason}")
        self.spec = spec
        self.reason = reason
