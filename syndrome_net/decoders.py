"""Decoder implementations for syndrome-net.

Provides concrete implementations of the Decoder protocol
for various syndrome decoding algorithms.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
import stim

from syndrome_net import Decoder, Syndrome, Correction


class MWPMDecoder(Decoder):
    """Minimum Weight Perfect Matching (MWPM) decoder via pymatching.
    
    Wraps `pymatching.Matching` for fast, graph-based syndrome decoding.
    Supports both per-syndrome decoding and batch circuit-level decoding.
    """
    
    def __init__(self) -> None:
        self._matcher: Any | None = None
        self._circuit: stim.Circuit | None = None
    
    @property
    def name(self) -> str:
        return "mwpm"
    
    def reset(self) -> None:
        """Reset decoder state (clears cached matcher)."""
        self._matcher = None
        self._circuit = None
    
    def attach_circuit(self, circuit: stim.Circuit) -> None:
        """Build the matching graph from a Stim circuit.
        
        Must be called before `decode()` or `decode_circuit()`.
        
        Args:
            circuit: Circuit whose detector error model defines the graph
        """
        try:
            import pymatching
        except ImportError:
            raise ImportError("pymatching is required for MWPMDecoder. "
                              "Install with: pip install pymatching")
        self._circuit = circuit
        self._matcher = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model(decompose_errors=True)
        )
    
    def decode(self, syndrome: Syndrome) -> Correction:
        """Decode a syndrome using MWPM.

        The syndrome's x_syndrome + z_syndrome arrays are treated as the
        detection event vector fed directly into pymatching.  Callers must
        ensure the array length matches the circuit's ``num_detectors``.
        If no circuit has been attached, returns an empty (no-op) correction.

        Args:
            syndrome: Measured stabilizer outcomes (concatenated detection events)

        Returns:
            Correction indicating logical flip predictions
        """
        if self._matcher is None:
            n = len(syndrome.x_syndrome) + len(syndrome.z_syndrome)
            return Correction(
                x_flips=np.zeros(n, dtype=bool),
                z_flips=np.zeros(n, dtype=bool),
                confidence=1.0,
            )

        # Full detection-event vector (x first, then z stabilizers)
        det_events = np.concatenate([
            syndrome.x_syndrome.astype(np.uint8),
            syndrome.z_syndrome.astype(np.uint8),
        ])

        # Trim or pad to match expected detector count
        expected = self._matcher.num_detectors
        if len(det_events) < expected:
            det_events = np.pad(det_events, (0, expected - len(det_events)))
        elif len(det_events) > expected:
            det_events = det_events[:expected]

        prediction = self._matcher.decode(det_events)

        # prediction is a bool array of observable flips (one per logical)
        # Map to x_flips / z_flips by convention: prediction[0] → X logical, etc.
        n_obs = len(prediction)
        return Correction(
            x_flips=np.array(prediction[:n_obs // 2 + n_obs % 2], dtype=bool),
            z_flips=np.array(prediction[n_obs // 2 + n_obs % 2:], dtype=bool),
            confidence=1.0,
        )
    
    def decode_circuit(
        self,
        circuit: stim.Circuit,
        shots: int = 1000,
        seed: int = 42,
    ) -> dict[str, float]:
        """Sample and decode a full circuit, reporting logical error rate.
        
        Args:
            circuit: Stim circuit to sample from
            shots: Number of Monte Carlo samples
            seed: Random seed for reproducibility
            
        Returns:
            Dict with 'logical_error_rate' and 'shots' keys
        """
        try:
            import pymatching
        except ImportError:
            raise ImportError("pymatching required. pip install pymatching")
        
        matcher = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model(decompose_errors=True)
        )
        
        sampler = circuit.compile_detector_sampler(seed=seed)
        detection_events, observable_flips = sampler.sample(
            shots, separate_observables=True
        )
        
        predictions = matcher.decode_batch(detection_events)
        logical_errors = int(np.sum(
            np.any(predictions != observable_flips, axis=1)
        ))
        
        return {
            "logical_error_rate": logical_errors / shots,
            "shots": shots,
            "logical_errors": logical_errors,
        }


class UnionFindDecoder(Decoder):
    """Union-Find decoder for surface codes.
    
    A fast decoder that uses union-find data structures to identify
    error clusters. Simpler than MWPM but often nearly as effective.
    """
    
    def __init__(self) -> None:
        self._clusters: dict[int, set[int]] = {}
    
    @property
    def name(self) -> str:
        return "union_find"
    
    def reset(self) -> None:
        """Reset decoder state."""
        self._clusters.clear()
    
    def decode(self, syndrome: Syndrome) -> Correction:
        """Decode syndrome using union-find algorithm.
        
        Args:
            syndrome: Measured stabilizer outcomes
            
        Returns:
            Correction indicating which data qubits to flip
        """
        # Placeholder implementation
        n_data = len(syndrome.x_syndrome) + len(syndrome.z_syndrome)
        
        return Correction(
            x_flips=np.zeros(n_data, dtype=bool),
            z_flips=np.zeros(n_data, dtype=bool),
            confidence=1.0
        )


class BeliefPropagationDecoder(Decoder):
    """Belief propagation (BP) decoder.
    
    Uses message passing on the Tanner graph to estimate error probabilities.
    Good for LDPC codes and can be combined with ordered statistics decoding.
    """
    
    def __init__(self, max_iterations: int = 100) -> None:
        self.max_iterations = max_iterations
        self._log_likelihoods: NDArray[np.float64] | None = None
    
    @property
    def name(self) -> str:
        return "belief_propagation"
    
    def reset(self) -> None:
        """Reset decoder state."""
        self._log_likelihoods = None
    
    def decode(self, syndrome: Syndrome) -> Correction:
        """Decode syndrome using belief propagation.
        
        Args:
            syndrome: Measured stabilizer outcomes
            
        Returns:
            Correction indicating which data qubits to flip
        """
        # Placeholder implementation
        n_data = len(syndrome.x_syndrome) + len(syndrome.z_syndrome)
        
        return Correction(
            x_flips=np.zeros(n_data, dtype=bool),
            z_flips=np.zeros(n_data, dtype=bool),
            confidence=0.5  # BP provides confidence estimates
        )


class NeuralDecoder(Decoder):
    """Neural network-based decoder.
    
    Uses trained neural networks (CNNs, transformers, etc.) to decode
    syndromes. Can be trained on specific noise models for optimal performance.
    """
    
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._model: Any | None = None
    
    @property
    def name(self) -> str:
        return "neural"
    
    def reset(self) -> None:
        """Reset decoder state."""
        pass
    
    def decode(self, syndrome: Syndrome) -> Correction:
        """Decode syndrome using neural network.
        
        Args:
            syndrome: Measured stabilizer outcomes
            
        Returns:
            Correction indicating which data qubits to flip
            
        Raises:
            RuntimeError: If no model is loaded
        """
        if self._model is None:
            raise RuntimeError("Neural decoder requires a trained model")
        
        # Placeholder implementation
        n_data = len(syndrome.x_syndrome) + len(syndrome.z_syndrome)
        
        return Correction(
            x_flips=np.zeros(n_data, dtype=bool),
            z_flips=np.zeros(n_data, dtype=bool),
            confidence=1.0
        )
    
    def load_model(self, path: str) -> None:
        """Load a trained neural network model.
        
        Args:
            path: Path to model file
        """
        # Implementation would load PyTorch/JAX model
        self._model = None  # Placeholder


class CompositeDecoder(Decoder):
    """Composite decoder that combines multiple decoding strategies.
    
    Can use a fast decoder for real-time decoding and a more accurate
    decoder for final results, or combine multiple decoders via voting.
    """
    
    def __init__(self, decoders: list[Decoder], mode: str = "vote") -> None:
        """
        Args:
            decoders: List of decoders to combine
            mode: Combination mode ("vote", "cascade", "confidence")
        """
        self.decoders = decoders
        self.mode = mode
    
    @property
    def name(self) -> str:
        names = [d.name for d in self.decoders]
        return f"composite_{self.mode}({'+'.join(names)})"
    
    def reset(self) -> None:
        """Reset all component decoders."""
        for decoder in self.decoders:
            decoder.reset()
    
    def decode(self, syndrome: Syndrome) -> Correction:
        """Decode using composite strategy.
        
        Args:
            syndrome: Measured stabilizer outcomes
            
        Returns:
            Combined correction from all decoders
        """
        corrections = [d.decode(syndrome) for d in self.decoders]
        
        if self.mode == "vote":
            # Majority vote on each qubit
            x_votes = np.stack([c.x_flips for c in corrections])
            z_votes = np.stack([c.z_flips for c in corrections])
            
            return Correction(
                x_flips=np.mean(x_votes, axis=0) > 0.5,
                z_flips=np.mean(z_votes, axis=0) > 0.5,
                confidence=np.mean([c.confidence for c in corrections])
            )
        
        elif self.mode == "confidence":
            # Weight by confidence
            best = max(corrections, key=lambda c: c.confidence)
            return best
        
        else:
            # Return first decoder's result
            return corrections[0]
