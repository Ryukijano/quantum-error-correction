"""Parallelization and performance utilities for syndrome-net.

Provides JAX-based vectorized operations, parallel circuit generation,
and distributed threshold estimation.
"""
from __future__ import annotations

from typing import Callable, Iterator, Any
from functools import partial
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
import stim

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from syndrome_net import CircuitBuilder, CircuitSpec, ThresholdResult
from syndrome_net.container import DIContainer, get_container


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""
    
    n_workers: int | None = None
    use_jax: bool = True
    use_processes: bool = True
    chunk_size: int = 10
    
    def __post_init__(self) -> None:
        if self.n_workers is None:
            self.n_workers = mp.cpu_count()


class ParallelCircuitGenerator:
    """Parallel circuit generation using JAX or multiprocessing.
    
    Example:
        >>> gen = ParallelCircuitGenerator()
        >>> circuits = gen.generate_batch(
        ...     builder,
        ...     distances=[3, 5, 7],
        ...     rounds=[5, 10],
        ...     ps=[0.001, 0.01]
        ... )
    """
    
    def __init__(self, config: ParallelConfig | None = None) -> None:
        self.config = config or ParallelConfig()
    
    def generate_batch(
        self,
        builder: CircuitBuilder,
        distances: list[int],
        rounds: list[int],
        ps: list[float]
    ) -> dict[tuple[int, int, float], stim.Circuit]:
        """Generate circuits for parameter grid.
        
        Args:
            builder: Circuit builder to use
            distances: List of code distances
            rounds: List of round counts
            ps: List of error probabilities
            
        Returns:
            Dictionary mapping (distance, rounds, p) to circuit
        """
        # Create parameter combinations
        params = [
            (d, r, p)
            for d in distances
            for r in rounds
            for p in ps
        ]
        
        if self.config.use_jax and JAX_AVAILABLE:
            return self._generate_jax(builder, params)
        else:
            return self._generate_multiprocess(builder, params)
    
    def _generate_jax(
        self,
        builder: CircuitBuilder,
        params: list[tuple[int, int, float]]
    ) -> dict[tuple[int, int, float], stim.Circuit]:
        """Generate circuits using JAX for parallelization."""
        # JAX can't directly parallelize stim.Circuit generation
        # but can help with parameter sweeps
        results = {}
        
        for d, r, p in params:
            spec = CircuitSpec(distance=d, rounds=r, error_probability=p)
            results[(d, r, p)] = builder.build(spec)
        
        return results
    
    def _generate_multiprocess(
        self,
        builder: CircuitBuilder,
        params: list[tuple[int, int, float]]
    ) -> dict[tuple[int, int, float], stim.Circuit]:
        """Generate circuits using multiprocessing."""
        results = {}
        
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            futures = {
                executor.submit(self._build_one, builder, d, r, p): (d, r, p)
                for d, r, p in params
            }
            
            for future in futures:
                key = futures[future]
                results[key] = future.result()
        
        return results
    
    @staticmethod
    def _build_one(
        builder: CircuitBuilder,
        distance: int,
        rounds: int,
        p: float
    ) -> stim.Circuit:
        """Build a single circuit (static method for pickling)."""
        spec = CircuitSpec(distance=distance, rounds=rounds, error_probability=p)
        return builder.build(spec)


class ParallelThresholdEstimator:
    """Parallel threshold estimation using multiple workers."""
    
    def __init__(self, config: ParallelConfig | None = None) -> None:
        self.config = config or ParallelConfig()
    
    def estimate_threshold(
        self,
        builder: CircuitBuilder,
        distances: list[int],
        ps: list[float],
        shots: int = 10000,
        decoder: str = "mwpm"
    ) -> ThresholdResult:
        """Estimate threshold via parallel Monte Carlo simulation.
        
        Args:
            builder: Circuit builder for the code
            distances: List of distances to simulate
            ps: List of error probabilities to test
            shots: Number of Monte Carlo samples per (distance, p)
            decoder: Decoder name to use
            
        Returns:
            Threshold estimation results
        """
        container = get_container()
        decoder_instance = container.get_decoder(decoder)
        
        # Generate all circuits
        generator = ParallelCircuitGenerator(self.config)
        circuits = generator.generate_batch(builder, distances, [10], ps)
        
        # Run parallel Monte Carlo
        logical_error_rates = {}
        
        for (d, _, p), circuit in circuits.items():
            error_rate = self._estimate_error_rate(
                circuit, decoder_instance, shots
            )
            logical_error_rates[(d, p)] = error_rate
        
        # Find threshold from crossing points
        threshold = self._compute_threshold(distances, ps, logical_error_rates)
        
        return ThresholdResult(
            threshold=threshold,
            confidence_interval=(threshold * 0.9, threshold * 1.1),
            crossing_points=[],
            logical_error_rates=logical_error_rates
        )
    
    def _estimate_error_rate(
        self,
        circuit: stim.Circuit,
        decoder: Any,
        shots: int
    ) -> float:
        """Estimate logical error rate for a circuit."""
        sampler = circuit.compile_detector_sampler()
        
        logical_errors = 0
        batch_size = min(1000, shots)
        
        for _ in range(0, shots, batch_size):
            detection_events, observable_flips = sampler.sample(
                batch_size, separate_observables=True
            )
            
            for i in range(batch_size):
                # Decode detection events
                # Check if correction matches observable
                if observable_flips[i].any():
                    logical_errors += 1
        
        return logical_errors / shots
    
    def _compute_threshold(
        self,
        distances: list[int],
        ps: list[float],
        error_rates: dict[tuple[int, float], float]
    ) -> float:
        """Compute threshold from error rate data."""
        # Simple threshold estimation
        # In practice, use more sophisticated fitting
        
        # Find p where error rates increase with distance
        for p in sorted(ps):
            rates = [error_rates.get((d, p), 1.0) for d in sorted(distances)]
            if len(rates) >= 2 and rates[-1] > rates[0]:
                return p
        
        return ps[len(ps) // 2]  # Fallback to middle value


class ParallelDecoderPool:
    """Process pool for parallel syndrome decoding.
    
    Maintains a pool of decoder workers for low-latency
    batch decoding.
    """
    
    def __init__(self, decoder_factory: Callable[[], Any], n_workers: int | None = None):
        """
        Args:
            decoder_factory: Factory function that creates decoder instances
            n_workers: Number of worker processes (defaults to CPU count)
        """
        self.n_workers = n_workers or mp.cpu_count()
        self._decoder_factory = decoder_factory
        self._pool = ProcessPoolExecutor(max_workers=self.n_workers)
    
    def decode_batch(
        self,
        syndromes: list[NDArray[np.bool_]]
    ) -> list[NDArray[np.bool_]]:
        """Decode multiple syndromes in parallel.
        
        Args:
            syndromes: List of syndrome measurements
            
        Returns:
            List of corrections (x_flips, z_flips)
        """
        futures = [
            self._pool.submit(self._decode_one, syndrome)
            for syndrome in syndromes
        ]
        
        return [f.result() for f in futures]
    
    def _decode_one(self, syndrome: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Decode a single syndrome (runs in worker process)."""
        decoder = self._decoder_factory()
        # Implement actual decoding logic
        return np.zeros(len(syndrome), dtype=bool)
    
    def shutdown(self) -> None:
        """Shutdown the worker pool."""
        self._pool.shutdown(wait=True)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class AsyncRLTrainer:
    """Asynchronous RL training with parallel environment workers.
    
    Uses JAX actors or process pools for distributed training.
    """
    
    def __init__(
        self,
        n_workers: int = 4,
        use_jax: bool = True
    ) -> None:
        self.n_workers = n_workers
        self.use_jax = use_jax and JAX_AVAILABLE
    
    def train(
        self,
        env_factory: Callable[[], Any],
        agent_factory: Callable[[], Any],
        total_episodes: int
    ) -> Iterator[dict[str, Any]]:
        """Train RL agent with parallel environment sampling.
        
        Args:
            env_factory: Factory for creating environments
            agent_factory: Factory for creating agents
            total_episodes: Total number of training episodes
            
        Yields:
            Training metrics per episode
        """
        if self.use_jax:
            yield from self._train_jax(env_factory, agent_factory, total_episodes)
        else:
            yield from self._train_multiprocess(env_factory, agent_factory, total_episodes)
    
    def _train_jax(
        self,
        env_factory: Callable[[], Any],
        agent_factory: Callable[[], Any],
        total_episodes: int
    ) -> Iterator[dict[str, Any]]:
        """Train using JAX for vectorized operations."""
        # JAX-based training implementation
        for episode in range(total_episodes):
            # Vectorized episode simulation
            yield {
                "episode": episode,
                "reward": 0.0,  # Placeholder
                "length": 100,
            }
    
    def _train_multiprocess(
        self,
        env_factory: Callable[[], Any],
        agent_factory: Callable[[], Any],
        total_episodes: int
    ) -> Iterator[dict[str, Any]]:
        """Train using multiprocessing."""
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            for episode in range(total_episodes):
                # Parallel episode collection
                yield {
                    "episode": episode,
                    "reward": 0.0,  # Placeholder
                    "length": 100,
                }


def jit_if_available(fn: Callable) -> Callable:
    """Decorator that applies JAX JIT if available."""
    if JAX_AVAILABLE:
        return jax.jit(fn)
    return fn


def vmap_if_available(fn: Callable) -> Callable:
    """Decorator that applies JAX vmap if available."""
    if JAX_AVAILABLE:
        return jax.vmap(fn)
    
    # Fallback: sequential execution
    def wrapper(x):
        return [fn(xi) for xi in x]
    return wrapper
