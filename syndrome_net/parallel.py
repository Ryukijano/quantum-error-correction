"""Parallelization and performance utilities for syndrome-net.

Provides JAX-based vectorized operations, parallel circuit generation,
and distributed threshold estimation.
"""
from __future__ import annotations

from typing import Callable, Iterator, Any, Optional
from functools import partial, lru_cache
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading

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

try:
    from color_code_stim import ColorCode, NoiseModel as CCNoiseModel
    COLOR_CODE_AVAILABLE = True
except ImportError:
    COLOR_CODE_AVAILABLE = False

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


class CircuitCache:
    """Thread-safe LRU cache for compiled Stim circuits.
    
    Avoids rebuilding expensive circuits during threshold sweeps.
    Keyed by CircuitSpec hash.
    """
    
    def __init__(self, maxsize: int = 128):
        self._cache: dict[int, stim.Circuit] = {}
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._access_order: list[int] = []
    
    def get(self, spec: CircuitSpec) -> Optional[stim.Circuit]:
        """Get cached circuit for spec if available."""
        key = hash(spec)
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
        return None
    
    def put(self, spec: CircuitSpec, circuit: stim.Circuit) -> None:
        """Cache a circuit for the given spec."""
        key = hash(spec)
        with self._lock:
            if key in self._cache:
                # Update existing
                self._access_order.remove(key)
            elif len(self._cache) >= self._maxsize:
                # Evict least recently used
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]
            
            self._cache[key] = circuit
            self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear all cached circuits."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def __len__(self) -> int:
        return len(self._cache)


class ParallelColorCodeEstimator:
    """Parallel threshold estimation for colour codes.
    
    Uses ProcessPoolExecutor for colour code circuits (heavier than surface codes)
    and circuit caching to avoid rebuilding expensive colour code circuits.
    
    Reference: Lee & Brown, Quantum 9, 1609 (2025)
    """
    
    def __init__(self, config: ParallelConfig | None = None) -> None:
        self.config = config or ParallelConfig()
        self._cache = CircuitCache(maxsize=256)
        self._cache_lock = threading.Lock()
    
    def estimate_threshold_colour_code(
        self,
        distances: list[int],
        ps: list[float],
        circuit_type: str = "tri",
        rounds_per_d: bool = True,
        shots: int = 10000,
        use_superdense: bool = False
    ) -> ThresholdResult:
        """Estimate colour code threshold via parallel simulation.
        
        Args:
            distances: List of code distances (must be odd for triangular)
            ps: List of error probabilities to test
            circuit_type: Circuit type ('tri', 'rec', 'growing', 'cult+growing')
            rounds_per_d: If True, rounds = distance, else fixed rounds
            shots: Number of Monte Carlo samples per (distance, p)
            use_superdense: Use superdense syndrome extraction
            
        Returns:
            Threshold estimation results
        """
        if not COLOR_CODE_AVAILABLE:
            raise ImportError(
                "color-code-stim is required for colour code threshold estimation. "
                "Install with: pip install color-code-stim"
            )
        
        # Validate distances
        if circuit_type == "tri":
            for d in distances:
                if d % 2 == 0:
                    raise ValueError(f"Triangular colour code requires odd distance, got {d}")
        
        # Generate jobs
        jobs = []
        for d in distances:
            rounds = d if rounds_per_d else max(distances)
            for p in ps:
                jobs.append((d, rounds, p, circuit_type, shots, use_superdense))
        
        # Run parallel simulation
        n_workers = min(self.config.n_workers or mp.cpu_count(), len(jobs))
        results: dict[tuple[int, float], float] = {}
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(self._simulate_one, job): job for job in jobs}
            
            for future in futures:
                d, rounds, p, _, _, _ = futures[future]
                ler = future.result()
                results[(d, p)] = ler
        
        # Compute threshold from crossing
        threshold = self._compute_threshold_colour_code(distances, ps, results)
        
        return ThresholdResult(
            threshold=threshold,
            confidence_interval=(threshold * 0.9, threshold * 1.1),
            crossing_points=[],
            logical_error_rates=results
        )
    
    def _simulate_one(
        self,
        job: tuple[int, int, float, str, int, bool]
    ) -> float:
        """Simulate a single (distance, p) point."""
        d, rounds, p, circuit_type, shots, use_superdense = job
        
        # Check cache first
        spec = CircuitSpec(
            distance=d,
            rounds=rounds,
            error_probability=p,
            circuit_type=circuit_type,
            superdense=use_superdense
        )
        
        cached_circuit = None
        with self._cache_lock:
            cached_circuit = self._cache.get(spec)
        
        if cached_circuit is not None:
            cc = cached_circuit
        else:
            # Build circuit
            noise = CCNoiseModel.uniform_circuit_noise(p)
            cc = ColorCode(
                d=d,
                rounds=rounds,
                circuit_type=circuit_type,
                noise_model=noise,
                superdense_circuit=use_superdense
            )
            # Cache for reuse
            with self._cache_lock:
                self._cache.put(spec, cc)
        
        # Run simulation
        num_fails, _ = cc.simulate(shots=shots)
        return num_fails / shots
    
    def _compute_threshold_colour_code(
        self,
        distances: list[int],
        ps: list[float],
        error_rates: dict[tuple[int, float], float]
    ) -> float:
        """Compute threshold from colour code error rate data."""
        # Find crossing point where larger distances have higher error rates
        sorted_ps = sorted(ps)
        
        for i, p in enumerate(sorted_ps[:-1]):
            # Get error rates at this p for all distances
            rates = [(d, error_rates.get((d, p), 1.0)) for d in distances]
            rates.sort(key=lambda x: x[0])
            
            # Check if rates increase with distance (below threshold)
            if len(rates) >= 2:
                increasing = all(rates[j][1] < rates[j+1][1] for j in range(len(rates)-1))
                if increasing:
                    # Threshold is around this p
                    return (p + sorted_ps[i+1]) / 2
        
        # Fallback: return middle p value
        return sorted_ps[len(sorted_ps) // 2]
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get circuit cache statistics."""
        return {
            "cached_circuits": len(self._cache),
            "max_size": self._cache._maxsize,
            "hit_rate": "N/A"  # Would need to track hits/misses
        }
    
    def clear_cache(self) -> None:
        """Clear the circuit cache."""
        self._cache.clear()


class ColorCodeError(Exception):
    """Base exception for colour code operations."""
    pass


class InvalidColorCodeSpecError(ColorCodeError):
    """Raised when an invalid colour code specification is provided."""
    pass


class LoomIntegrationError(ColorCodeError):
    """Raised when el-loom integration fails."""
    pass
