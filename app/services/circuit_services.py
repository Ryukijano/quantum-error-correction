"""Service helpers used by Streamlit app for SOLID-friendly orchestration."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from syndrome_net import CircuitSpec
from syndrome_net.container import get_container
from surface_code_in_stem.decoders.resolution import resolve_threshold_decoder


@lru_cache(maxsize=4)
def get_builder_names() -> list[str]:
    """Return all registered circuit builder identifiers."""
    container = get_container()
    return sorted(container.circuit_builders.list())


@lru_cache(maxsize=4)
def get_decoder_names() -> list[str]:
    """Return all registered surface-decoder identifiers for threshold sweeps."""
    return get_threshold_decoder_factory().available()


class ThresholdDecoderFactory:
    """Factory for threshold decoders backed by the global DI container."""

    def available(self) -> list[str]:
        """Return all registered threshold-decoder names."""
        return sorted(get_container().decoders.list())

    def build(self, decoder_name: str):
        """Resolve a decoder by name with a consistent validation path."""
        return resolve_threshold_decoder(decoder_name)


@lru_cache(maxsize=4)
def get_threshold_decoder_factory() -> ThresholdDecoderFactory:
    """Return the cached threshold decoder factory."""
    return ThresholdDecoderFactory()


@lru_cache(maxsize=4)
def get_visualizer_names() -> list[str]:
    """Return all registered visualizer identifiers."""
    container = get_container()
    names = sorted(container.visualizers.list())
    if "crumble" not in names:
        names.append("crumble")
    return names


def get_visualizer(name: str):
    """Return a DI-registered visualizer by name."""
    container = get_container()
    return container.get_visualizer(name)


def build_circuit(distance: int, rounds: int, p: float, builder_name: str | None = None):
    """Build a Stim circuit using the DI-registered builder."""
    container = get_container()
    builder = container.get_builder(builder_name)
    spec = CircuitSpec(distance=distance, rounds=rounds, error_probability=float(p), code_family=builder_name)
    return builder.build(spec)


def service_build_circuit(
    distance: int,
    rounds: int,
    p: float,
    builder_name: str | None = None,
):
    """Compatibility shim for legacy callers expecting `service_build_circuit`."""
    return build_circuit(
        distance=distance,
        rounds=rounds,
        p=p,
        builder_name=builder_name,
    )


def build_threshold_decoder(decoder_name: str):
    """Build a threshold-mode decoder.

    This keeps threshold UI code agnostic to decoder implementations.
    """
    return get_threshold_decoder_factory().build(decoder_name)


def estimate_logical_error_rate(
    circuit: Any,
    *,
    shots: int,
    seed: int | None,
    decoder_name: str,
) -> float:
    """Estimate logical error rate for a provided circuit via the existing nested-learning helper."""
    from surface_code_in_stem.rl_nested_learning import _logical_error_rate

    circuit_string = circuit if isinstance(circuit, str) else str(circuit)
    decoder = build_threshold_decoder(decoder_name)
    return float(_logical_error_rate(circuit_string, shots=shots, seed=seed, decoder=decoder))

