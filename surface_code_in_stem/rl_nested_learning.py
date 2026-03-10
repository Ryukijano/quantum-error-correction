"""Helpers for quickly comparing nested surface-code policies.

The helpers in this module run tiny Stim simulations for both the original
static surface code builder and one of the dynamic builders. They return simple
tables of logical error rates that can be consumed by notebooks or tests
without requiring long simulation times.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Callable, Dict, Iterable

import numpy as np

from surface_code_in_stem.dynamic import hexagonal_surface_code
from surface_code_in_stem.decoders import DecoderMetadata, DecoderProtocol, MWPMDecoder
from surface_code_in_stem.surface_code import surface_code_circuit_string


StimBuilder = Callable[[int, int, float], str]


def _logical_error_rate(
    circuit_string: str,
    shots: int,
    seed: int | None,
    decoder: DecoderProtocol | None = None,
) -> float:
    """Estimate post-decoding logical-observable-0 error probability."""

    try:
        import stim
    except ModuleNotFoundError as exc:  # pragma: no cover - handled by tests
        raise ImportError("Stim is required to sample logical error rates.") from exc

    circuit = stim.Circuit(circuit_string)
    if circuit.num_observables == 0:
        raise ValueError("Circuit must define observable 0 to estimate logical error rate.")

    sampler = circuit.compile_detector_sampler(seed=seed)
    detector_samples, observable_samples = sampler.sample(shots, separate_observables=True)

    active_decoder = decoder or MWPMDecoder()
    metadata = DecoderMetadata(
        num_observables=circuit.num_observables,
        detector_error_model=None,
        circuit=circuit,
        seed=seed,
    )
    if isinstance(active_decoder, MWPMDecoder) and find_spec("pymatching") is not None:
        metadata.detector_error_model = circuit.detector_error_model(decompose_errors=True)
    decoded = active_decoder.decode(detector_samples, metadata=metadata)

    # Post-decoding logical error is the residual mismatch between decoder
    # predictions and the sampled observable values.
    logical_predictions = np.asarray(decoded.logical_predictions)

    if logical_predictions.shape != observable_samples.shape:
        raise ValueError(
            f"Decoder returned logical_predictions with shape {logical_predictions.shape}, "
            f"but expected {observable_samples.shape} to match observable_samples."
        )

    if logical_predictions.dtype != observable_samples.dtype:
        logical_predictions = logical_predictions.astype(observable_samples.dtype, copy=False)

    logical_mismatch = np.logical_xor(logical_predictions, observable_samples)
    return float(np.mean(logical_mismatch[:, 0]))


def compare_nested_policies(
    *,
    distance: int,
    rounds: int,
    p: float,
    shots: int,
    seed: int | None = None,
    static_builder: StimBuilder = surface_code_circuit_string,
    dynamic_builder: StimBuilder = hexagonal_surface_code,
) -> Dict[str, Dict[str, float | int | str | None]]:
    """Run small simulations for static and dynamic builders.

    Returns a dictionary keyed by policy name containing the logical error rate
    and the simulation metadata used to generate it.
    """

    if not isinstance(distance, int):
        raise ValueError("distance must be an integer.")
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be an odd integer >= 3.")

    if not isinstance(rounds, int) or rounds <= 0:
        raise ValueError("rounds must be a positive integer.")

    if not isinstance(shots, int) or shots <= 0:
        raise ValueError("shots must be a positive integer.")

    if not isinstance(p, float) or not 0.0 <= p <= 1.0:
        raise ValueError("p must be a float between 0 and 1 (inclusive).")

    if not callable(static_builder):
        raise ValueError("static_builder must be callable.")
    if not callable(dynamic_builder):
        raise ValueError("dynamic_builder must be callable.")

    policies: Dict[str, StimBuilder] = {
        "static": static_builder,
        "dynamic": dynamic_builder,
    }
    results: Dict[str, Dict[str, float | int | str | None]] = {}

    for name, builder in policies.items():
        circuit_str = builder(distance, rounds, p)
        results[name] = {
            "builder": builder.__name__,
            "distance": distance,
            "rounds": rounds,
            "p": p,
            "shots": shots,
            "seed": seed,
            "logical_error_rate": _logical_error_rate(circuit_str, shots, seed),
        }

    return results


def tabulate_comparison(comparison: Dict[str, Dict[str, float | int | str | None]]) -> Iterable[Dict[str, float | int | str | None]]:
    """Flatten a comparison dictionary into a list of rows."""

    for policy, metrics in comparison.items():
        yield {"policy": policy, **metrics}
