"""Benchmark harness for evaluating multiple code families uniformly."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .interfaces import CircuitGenerationConfig, DecoderFn
from .registry import get_plugin


def default_stim_decoder_evaluator(circuit_string: str, shots: int, seed: int | None) -> Mapping[str, Any]:
    """Evaluate a Stim circuit by estimating logical observable-0 error rate."""

    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError("NumPy is required to run benchmark evaluations.") from exc

    try:
        import stim
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Stim is required to run benchmark evaluations.") from exc

    circuit = stim.Circuit(circuit_string)
    if circuit.num_observables == 0:
        raise ValueError("Circuit must define observable 0 for benchmark evaluation.")

    sampler = circuit.compile_detector_sampler(seed=seed)
    _, observable_samples = sampler.sample(shots, separate_observables=True)
    logical_error_rate = float(np.mean(observable_samples[:, 0]))

    return {
        "logical_error_rate": logical_error_rate,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
    }


def benchmark_code_families(
    families: Iterable[str],
    config: CircuitGenerationConfig,
    *,
    shots: int,
    decoder_evaluator: DecoderFn = default_stim_decoder_evaluator,
    seed: int | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Run the same decoder/evaluation API across multiple code families."""

    if shots <= 0:
        raise ValueError("shots must be a positive integer.")

    results: dict[str, Mapping[str, Any]] = {}
    for family in families:
        plugin = get_plugin(family)
        metadata = plugin.decoder_metadata()
        circuit_string = plugin.build_circuit(config)
        metrics = dict(decoder_evaluator(circuit_string, shots, seed))
        metrics["decoder_metadata"] = {
            "compatible_decoders": metadata.compatible_decoders,
            "required_inputs": metadata.required_inputs,
            "syndrome_format": metadata.syndrome_spec.format_name,
        }
        results[family] = metrics

    return results
