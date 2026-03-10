"""Helpers for comparing RL policy configurations and decoding strategies.

This module includes deterministic seed utilities and a runnable
confidence-aware decoding demo that compares hard-decision MWPM against a
confidence-weighted variant on repetition-code and distance-3 rotated surface
code circuits.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np


def _deterministic_seed(component: str, *, base_seed: int = 0) -> int:
    """Return a stable integer seed derived from a component name."""

    digest = hashlib.md5(component.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:16], 16)


def compare_nested_policies(
    policies: Mapping[str, Iterable[str]], *, base_seed: int = 0
) -> Dict[Tuple[str, str], int]:
    """Return deterministic seeds for nested builder/policy pairs."""

    seeds: Dict[Tuple[str, str], int] = {}
    for builder_name, policy_names in policies.items():
        for policy_name in policy_names:
            component = f"{builder_name}:{policy_name}"
            seeds[(builder_name, policy_name)] = _deterministic_seed(component, base_seed=base_seed)
    return seeds


def _simulate_circuit_for_confidence_demo(
    circuit: "stim.Circuit",
    *,
    shots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample detector events and observables for a confidence demo."""

    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, observables = sampler.sample(shots=shots, separate_observables=True)
    return np.asarray(dets, dtype=np.uint8), np.asarray(observables[:, 0], dtype=np.uint8)


def _synthetic_confidence_from_syndromes(
    hard_bits: np.ndarray,
    *,
    seed: int,
    low_for_one: float = 0.15,
    high_for_zero: float = 0.9,
    jitter: float = 0.1,
) -> np.ndarray:
    """Generate synthetic confidence values correlated with detector outcomes."""

    rng = np.random.default_rng(seed)
    baseline = np.where(hard_bits == 1, low_for_one, high_for_zero)
    noise = rng.uniform(-jitter, jitter, size=hard_bits.shape)
    return np.clip(baseline + noise, 0.0, 1.0)


def run_confidence_aware_decoding_demo(*, shots: int = 2_000, p: float = 0.02, seed: int = 1234) -> list[dict[str, float | str]]:
    """Run hard-vs-soft decoding comparison on repetition and d=3 surface code."""

    try:
        import stim
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError("stim is required for the confidence-aware decoding demo") from exc

    from surface_code_in_stem.confidence_decoding import SyndromeBatch, WeightedMWPMDecoder

    circuits = {
        "repetition_d3": stim.Circuit.generated(
            "repetition_code:memory",
            distance=3,
            rounds=3,
            before_round_data_depolarization=p,
        ),
        "surface_d3": stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=3,
            rounds=3,
            after_clifford_depolarization=p,
        ),
    }

    rows: list[dict[str, float | str]] = []
    for offset, (name, circuit) in enumerate(circuits.items()):
        hard_bits, logicals = _simulate_circuit_for_confidence_demo(circuit, shots=shots, seed=seed + offset)
        confidence = _synthetic_confidence_from_syndromes(hard_bits, seed=seed + 100 + offset)

        decoder = WeightedMWPMDecoder(circuit.detector_error_model(decompose_errors=True), confidence_scale=1.5)
        hard_predictions = decoder.decode_batch(SyndromeBatch(hard_bits=hard_bits))
        soft_predictions = decoder.decode_batch(SyndromeBatch(hard_bits=hard_bits, confidence=confidence))

        rows.append(
            {
                "code": name,
                "hard_logical_error_rate": float(np.mean(hard_predictions != logicals)),
                "soft_logical_error_rate": float(np.mean(soft_predictions != logicals)),
            }
        )

    return rows


if __name__ == "__main__":
    report = run_confidence_aware_decoding_demo()
    print("Hard-vs-soft decoding comparison")
    for row in report:
        print(
            f"- {row['code']}: hard={row['hard_logical_error_rate']:.4f}, "
            f"soft={row['soft_logical_error_rate']:.4f}"
        )
