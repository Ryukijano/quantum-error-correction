"""Analytical benchmark models for quantum error-correction experiments."""

from __future__ import annotations

import math
from typing import Iterable


def repetition_logical_error_rate(distance: int, physical_error_rate: float) -> float:
    """Majority-vote logical error probability for odd-distance repetition code."""
    if distance <= 0 or distance % 2 == 0:
        raise ValueError("distance must be a positive odd integer")
    if not 0.0 <= physical_error_rate <= 1.0:
        raise ValueError("physical_error_rate must be in [0, 1]")

    start = distance // 2 + 1
    return sum(
        math.comb(distance, flips)
        * (physical_error_rate**flips)
        * ((1 - physical_error_rate) ** (distance - flips))
        for flips in range(start, distance + 1)
    )


def surface_code_logical_error_rate(
    distance: int,
    physical_error_rate: float,
    threshold: float,
    prefactor: float = 0.1,
) -> float:
    """Power-law surrogate for surface-code logical error rate below threshold."""
    if distance <= 0:
        raise ValueError("distance must be positive")
    exponent = (distance + 1) / 2
    ratio = max(physical_error_rate, 1e-15) / max(threshold, 1e-15)
    return min(1.0, prefactor * (ratio**exponent))


def qldpc_logical_error_rate(
    distance: int,
    physical_error_rate: float,
    threshold: float,
    alpha: float,
    prefactor: float = 0.08,
) -> float:
    """Simple qLDPC-inspired scaling model with linear-in-distance exponent."""
    if distance <= 0:
        raise ValueError("distance must be positive")
    ratio = max(physical_error_rate, 1e-15) / max(threshold, 1e-15)
    exponent = alpha * distance
    return min(1.0, prefactor * (ratio**exponent))


def suppression_factors(logical_rates: Iterable[float]) -> list[float | None]:
    """Compute suppression factors between consecutive logical error rates."""
    rates = list(logical_rates)
    if not rates:
        return []

    factors: list[float | None] = [None]
    for prev, curr in zip(rates[:-1], rates[1:]):
        if curr <= 0:
            factors.append(None)
        else:
            factors.append(prev / curr)
    return factors


def estimate_surface_overhead(distance: int) -> tuple[int, int, float]:
    """Return qubit overhead, cycle count, and decoder latency estimate (microseconds)."""
    qubits = 2 * distance * distance
    cycles = distance
    decode_latency_us = 8.0 * distance
    return qubits, cycles, decode_latency_us


def estimate_qldpc_overhead(distance: int, rate: float = 0.1) -> tuple[int, int, float]:
    """Crude qLDPC overhead assuming constant check weight and batched decoding."""
    logical_qubits = max(1, distance // 2)
    physical_qubits = int(math.ceil(logical_qubits / rate))
    cycles = int(math.ceil(math.log2(distance + 1)))
    decode_latency_us = 25.0 * math.log2(distance + 1)
    return physical_qubits, cycles, decode_latency_us
