"""Helpers for comparing RL policy configurations.

This module focuses on deterministic seed generation so that repeated
experiments remain reproducible regardless of Python's salted hash
randomization.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Mapping, Tuple


def _deterministic_seed(component: str, *, base_seed: int = 0) -> int:
    """Return a stable integer seed derived from a component name.

    Python's built-in ``hash`` is intentionally salted per interpreter
    start, which makes seeds derived from it non-reproducible across
    processes. Instead, this helper uses ``hashlib.md5`` to derive a
    deterministic 64-bit value and offsets it by ``base_seed`` so that
    repeated runs always assign the same seed to a given name.
    """

    digest = hashlib.md5(component.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:16], 16)


def compare_nested_policies(
    policies: Mapping[str, Iterable[str]], *, base_seed: int = 0
) -> Dict[Tuple[str, str], int]:
    """Return deterministic seeds for nested builder/policy pairs.

    Args:
        policies: Mapping of builder names to the iterable of policy names
            defined for that builder.
        base_seed: Optional integer offset that is applied to every seed so
            that experiments can be grouped under a shared top-level seed
            while remaining deterministic within each run.

    Returns:
        A dictionary keyed by ``(builder_name, policy_name)`` tuples with
        integer seeds that are stable across interpreter invocations.

    Notes:
        This function previously relied on Python's salted ``hash`` output,
        which changes between runs. The md5-based strategy here guarantees
        reproducibility as long as builder and policy names remain constant.
    """

    seeds: Dict[Tuple[str, str], int] = {}
    for builder_name, policy_names in policies.items():
        for policy_name in policy_names:
            component = f"{builder_name}:{policy_name}"
            seeds[(builder_name, policy_name)] = _deterministic_seed(component, base_seed=base_seed)
    return seeds
