"""Surface code builders and helpers."""

from __future__ import annotations

from .surface_code import surface_code_circuit_string
from .dynamic_surface_codes import (
    DynamicLayout,
    StimStringBuilder,
    hexagonal_surface_code,
    iswap_surface_code,
    walking_surface_code,
)


def compare_nested_policies(*args, **kwargs):
    """Lazy wrapper for RL comparison utility to preserve import compatibility."""

    from .rl_nested_learning import compare_nested_policies as _impl

    return _impl(*args, **kwargs)


def tabulate_comparison(*args, **kwargs):
    """Lazy wrapper for RL comparison tabulation utility."""

    from .rl_nested_learning import tabulate_comparison as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "surface_code_circuit_string",
    "DynamicLayout",
    "StimStringBuilder",
    "hexagonal_surface_code",
    "iswap_surface_code",
    "walking_surface_code",
    "compare_nested_policies",
    "tabulate_comparison",
]
