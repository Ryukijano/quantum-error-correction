"""Reinforcement learning helpers that optionally rely on Stim.

This module lazily imports :mod:`stim` so that users without the
extra dependency can still import the file without immediate failures.
Call :func:`get_stim` to load the library or check :func:`stim_available`
first to branch behavior when Stim is missing.
"""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Optional

_stim: Optional[ModuleType] = None
_stim_import_error: Optional[ImportError] = None


def _lazy_load_stim() -> ModuleType:
    """Import and cache :mod:`stim`, raising a clear error if unavailable."""
    global _stim, _stim_import_error
    if _stim is not None:
        return _stim

    try:
        _stim = import_module("stim")
    except ImportError as exc:  # pragma: no cover - executed only without Stim
        _stim_import_error = exc
        raise ImportError(
            "`rl_nested_learning` requires the optional dependency `stim`. "
            "Install it with `pip install stim` to enable circuit generation "
            "and simulation features."
        ) from exc

    return _stim


def get_stim() -> ModuleType:
    """Return the Stim module, loading it on demand."""
    return _lazy_load_stim()


def stim_available() -> bool:
    """Check whether Stim can be imported without raising."""
    try:
        _lazy_load_stim()
    except ImportError:
        return False
    return True


__all__ = ["get_stim", "stim_available"]
