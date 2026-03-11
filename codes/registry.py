"""In-memory registry for code-family plugins."""

from __future__ import annotations

from typing import Dict

from .interfaces import CodeFamilyPlugin


_PLUGIN_REGISTRY: Dict[str, CodeFamilyPlugin] = {}


def register_plugin(plugin: CodeFamilyPlugin) -> None:
    """Register a plugin under its family name."""

    _PLUGIN_REGISTRY[plugin.family] = plugin


def get_plugin(family: str) -> CodeFamilyPlugin:
    """Return a plugin by family name."""

    try:
        return _PLUGIN_REGISTRY[family]
    except KeyError as exc:
        available = ", ".join(sorted(_PLUGIN_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown code family '{family}'. Available families: {available}.") from exc


def list_plugins() -> tuple[str, ...]:
    """Return all registered plugin family names."""

    return tuple(sorted(_PLUGIN_REGISTRY))
