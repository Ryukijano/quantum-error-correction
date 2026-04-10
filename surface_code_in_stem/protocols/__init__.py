"""Protocol registry and contracts for syndrome-net execution backends."""

from __future__ import annotations

import logging
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Iterable, Iterator

from surface_code_in_stem.protocols.base import ProtocolContract, ProtocolRegistry
from surface_code_in_stem.protocols.nisq_protocol import NISQProtocol
from surface_code_in_stem.protocols.sqkd_protocol import SQKDProtocol
from surface_code_in_stem.protocols.surface_protocol import SurfaceProtocol

_LOGGER = logging.getLogger(__name__)
_PROTOCOL_ENTRYPOINT_GROUP = "syndrome_net.protocol_backends"


def _iter_entry_points(group: str) -> Iterable[EntryPoint]:
    """Return entry-point definitions for discovery while handling import failures."""
    try:
        points = entry_points()
    except Exception as exc:
        _LOGGER.warning("Unable to enumerate protocol entry points for %s: %s", group, exc)
        if __debug__:
            _LOGGER.debug("Error enumerating protocol entry points for %s", group, exc_info=True)
        return ()

    if hasattr(points, "select"):
        try:
            discovered = tuple(points.select(group=group))
        except Exception as exc:  # pragma: no cover - compatibility fallback
            _LOGGER.warning("Failed to select protocol entry points for %s: %s", group, exc)
            if __debug__:
                _LOGGER.debug("Error selecting protocol entry points for %s", group, exc_info=True)
            return ()
        try:
            return tuple(sorted(discovered, key=lambda point: point.name))
        except Exception as exc:  # pragma: no cover - compatibility fallback
            _LOGGER.warning("Unable to sort protocol entry points for %s: %s", group, exc)
            if __debug__:
                _LOGGER.debug("Error sorting protocol entry points for %s", group, exc_info=True)
            return discovered

    try:
        discovered = tuple(points.get(group, ()))
    except Exception as exc:  # pragma: no cover - compatibility fallback
        _LOGGER.warning("Unable to read protocol entry points for %s: %s", group, exc)
        if __debug__:
            _LOGGER.debug("Error reading protocol entry points for %s", group, exc_info=True)
        return ()

    try:
        return tuple(sorted(discovered, key=lambda point: point.name))
    except Exception as exc:  # pragma: no cover - compatibility fallback
        _LOGGER.warning("Unable to sort protocol entry points for %s: %s", group, exc)
        if __debug__:
            _LOGGER.debug("Error sorting protocol entry points for %s", group, exc_info=True)
        return discovered


def _materialize_component(component_factory: object) -> Any:
    if isinstance(component_factory, type):
        return component_factory()

    if callable(component_factory):
        return component_factory()

    return component_factory


def _is_valid_protocol(protocol: object) -> bool:
    return (
        hasattr(protocol, "contract")
        and hasattr(protocol, "supports")
        and hasattr(protocol, "normalize_context")
        and hasattr(protocol, "validate_context")
    )


def _iter_discovered_protocols() -> Iterator[tuple[str, object]]:
    """Yield (name, protocol_instance) pairs from discovered protocol backends."""
    try:
        discovered_points = tuple(_iter_entry_points(_PROTOCOL_ENTRYPOINT_GROUP))
    except Exception as exc:
        _LOGGER.warning(
            "Unable to iterate protocol entry points for %s: %s",
            _PROTOCOL_ENTRYPOINT_GROUP,
            exc,
        )
        if __debug__:
            _LOGGER.debug("Error iterating protocol entry points", exc_info=True)
        return

    for point in discovered_points:
        try:
            loaded = point.load()
            protocol = _materialize_component(loaded)
        except Exception as exc:
            _LOGGER.warning(
                "Skipping protocol entry point '%s' (%s): %s",
                point.name,
                _PROTOCOL_ENTRYPOINT_GROUP,
                exc,
            )
            if __debug__:
                _LOGGER.debug("Protocol entry point load failure", exc_info=True)
            continue

        if not _is_valid_protocol(protocol):
            _LOGGER.warning(
                "Skipping protocol entry point '%s': protocol interface missing required attributes",
                point.name,
            )
            if __debug__:
                _LOGGER.debug("Invalid protocol candidate: %r", protocol)
            continue

        yield point.name, protocol


def create_default_protocol_registry() -> ProtocolRegistry:
    registry = ProtocolRegistry()
    registry.register(SurfaceProtocol())
    registry.register(NISQProtocol())
    registry.register(SQKDProtocol())

    for protocol_name, protocol in sorted(
        _iter_discovered_protocols(),
        key=lambda item: item[0],
    ):
        if protocol.contract.name != protocol_name:
            _LOGGER.warning(
                "Overriding protocol entry point name '%s' with contract name '%s'",
                protocol_name,
                protocol.contract.name,
            )

        try:
            registry.register(protocol)
        except Exception as exc:
            _LOGGER.warning("Skipping protocol '%s': %s", protocol_name, exc)
            if __debug__:
                _LOGGER.debug("Error registering dynamic protocol backend", exc_info=True)

    return registry


DEFAULT_PROTOCOL_REGISTRY = create_default_protocol_registry()

__all__ = [
    "ProtocolRegistry",
    "ProtocolContract",
    "SurfaceProtocol",
    "NISQProtocol",
    "SQKDProtocol",
    "create_default_protocol_registry",
    "DEFAULT_PROTOCOL_REGISTRY",
]
