"""Composable environment builders for RL strategies."""

import logging
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Iterable, Iterator

from .base import EnvBuildContext, EnvBuilderRegistry, EnvironmentBuilder
from .colour import (
    ColourCodeCalibrationEnvBuilder,
    ColourCodeDiscoveryEnvBuilder,
    ColourCodeGymEnvBuilder,
)
from .qec import QECContinuousControlEnvBuilder, QECGymEnvBuilder

_LOGGER = logging.getLogger(__name__)
_ENV_BUILDER_ENTRYPOINT_GROUP = "syndrome_net.environment_builders"


def _iter_entry_points(group: str) -> Iterable[EntryPoint]:
    """Return entry-point definitions for discovery while handling import failures."""
    try:
        points = entry_points()
    except Exception as exc:
        _LOGGER.debug("Unable to enumerate entry points for %s: %s", group, exc)
        return ()

    if hasattr(points, "select"):
        try:
            return points.select(group=group)
        except Exception as exc:  # pragma: no cover - compatibility fallback
            _LOGGER.debug("Failed to select entry points for %s: %s", group, exc)
            return ()

    return points.get(group, ())


def _materialize_component(component_factory: object) -> Any:
    if isinstance(component_factory, type):
        return component_factory()

    if callable(component_factory):
        return component_factory()

    return component_factory


def _is_valid_builder(builder: object) -> bool:
    return hasattr(builder, "name") and hasattr(builder, "build") and callable(builder.build)


def _iter_discovered_builders() -> Iterator[tuple[str, object]]:
    """Yield (name, builder_instance) pairs from discovered env builders."""
    try:
        discovered_points = tuple(_iter_entry_points(_ENV_BUILDER_ENTRYPOINT_GROUP))
    except Exception as exc:
        _LOGGER.warning(
            "Unable to iterate env-builder entry points for %s: %s",
            _ENV_BUILDER_ENTRYPOINT_GROUP,
            exc,
        )
        if __debug__:
            _LOGGER.debug("Error iterating env-builder entry points", exc_info=True)
        return

    for point in discovered_points:
        try:
            loaded = point.load()
            builder = _materialize_component(loaded)
        except Exception as exc:
            _LOGGER.warning(
                "Skipping env-builder entry point '%s' (%s): %s",
                point.name,
                _ENV_BUILDER_ENTRYPOINT_GROUP,
                exc,
            )
            if __debug__:
                _LOGGER.debug("Environment builder entry point load failure", exc_info=True)
            continue

        if not _is_valid_builder(builder):
            _LOGGER.warning(
                "Skipping env-builder entry point '%s': env builder interface missing required attributes",
                point.name,
            )
            if __debug__:
                _LOGGER.debug("Invalid env-builder candidate: %r", builder)
            continue

        yield point.name, builder


def default_builder_registry() -> EnvBuilderRegistry:
    """Create a default registry with all supported environment builders."""
    registry = EnvBuilderRegistry()
    registry.register(QECGymEnvBuilder())
    registry.register(QECContinuousControlEnvBuilder())
    registry.register(ColourCodeGymEnvBuilder())
    registry.register(ColourCodeCalibrationEnvBuilder())
    registry.register(ColourCodeDiscoveryEnvBuilder())

    for builder_name, builder in _iter_discovered_builders():
        try:
            registry.register(builder)
        except Exception as exc:
            _LOGGER.warning("Skipping env builder '%s': %s", builder_name, exc)
            if __debug__:
                _LOGGER.debug("Error registering dynamic env builder", exc_info=True)

    return registry


__all__ = [
    "EnvBuildContext",
    "EnvBuilderRegistry",
    "EnvironmentBuilder",
    "default_builder_registry",
    "QECGymEnvBuilder",
    "QECContinuousControlEnvBuilder",
    "ColourCodeGymEnvBuilder",
    "ColourCodeCalibrationEnvBuilder",
    "ColourCodeDiscoveryEnvBuilder",
]
