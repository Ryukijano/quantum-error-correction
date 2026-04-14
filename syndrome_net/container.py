"""Dependency injection container for syndrome-net.

Provides a centralized container for managing component dependencies,
enabling loose coupling and easier testing.
"""
from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points
from typing import Any, Callable, Iterable, Iterator, TypeVar
from dataclasses import dataclass
import logging
_LOGGER = logging.getLogger(__name__)



from syndrome_net import CircuitBuilder, Decoder, NoiseModel, Visualizer
from syndrome_net.registry import (
    CircuitBuilderRegistry,
    DecoderRegistry,
    NoiseModelRegistry,
    VisualizerRegistry,
)

T = TypeVar("T")

_CIRCUIT_BUILDER_ENTRYPOINT_GROUP = "syndrome_net.circuit_builders"
_DECODER_ENTRYPOINT_GROUP = "syndrome_net.decoders"


def _iter_entry_points(group: str) -> Iterable[EntryPoint]:
    """Return entry-point definitions for discovery while handling unsupported runtimes."""
    try:
        points = entry_points()
    except Exception as exc:
        _LOGGER.warning("Unable to enumerate entry points for %s: %s", group, exc)
        if __debug__:
            _LOGGER.debug("Error enumerating entry points for %s", group, exc_info=True)
        return ()

    if hasattr(points, "select"):
        try:
            raw_points = tuple(points.select(group=group))
        except Exception as exc:  # pragma: no cover - compatibility fallback
            _LOGGER.warning("Failed to select entry points for %s: %s", group, exc)
            if __debug__:
                _LOGGER.debug("Error selecting entry points for %s", group, exc_info=True)
            return ()
        try:
            return tuple(sorted(raw_points, key=lambda point: point.name))
        except Exception as exc:  # pragma: no cover - compatibility fallback
            _LOGGER.warning("Unable to sort discovered entry points for %s: %s", group, exc)
            if __debug__:
                _LOGGER.debug("Error sorting entry points for %s", group, exc_info=True)
            return raw_points

    try:
        discovered = tuple(points.get(group, ()))
    except Exception as exc:  # pragma: no cover - compatibility fallback
        _LOGGER.warning("Unable to read entry points for %s: %s", group, exc)
        if __debug__:
            _LOGGER.debug("Error reading entry points for %s", group, exc_info=True)
        return ()

    try:
        return tuple(sorted(discovered, key=lambda point: point.name))
    except Exception as exc:  # pragma: no cover - compatibility fallback
        _LOGGER.warning("Unable to sort discovered entry points for %s: %s", group, exc)
        if __debug__:
            _LOGGER.debug("Error sorting entry points for %s", group, exc_info=True)
        return discovered


def _safe_optional_component_available(
    component_name: str,
    available: Callable[[], bool],
) -> bool:
    """Probe optional-component availability with robust fallback logging."""
    try:
        return bool(available())
    except Exception as exc:
        _LOGGER.warning("Unable to probe optional component '%s': %s", component_name, exc)
        if __debug__:
            _LOGGER.debug("Optional component probe failure", exc_info=True)
        return False


def _materialize_component(component_factory: object) -> Any:
    """Create a component instance from a builder-like object."""
    if isinstance(component_factory, type):
        return component_factory()

    if callable(component_factory):
        return component_factory()

    return component_factory


def _iter_discovered_circuit_builders() -> Iterator[tuple[str, Any]]:
    """Yield (name, builder_instance) pairs from entry-point discovery."""
    for point in _iter_entry_points(_CIRCUIT_BUILDER_ENTRYPOINT_GROUP):
        try:
            loaded = point.load()
        except Exception as exc:
            _LOGGER.warning(
                "Skipping circuit-builder entry point '%s' (%s): %s",
                point.name,
                _CIRCUIT_BUILDER_ENTRYPOINT_GROUP,
                exc,
            )
            if __debug__:
                _LOGGER.debug("Error loading circuit-builder entry point", exc_info=True)
            continue

        try:
            yield point.name, _materialize_component(loaded)
        except Exception as exc:  # pragma: no cover - defensive, exercised via tests
            _LOGGER.warning("Skipping circuit-builder '%s': %s", point.name, exc)
            if __debug__:
                _LOGGER.debug("Error materializing circuit-builder entry point", exc_info=True)


def _iter_discovered_decoders() -> Iterator[tuple[str, Any]]:
    """Yield (name, decoder_instance) pairs from entry-point discovery."""
    for point in _iter_entry_points(_DECODER_ENTRYPOINT_GROUP):
        try:
            loaded = point.load()
        except Exception as exc:
            _LOGGER.warning(
                "Skipping decoder entry point '%s' (%s): %s",
                point.name,
                _DECODER_ENTRYPOINT_GROUP,
                exc,
            )
            if __debug__:
                _LOGGER.debug("Error loading decoder entry point", exc_info=True)
            continue

        try:
            yield point.name, _materialize_component(loaded)
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.warning("Skipping decoder '%s': %s", point.name, exc)
            if __debug__:
                _LOGGER.debug("Error materializing decoder entry point", exc_info=True)


def _is_valid_circuit_builder(candidate: object) -> bool:
    """Return True when a candidate can be registered as a circuit builder."""
    return (
        hasattr(candidate, "name")
        and hasattr(candidate, "build")
        and hasattr(candidate, "supported_distances")
        and callable(candidate.build)
    )


def _is_valid_decoder(candidate: object) -> bool:
    return (
        hasattr(candidate, "name")
        and hasattr(candidate, "decode")
        and callable(candidate.decode)
    )


def _register_optional_builder(
    registry: "CircuitBuilderRegistry",
    name: str,
    factory: Callable[[], Any],
    *,
    fallback_message: str,
) -> None:
    """Register a builder if available; otherwise log a warning and continue."""
    try:
        registry.register(name, factory())
        return
    except Exception as exc:
        _LOGGER.warning("%s: %s", fallback_message, exc)
        if __debug__:
            _LOGGER.debug(
                "%s builder details", fallback_message.lower().replace(" ", "_"), exc_info=True
            )


def _register_optional_builder_if_available(
    registry: "CircuitBuilderRegistry",
    name: str,
    builder_type: Callable[[], Any],
    component_name: str,
    availability_probe: Callable[[], bool],
    unavailable_message: str,
) -> None:
    if not _safe_optional_component_available(
        component_name=component_name,
        available=availability_probe,
    ):
        _LOGGER.warning("%s", unavailable_message)
        return

    _register_optional_builder(
        registry,
        name,
        builder_type,
        fallback_message=unavailable_message,
    )


def _register_discovered_components(
    registry: Any,
    discovered: Iterable[tuple[str, Any]],
    *,
    validate: Callable[[Any], bool],
    is_duplicate: Callable[[str], bool],
    type_label: str,
) -> None:
    for item_name, item in sorted(discovered, key=lambda pair: pair[0]):
        if is_duplicate(item_name):
            _LOGGER.debug("Skipping duplicate %s entry '%s'", type_label, item_name)
            continue
        if not validate(item):
            _LOGGER.warning("Skipping invalid %s '%s': missing interface", type_label, item_name)
            if __debug__:
                _LOGGER.debug("Invalid %s candidate: %r", type_label, item)
            continue
        try:
            registry.register(item_name, item)
        except Exception as exc:
            _LOGGER.warning("Skipping %s '%s': %s", type_label, item_name, exc)
            if __debug__:
                _LOGGER.debug("Error registering %s '%s'", type_label, item_name, exc_info=True)


@dataclass
class ContainerConfig:
    """Configuration for the dependency injection container."""
    
    # Default component selections
    default_builder: str = "surface"
    default_decoder: str = "mwpm"
    default_noise_model: str = "depolarizing"
    default_visualizer: str = "plotly"
    
    # Feature flags
    enable_parallel: bool = True
    enable_jit: bool = True
    cache_circuits: bool = True


class DIContainer:
    """Dependency injection container for syndrome-net components.
    
    Provides centralized access to all QEC components with support
    for lazy initialization, mocking, and configuration-based setup.
    
    Example:
        >>> container = DIContainer()
        >>> container.register_defaults()
        >>> builder = container.circuit_builders.get("surface")
        >>> decoder = container.decoders.get("mwpm")
    """
    
    def __init__(self, config: ContainerConfig | None = None) -> None:
        """Initialize the DI container.
        
        Args:
            config: Container configuration (uses defaults if None)
        """
        self.config = config or ContainerConfig()
        
        # Component registries
        self._circuit_builders = CircuitBuilderRegistry()
        self._decoders = DecoderRegistry()
        self._noise_models = NoiseModelRegistry()
        self._visualizers = VisualizerRegistry()
        
        # Factories for lazy initialization
        self._factories: dict[str, Callable[[], Any]] = {}
        
        # Cache for expensive objects
        self._cache: dict[str, Any] = {}
    
    @property
    def circuit_builders(self) -> CircuitBuilderRegistry:
        """Access the circuit builder registry."""
        return self._circuit_builders
    
    @property
    def decoders(self) -> DecoderRegistry:
        """Access the decoder registry."""
        return self._decoders
    
    @property
    def noise_models(self) -> NoiseModelRegistry:
        """Access the noise model registry."""
        return self._noise_models
    
    @property
    def visualizers(self) -> VisualizerRegistry:
        """Access the visualizer registry."""
        return self._visualizers
    
    def register_defaults(self) -> None:
        """Register all default syndrome-net components."""
        # Import here to avoid circular dependencies
        from syndrome_net.codes import (
            SurfaceCodeBuilder,
            HexagonalCodeBuilder,
            WalkingCodeBuilder,
            ISwapCodeBuilder,
            XYZ2HexagonalBuilder,
            ColorCodeStimBuilder,
            LoomColorCodeBuilder,
        )
        from surface_code_in_stem.decoders import (
            CudaQDecoder,
            CuQNNBackendAdapterDecoder,
            MWPMDecoder,
            JAXConfidenceDecoderAdapter,
            SparseBlossomDecoder,
            QuJaxNeuralBPDecoder,
            UnionFindDecoder,
            IsingDecoder,
        )
        from syndrome_net.noise import IIDDepolarizingModel, BiasedNoiseModel
        from syndrome_net.visualizers import CrumbleVisualizer, PlotlyVisualizer, SVGVisualizer
        
        # Circuit builders
        self._circuit_builders.register("surface", SurfaceCodeBuilder())
        self._circuit_builders.register("hexagonal", HexagonalCodeBuilder())
        self._circuit_builders.register("walking", WalkingCodeBuilder())
        self._circuit_builders.register("iswap", ISwapCodeBuilder())
        self._circuit_builders.register("xyz2", XYZ2HexagonalBuilder())

        # Colour code builders (graceful fallback if optional dependencies are missing)
        _register_optional_builder_if_available(
            self._circuit_builders,
            "color_code",
            ColorCodeStimBuilder,
            component_name="color-code-stim builder",
            availability_probe=ColorCodeStimBuilder.is_available,
            unavailable_message="color-code-stim builder unavailable",
        )
        _register_optional_builder_if_available(
            self._circuit_builders,
            "loom_color_code",
            LoomColorCodeBuilder,
            component_name="el-loom builder",
            availability_probe=LoomColorCodeBuilder.is_available,
            unavailable_message="el-loom builder unavailable",
        )

        _register_discovered_components(
            self._circuit_builders,
            _iter_discovered_circuit_builders(),
            validate=_is_valid_circuit_builder,
            is_duplicate=lambda name: False,
            type_label="circuit-builder",
        )
        
        _register_discovered_components(
            self._decoders,
            _iter_discovered_decoders(),
            validate=_is_valid_decoder,
            is_duplicate=lambda name: name in self._decoders,
            type_label="decoder",
        )

        # Fallback defaults for built-in decoders (keeps behaviour stable if no
        # plugin metadata is available in a runtime environment).
        default_decoders = {
            "mwpm": MWPMDecoder,
            "union_find": UnionFindDecoder,
            "sparse_blossom": SparseBlossomDecoder,
            "cudaq": CudaQDecoder,
            "qujax": QuJaxNeuralBPDecoder,
            "cuqnn": CuQNNBackendAdapterDecoder,
            "jax_confidence": JAXConfidenceDecoderAdapter,
            "ising": IsingDecoder,
        }
        for decoder_name, decoder_factory in default_decoders.items():
            if decoder_name in self._decoders:
                continue
            try:
                self._decoders.register(decoder_name, decoder_factory())
            except Exception as exc:
                _LOGGER.warning("Skipping default decoder '%s': %s", decoder_name, exc)
                if __debug__:
                    _LOGGER.debug("Error registering default decoder '%s'", decoder_name, exc_info=True)
        
        # Noise models
        self._noise_models.register("depolarizing", IIDDepolarizingModel())
        self._noise_models.register("biased", BiasedNoiseModel())
        
        # Visualizers
        self._visualizers.register("plotly", PlotlyVisualizer())
        self._visualizers.register("svg", SVGVisualizer())
        self._visualizers.register("crumble", CrumbleVisualizer())
    
    def get_builder(self, name: str | None = None) -> CircuitBuilder:
        """Get a circuit builder by name (or default).
        
        Args:
            name: Builder name (uses config.default_builder if None)
            
        Returns:
            Circuit builder instance
        """
        name = name or self.config.default_builder
        return self._circuit_builders.get(name)
    
    def get_decoder(self, name: str | None = None) -> Decoder:
        """Get a decoder by name (or default).
        
        Args:
            name: Decoder name (uses config.default_decoder if None)
            
        Returns:
            Decoder instance
        """
        name = name or self.config.default_decoder
        return self._decoders.get(name)
    
    def get_noise_model(self, name: str | None = None) -> NoiseModel:
        """Get a noise model by name (or default).
        
        Args:
            name: Noise model name (uses config.default_noise_model if None)
            
        Returns:
            Noise model instance
        """
        name = name or self.config.default_noise_model
        return self._noise_models.get(name)
    
    def get_visualizer(self, name: str | None = None) -> Visualizer:
        """Get a visualizer by name (or default).
        
        Args:
            name: Visualizer name (uses config.default_visualizer if None)
            
        Returns:
            Visualizer instance
        """
        name = name or self.config.default_visualizer
        return self._visualizers.get(name)
    
    def register_factory(self, name: str, factory: Callable[[], T]) -> None:
        """Register a factory function for lazy initialization.
        
        Args:
            name: Factory identifier
            factory: Callable that returns the component
        """
        self._factories[name] = factory
    
    def get_from_factory(self, name: str) -> Any:
        """Get or create a component from a registered factory.
        
        Uses caching if cache_circuits is enabled in config.
        
        Args:
            name: Factory identifier
            
        Returns:
            Component instance
            
        Raises:
            KeyError: If factory not registered
        """
        if name in self._cache and self.config.cache_circuits:
            return self._cache[name]
        
        if name not in self._factories:
            raise KeyError(f"No factory registered for '{name}'")
        
        instance = self._factories[name]()
        
        if self.config.cache_circuits:
            self._cache[name] = instance
        
        return instance
    
    def clear_cache(self) -> None:
        """Clear the component cache."""
        self._cache.clear()
    
    def create_scope(self) -> DIScope:
        """Create a new scoped container for request-level dependencies.
        
        Returns:
            Scoped container with isolated state
        """
        return DIScope(self)


class DIScope:
    """Scoped dependency container for request-level isolation.
    
    Useful for web applications where each request needs
    fresh component instances.
    
    Example:
        >>> with container.create_scope() as scope:
        ...     builder = scope.get_builder()
        ...     circuit = builder.build(spec)
    """
    
    def __init__(self, parent: DIContainer) -> None:
        """Initialize scope with parent container.
        
        Args:
            parent: Parent DI container
        """
        self._parent = parent
        self._scoped_instances: dict[str, Any] = {}
    
    def __enter__(self) -> DIScope:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._scoped_instances.clear()
    
    def get_builder(self, name: str | None = None) -> CircuitBuilder:
        """Get circuit builder (creates new instance if in scope)."""
        name = name or self._parent.config.default_builder
        cache_key = f"builder:{name}"
        
        if cache_key not in self._scoped_instances:
            builder = self._parent.get_builder(name)
            self._scoped_instances[cache_key] = builder
        
        return self._scoped_instances[cache_key]
    
    def get_decoder(self, name: str | None = None) -> Decoder:
        """Get decoder (creates new instance if in scope)."""
        name = name or self._parent.config.default_decoder
        cache_key = f"decoder:{name}"
        
        if cache_key not in self._scoped_instances:
            decoder = self._parent.get_decoder(name)
            self._scoped_instances[cache_key] = decoder
        
        return self._scoped_instances[cache_key]
    
    def get_noise_model(self, name: str | None = None) -> NoiseModel:
        """Get noise model (creates new instance if in scope)."""
        name = name or self._parent.config.default_noise_model
        cache_key = f"noise:{name}"
        
        if cache_key not in self._scoped_instances:
            noise = self._parent.get_noise_model(name)
            self._scoped_instances[cache_key] = noise
        
        return self._scoped_instances[cache_key]
    
    def get_visualizer(self, name: str | None = None) -> Visualizer:
        """Get visualizer (creates new instance if in scope)."""
        name = name or self._parent.config.default_visualizer
        cache_key = f"viz:{name}"
        
        if cache_key not in self._scoped_instances:
            viz = self._parent.get_visualizer(name)
            self._scoped_instances[cache_key] = viz
        
        return self._scoped_instances[cache_key]


# Global container instance (singleton pattern)
_global_container: DIContainer | None = None


def get_container() -> DIContainer:
    """Get the global DI container instance.
    
    Creates and configures the container on first call.
    
    Returns:
        Global DI container
    """
    global _global_container
    
    if _global_container is None:
        _global_container = DIContainer()
        _global_container.register_defaults()
    
    return _global_container


def set_container(container: DIContainer) -> None:
    """Set the global DI container instance.
    
    Useful for testing with mock containers.
    
    Args:
        container: Container to use as global
    """
    global _global_container
    _global_container = container


def reset_container() -> None:
    """Reset the global container to None.
    
    Useful for testing isolation.
    """
    global _global_container
    _global_container = None
