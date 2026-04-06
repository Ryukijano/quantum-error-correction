"""Dependency injection container for syndrome-net.

Provides a centralized container for managing component dependencies,
enabling loose coupling and easier testing.
"""
from __future__ import annotations

from typing import Any, TypeVar, Callable
from dataclasses import dataclass, field

from syndrome_net import CircuitBuilder, Decoder, NoiseModel, Visualizer
from syndrome_net.registry import (
    CircuitBuilderRegistry,
    DecoderRegistry,
    NoiseModelRegistry,
    VisualizerRegistry,
)

T = TypeVar("T")


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
        from syndrome_net.decoders import MWPMDecoder, UnionFindDecoder
        from syndrome_net.noise import IIDDepolarizingModel, BiasedNoiseModel
        from syndrome_net.visualizers import PlotlyVisualizer, SVGVisualizer
        
        # Circuit builders
        self._circuit_builders.register("surface", SurfaceCodeBuilder())
        self._circuit_builders.register("hexagonal", HexagonalCodeBuilder())
        self._circuit_builders.register("walking", WalkingCodeBuilder())
        self._circuit_builders.register("iswap", ISwapCodeBuilder())
        self._circuit_builders.register("xyz2", XYZ2HexagonalBuilder())
        
        # Colour code builders (graceful if deps missing)
        try:
            self._circuit_builders.register("color_code", ColorCodeStimBuilder())
        except Exception:  # color-code-stim not installed
            pass
        try:
            self._circuit_builders.register("loom_color_code", LoomColorCodeBuilder())
        except Exception:  # el-loom not installed
            pass
        
        # Decoders
        self._decoders.register("mwpm", MWPMDecoder())
        self._decoders.register("union_find", UnionFindDecoder())
        
        # Noise models
        self._noise_models.register("depolarizing", IIDDepolarizingModel())
        self._noise_models.register("biased", BiasedNoiseModel())
        
        # Visualizers
        self._visualizers.register("plotly", PlotlyVisualizer())
        self._visualizers.register("svg", SVGVisualizer())
    
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
