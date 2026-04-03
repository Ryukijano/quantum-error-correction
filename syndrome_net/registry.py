"""Plugin registry system for syndrome-net components.

Provides type-safe registries for circuit builders, decoders, noise models,
and visualizers. Enables runtime discovery and selection of components.
"""
from __future__ import annotations

from typing import TypeVar, Generic, Iterator
from collections.abc import Mapping

from syndrome_net import (
    CircuitBuilder,
    Decoder,
    NoiseModel,
    Visualizer,
    SyndromeNetError,
)

T = TypeVar("T")


class Registry(Generic[T], Mapping[str, T]):
    """Generic type-safe registry for syndrome-net components.
    
    Supports dict-like access while enforcing component protocols.
    
    Example:
        >>> registry = Registry[CircuitBuilder]()
        >>> registry.register("surface", SurfaceCodeBuilder())
        >>> builder = registry["surface"]
        >>> "hexagonal" in registry
        False
    """
    
    def __init__(self) -> None:
        self._items: dict[str, T] = {}
    
    def register(self, name: str, item: T) -> None:
        """Register a component with the given name.
        
        Args:
            name: Unique identifier for the component
            item: Component instance implementing the protocol
            
        Raises:
            ValueError: If name is already registered
            TypeError: If item doesn't implement the required protocol
        """
        if name in self._items:
            raise ValueError(f"Component '{name}' is already registered")
        self._items[name] = item
    
    def unregister(self, name: str) -> T:
        """Remove and return a registered component.
        
        Args:
            name: Name of the component to remove
            
        Returns:
            The removed component
            
        Raises:
            KeyError: If name is not registered
        """
        if name not in self._items:
            raise KeyError(f"No component registered with name '{name}'")
        return self._items.pop(name)
    
    def get(self, name: str) -> T:
        """Get a registered component by name.
        
        Args:
            name: Component name
            
        Returns:
            The registered component
            
        Raises:
            SyndromeNetError: If name is not registered
        """
        if name not in self._items:
            available = list(self._items.keys())
            raise SyndromeNetError(
                f"Unknown component '{name}'. "
                f"Available: {available}"
            )
        return self._items[name]
    
    def list(self) -> list[str]:
        """List all registered component names."""
        return list(self._items.keys())
    
    def clear(self) -> None:
        """Remove all registered components."""
        self._items.clear()
    
    def __getitem__(self, name: str) -> T:
        return self.get(name)
    
    def __contains__(self, name: object) -> bool:
        return name in self._items
    
    def __iter__(self) -> Iterator[str]:
        return iter(self._items)
    
    def __len__(self) -> int:
        return len(self._items)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self._items.keys())})"


class CircuitBuilderRegistry(Registry[CircuitBuilder]):
    """Registry for quantum error correction circuit builders.
    
    Provides convenient access to QEC code implementations.
    
    Example:
        >>> from syndrome_net.codes import SurfaceCodeBuilder
        >>> registry = CircuitBuilderRegistry()
        >>> registry.register("surface", SurfaceCodeBuilder())
        >>> builder = registry.get("surface")
        >>> circuit = builder.build(CircuitSpec(distance=3, rounds=5, error_probability=0.001))
    """
    
    def get_supported_distances(self, name: str) -> list[int]:
        """Get valid distances for a specific builder.
        
        Args:
            name: Builder name
            
        Returns:
            List of supported distances
        """
        builder = self.get(name)
        return builder.supported_distances
    
    def list_dynamic(self) -> list[str]:
        """List builders for dynamic codes (Floquet, etc.)."""
        return [
            name for name in self.list()
            if self.get(name).is_dynamic
        ]


class DecoderRegistry(Registry[Decoder]):
    """Registry for syndrome decoders.
    
    Enables runtime selection of decoding algorithms.
    """
    pass


class NoiseModelRegistry(Registry[NoiseModel]):
    """Registry for noise models.
    
    Supports different error channels (depolarizing, biased, etc.).
    """
    pass


class VisualizerRegistry(Registry[Visualizer]):
    """Registry for visualization backends.
    
    Supports multiple rendering engines (Plotly, SVG, WebGL, etc.).
    """
    pass


class CompositeRegistry:
    """Container for all syndrome-net registries.
    
    Provides a single point of access to all component types.
    
    Example:
        >>> registry = CompositeRegistry()
        >>> registry.circuit_builders.register("surface", SurfaceCodeBuilder())
        >>> registry.decoders.register("mwpm", MWPMDecoder())
    """
    
    def __init__(self) -> None:
        self.circuit_builders = CircuitBuilderRegistry()
        self.decoders = DecoderRegistry()
        self.noise_models = NoiseModelRegistry()
        self.visualizers = VisualizerRegistry()
    
    def register_defaults(self) -> None:
        """Register all default syndrome-net components."""
        # Import and register default components
        from syndrome_net.codes import (
            SurfaceCodeBuilder,
            HexagonalCodeBuilder,
            WalkingCodeBuilder,
            ISwapCodeBuilder,
            XYZ2HexagonalBuilder,
        )
        from syndrome_net.decoders import MWPMDecoder, UnionFindDecoder
        from syndrome_net.noise import IIDDepolarizingModel, BiasedNoiseModel
        from syndrome_net.visualizers import PlotlyVisualizer
        
        # Circuit builders
        self.circuit_builders.register("surface", SurfaceCodeBuilder())
        self.circuit_builders.register("hexagonal", HexagonalCodeBuilder())
        self.circuit_builders.register("walking", WalkingCodeBuilder())
        self.circuit_builders.register("iswap", ISwapCodeBuilder())
        self.circuit_builders.register("xyz2", XYZ2HexagonalBuilder())
        
        # Decoders
        self.decoders.register("mwpm", MWPMDecoder())
        self.decoders.register("union_find", UnionFindDecoder())
        
        # Noise models
        self.noise_models.register("depolarizing", IIDDepolarizingModel())
        self.noise_models.register("biased", BiasedNoiseModel())
        
        # Visualizers
        self.visualizers.register("plotly", PlotlyVisualizer())
