"""Code-family plugin package for shared circuit/decoder workflows."""

from .benchmark import benchmark_code_families, default_stim_decoder_evaluator
from .interfaces import (
    CircuitGenerationConfig,
    CodeFamilyPlugin,
    DecoderCompatibilityMetadata,
    SyndromeExtractionSpec,
)
from .registry import get_plugin, list_plugins, register_plugin
from .bosonic import BosonicCodePlugin
from .dual_rail_erasure import DualRailErasureCodePlugin
from .qldpc import QLDPCCodePlugin
from .surface import SurfaceCodePlugin


register_plugin(SurfaceCodePlugin())
register_plugin(QLDPCCodePlugin())
register_plugin(BosonicCodePlugin())
register_plugin(DualRailErasureCodePlugin())


__all__ = [
    "CircuitGenerationConfig",
    "CodeFamilyPlugin",
    "DecoderCompatibilityMetadata",
    "SyndromeExtractionSpec",
    "register_plugin",
    "get_plugin",
    "list_plugins",
    "benchmark_code_families",
    "default_stim_decoder_evaluator",
    "SurfaceCodePlugin",
    "QLDPCCodePlugin",
    "BosonicCodePlugin",
    "DualRailErasureCodePlugin",
]
