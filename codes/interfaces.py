"""Shared plugin interfaces for quantum error-correction code families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


DecoderFn = Callable[[str, int, int | None], Mapping[str, Any]]


@dataclass(frozen=True)
class CircuitGenerationConfig:
    """Configuration used by code-family plugins to generate circuits.

    Attributes:
        distance: Code distance or analogous size parameter.
        rounds: Number of syndrome-extraction rounds.
        physical_error_rate: Per-operation physical error rate.
        extra_params: Family-specific structured parameters.
    """

    distance: int
    rounds: int
    physical_error_rate: float
    extra_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SyndromeExtractionSpec:
    """Schema for syndrome extraction outputs expected by decoders."""

    format_name: str
    detector_axis: str
    observable_axis: str
    supports_separate_observables: bool


@dataclass(frozen=True)
class DecoderCompatibilityMetadata:
    """Metadata describing decoder expectations for a code family."""

    family: str
    compatible_decoders: tuple[str, ...]
    required_inputs: tuple[str, ...]
    syndrome_spec: SyndromeExtractionSpec


class CodeFamilyPlugin(ABC):
    """Base interface that all code-family plugins implement."""

    family: str

    @abstractmethod
    def build_circuit(self, config: CircuitGenerationConfig) -> str:
        """Build a Stim-compatible circuit string for the given configuration."""

    @abstractmethod
    def syndrome_spec(self) -> SyndromeExtractionSpec:
        """Return the syndrome extraction schema for the family."""

    @abstractmethod
    def decoder_metadata(self) -> DecoderCompatibilityMetadata:
        """Return decoder compatibility details for the family."""
