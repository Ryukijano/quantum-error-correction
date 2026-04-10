"""Environment builder contracts used by RL training strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from gymnasium import Env


@dataclass(frozen=True)
class EnvBuildContext:
    """Shared context for environment construction."""

    distance: int
    rounds: int
    physical_error_rate: float
    seed: int | None = None

    use_mwpm_baseline: bool = True
    use_soft_information: bool = False
    use_accelerated_sampling: bool = False
    sampling_backend: str | None = None
    protocol: str = "surface"
    decoder_name: str | None = None
    enable_profile_traces: bool = False
    benchmark_probe_token: str | None = None
    protocol_metadata: dict[str, Any] = field(default_factory=dict)

    parameter_dim: int = 4
    batch_shots: int = 128
    base_error_rate: float = 0.001

    circuit_type: str = "tri"
    use_superdense: bool = False

    max_distance: int = 13
    min_distance: int = 3
    max_rounds: int = 10
    target_threshold: float = 0.005
    max_steps: int = 20


class EnvironmentBuilder(Protocol):
    """Protocol for constructing concrete gym environments."""

    name: str

    def build(self, context: EnvBuildContext) -> Env:
        ...


class EnvBuilderRegistry:
    """Registry of environment builders used by training strategies."""

    def __init__(self) -> None:
        self._builders: dict[str, EnvironmentBuilder] = {}

    def register(self, builder: EnvironmentBuilder) -> None:
        if builder.name in self._builders:
            raise ValueError(f"Builder '{builder.name}' already registered.")
        self._builders[builder.name] = builder

    def get(self, name: str) -> EnvironmentBuilder:
        if name not in self._builders:
            raise KeyError(f"Unknown environment builder '{name}'.")
        return self._builders[name]

    def list(self) -> list[str]:
        return sorted(self._builders.keys())
