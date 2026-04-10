"""Protocol contracts for quantum execution workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from surface_code_in_stem.rl_control.envs.base import EnvBuildContext


@dataclass(frozen=True)
class ProtocolContract:
    """Execution protocol contract metadata."""

    name: str
    family: str
    description: str
    capabilities: list[str]

    def normalize_context(self, context: EnvBuildContext) -> EnvBuildContext:
        return context

    def validate_context(self, context: EnvBuildContext) -> None:
        if context.distance <= 0:
            raise ValueError("protocol requires distance > 0")
        if context.rounds <= 0:
            raise ValueError("protocol requires rounds > 0")


class QuantumProtocol(Protocol):
    """Extension point for protocol-specific execution behavior."""

    contract: ProtocolContract

    def supports(self, context: EnvBuildContext) -> bool:
        ...

    def normalize_context(self, context: EnvBuildContext) -> EnvBuildContext:
        ...

    def validate_context(self, context: EnvBuildContext) -> None:
        ...


class ProtocolRegistry:
    """Registry of quantum protocols available at runtime."""

    def __init__(self) -> None:
        self._protocols: dict[str, QuantumProtocol] = {}

    def register(self, protocol: QuantumProtocol) -> None:
        name = protocol.contract.name
        if name in self._protocols:
            raise ValueError(f"Protocol '{name}' already registered.")
        self._protocols[name] = protocol

    def get(self, name: str) -> QuantumProtocol:
        if name not in self._protocols:
            raise KeyError(f"Unknown protocol '{name}'.")
        return self._protocols[name]

    def list(self) -> list[str]:
        return sorted(self._protocols.keys())

