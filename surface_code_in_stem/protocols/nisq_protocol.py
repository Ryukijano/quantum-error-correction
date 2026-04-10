"""NISQ protocol scaffolding for future superconducting or NISQ flows."""

from __future__ import annotations

from dataclasses import replace

from surface_code_in_stem.rl_control.envs.base import EnvBuildContext
from surface_code_in_stem.protocols.base import ProtocolContract, QuantumProtocol


class NISQProtocol:
    """Experimental NISQ protocol contract."""

    contract = ProtocolContract(
        name="nisq",
        family="noise_aware",
        description="Pluggable contract for NISQ-oriented execution flows.",
        capabilities=["nqubits", "crosstalk", "noise_models"],
    )

    def supports(self, context: EnvBuildContext) -> bool:
        # Keep parity-agnostic for future adapters.
        return context.distance >= 2 and context.rounds >= 1

    def normalize_context(self, context: EnvBuildContext) -> EnvBuildContext:
        # NISQ workflows often benefit from slightly deeper rounds.
        normalized_rounds = max(context.rounds, 2)
        return replace(context, rounds=normalized_rounds)

    def validate_context(self, context: EnvBuildContext) -> None:
        self.contract.validate_context(context)
        if context.distance <= 1:
            raise ValueError("NISQ protocol requires distance > 1.")
