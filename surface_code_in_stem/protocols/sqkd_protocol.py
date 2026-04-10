"""SQKD protocol scaffolding for quantum key distribution flows."""

from __future__ import annotations

from dataclasses import replace

from surface_code_in_stem.rl_control.envs.base import EnvBuildContext
from surface_code_in_stem.protocols.base import ProtocolContract, QuantumProtocol


class SQKDProtocol:
    """Experimental SQKD protocol contract."""

    contract = ProtocolContract(
        name="sqkd",
        family="qkd",
        description="Pluggable contract for satellite QKD-oriented execution flows.",
        capabilities=["bb84_mapping", "satellite_channel", "basis_sifting"],
    )

    def supports(self, context: EnvBuildContext) -> bool:
        return context.distance >= 3 and context.rounds >= 1

    def normalize_context(self, context: EnvBuildContext) -> EnvBuildContext:
        metadata = dict(context.protocol_metadata)
        metadata.setdefault("qkd_mode", "bb84")
        return replace(context, protocol_metadata=metadata)

    def validate_context(self, context: EnvBuildContext) -> None:
        self.contract.validate_context(context)
        if context.distance < 3:
            raise ValueError("SQKD protocol currently requires distance >= 3.")
