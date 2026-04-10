"""Surface-code protocol adapter for existing RL/decoder workflows."""

from __future__ import annotations

from dataclasses import replace

from surface_code_in_stem.rl_control.envs.base import EnvBuildContext
from surface_code_in_stem.protocols.base import ProtocolContract, QuantumProtocol


class SurfaceProtocol:
    """Default protocol using surface code environments."""

    contract = ProtocolContract(
        name="surface",
        family="surface",
        description="Baseline surface-code simulator protocol.",
        capabilities=["qec", "decoder", "sampling"],
    )

    def supports(self, context: EnvBuildContext) -> bool:
        return context.distance % 2 == 1 and context.distance >= 3

    def normalize_context(self, context: EnvBuildContext) -> EnvBuildContext:
        # Keep deterministic odd-distance surfaces.
        normalized_distance = context.distance if context.distance % 2 == 1 else context.distance + 1
        if normalized_distance != context.distance:
            context = replace(context, distance=normalized_distance)
        return context

    def validate_context(self, context: EnvBuildContext) -> None:
        self.contract.validate_context(context)
        if not self.supports(context):
            raise ValueError("Surface protocol requires odd distance >= 3.")
