"""Builders for colour-code gym environments."""

from __future__ import annotations

from .base import EnvBuildContext, EnvironmentBuilder
from surface_code_in_stem.rl_control.gym_env import (
    ColourCodeCalibrationEnv,
    ColourCodeDiscoveryEnv,
    ColourCodeGymEnv,
)


class ColourCodeGymEnvBuilder(EnvironmentBuilder):
    """Builder for colour-code decoding environment."""

    name = "colour_gym"

    def build(self, context: EnvBuildContext):
        return ColourCodeGymEnv(
            distance=context.distance,
            rounds=context.rounds,
            physical_error_rate=context.physical_error_rate,
            circuit_type=context.circuit_type,
            seed=context.seed,
            use_mwpm_baseline=context.use_mwpm_baseline,
            use_superdense=context.use_superdense,
        )


class ColourCodeCalibrationEnvBuilder(EnvironmentBuilder):
    """Builder for colour-code calibration environment."""

    name = "colour_calibration"

    def build(self, context: EnvBuildContext):
        return ColourCodeCalibrationEnv(
            distance=context.distance,
            rounds=context.rounds,
            circuit_type=context.circuit_type,
            parameter_dim=context.parameter_dim,
            batch_shots=context.batch_shots,
            base_error_rate=context.base_error_rate,
            seed=context.seed,
        )


class ColourCodeDiscoveryEnvBuilder(EnvironmentBuilder):
    """Builder for colour-code architecture-discovery environment."""

    name = "colour_discovery"

    def build(self, context: EnvBuildContext):
        return ColourCodeDiscoveryEnv(
            max_distance=context.max_distance,
            min_distance=context.min_distance,
            max_rounds=context.max_rounds,
            target_threshold=context.target_threshold,
            max_steps=context.max_steps,
        )
