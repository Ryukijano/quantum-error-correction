"""Builders for standard QEC gym environments."""

from __future__ import annotations

from .base import EnvBuildContext, EnvironmentBuilder
from surface_code_in_stem.rl_control.gym_env import QECGymEnv, QECContinuousControlEnv


class QECGymEnvBuilder(EnvironmentBuilder):
    """Builder for the standard one-shot surface-code decoding environment."""

    name = "qec"

    def build(self, context: EnvBuildContext):
        return QECGymEnv(
            distance=context.distance,
            rounds=context.rounds,
            physical_error_rate=context.physical_error_rate,
            seed=context.seed,
            use_mwpm_baseline=context.use_mwpm_baseline,
            use_soft_information=context.use_soft_information,
            use_accelerated_sampling=context.use_accelerated_sampling,
            sampling_backend=context.sampling_backend,
            protocol_metadata=context.protocol_metadata,
            enable_profile_traces=context.enable_profile_traces,
            benchmark_probe_token=context.benchmark_probe_token,
            decoder_name=context.decoder_name,
        )


class QECContinuousControlEnvBuilder(EnvironmentBuilder):
    """Builder for the continuous calibration environment."""

    name = "qec_continuous"

    def build(self, context: EnvBuildContext):
        base_error_rate = context.base_error_rate if context.base_error_rate > 0 else context.physical_error_rate
        return QECContinuousControlEnv(
            distance=context.distance,
            rounds=context.rounds,
            parameter_dim=context.parameter_dim,
            batch_shots=context.batch_shots,
            base_error_rate=base_error_rate,
            seed=context.seed,
        )
