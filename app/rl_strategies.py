"""Strategy implementations for background RL training loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Any, Callable, Protocol, Mapping

import numpy as np

from surface_code_in_stem.rl_control.envs import (
    EnvBuildContext,
    EnvBuilderRegistry,
)
from surface_code_in_stem.protocols import ProtocolRegistry, DEFAULT_PROTOCOL_REGISTRY


EmitMetric = Callable[[dict[str, Any]], None]
EmitSyndrome = Callable[[np.ndarray, np.ndarray, bool, int], None]
EmitDone = Callable[[int], None]
EmitError = Callable[[str], None]
ShouldStop = Callable[[], bool]

def _safe_float(value: object) -> float | None:
    if isinstance(value, (int, float, np.number, bool)):
        converted = float(value)
        if np.isfinite(converted):
            return converted
    return None


def _sampling_trace_payload(env_info: dict[str, Any]) -> dict[str, Any]:
    sampling_backend = env_info.get("sampling_backend")
    if not isinstance(sampling_backend, dict):
        return {}

    payload = {
        "backend_id": sampling_backend.get("backend_id"),
        "backend_enabled": sampling_backend.get("backend_enabled"),
        "fallback_reason": sampling_backend.get("fallback_reason"),
    }
    backend_version = sampling_backend.get("backend_version")
    if isinstance(backend_version, str):
        payload["backend_version"] = backend_version

    sample_rate = _safe_float(sampling_backend.get("sample_rate"))
    if sample_rate is not None:
        payload["sample_rate"] = sample_rate

    trace_tokens = sampling_backend.get("trace_tokens")
    if isinstance(trace_tokens, list):
        payload["trace_tokens"] = trace_tokens

    details = sampling_backend.get("details")
    if isinstance(trace_tokens, list):
        payload["backend_chain"] = "->".join(str(token) for token in trace_tokens)
    else:
        backend_chain = sampling_backend.get("backend_chain")
        if backend_chain is None and isinstance(details, dict):
            backend_chain = details.get("backend_chain")
        if isinstance(backend_chain, str):
            payload["backend_chain"] = backend_chain
        elif isinstance(backend_chain, list):
            payload["backend_chain"] = "->".join(str(token) for token in backend_chain)

    if isinstance(details, dict):
        payload["details"] = details

    contract_flags = sampling_backend.get("contract_flags")
    if isinstance(contract_flags, str):
        payload["contract_flags"] = contract_flags

    profiler_flags = sampling_backend.get("profiler_flags")
    if isinstance(profiler_flags, str):
        payload["profiler_flags"] = profiler_flags

    sample_trace_id = sampling_backend.get("sample_trace_id")
    if sample_trace_id is None:
        sample_trace_id = env_info.get("sample_trace_id")
    if sample_trace_id is not None:
        payload["trace_id"] = sample_trace_id
        payload["sample_trace_id"] = sample_trace_id

    sample_us = _safe_float(env_info.get("sample_us"))
    if sample_us is None:
        sample_us = _safe_float(sampling_backend.get("sample_us"))
    if sample_us is not None:
        payload["sample_us"] = sample_us

    if "contract_flags" not in payload:
        if payload.get("backend_enabled") and payload.get("fallback_reason") is None:
            payload["contract_flags"] = "backend_enabled,contract_met"
        else:
            payload["contract_flags"] = "backend_disabled,contract_fallback"

    if "profiler_flags" not in payload:
        profiler_flags = []
        if sample_trace_id is not None:
            profiler_flags.append("sample_trace_present")
        else:
            profiler_flags.append("sample_trace_absent")
        if trace_tokens:
            profiler_flags.append("trace_chain_recorded")
        payload["profiler_flags"] = ",".join(profiler_flags)
    return payload


def _wilson_interval(success_count: float, trials: float, z: float = 1.96) -> tuple[float, float]:
    """Compute a Wilson score interval for a Bernoulli proportion.

    The interval is safe for small samples and extreme observed rates.
    """
    if not np.isfinite(success_count) or not np.isfinite(trials) or not np.isfinite(z):
        return 0.0, 0.0
    if trials <= 0 or success_count < 0:
        return 0.0, 0.0
    if z <= 0.0:
        z = 1.96

    success_count = float(success_count)
    total = float(trials)
    if success_count > total:
        success_count = total

    p_hat = success_count / total
    z2 = z * z
    denom = 1.0 + (z2 / total)
    center = (p_hat + z2 / (2.0 * total)) / denom
    half_width = (
        z
        * np.sqrt(max(p_hat * (1.0 - p_hat) / total + z2 / (4.0 * total * total), 0.0))
        / denom
    )
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return float(lower), float(upper)


def _append_ler_ci(metric: dict[str, Any], correct_flags: list[float]) -> dict[str, Any]:
    total = len(correct_flags)
    if total == 0:
        metric["ler_ci"] = {"lower": 0.0, "upper": 0.0}
        return metric

    flags = np.asarray(correct_flags, dtype=float)
    if not np.isfinite(flags).any():
        metric["ler_ci"] = {"lower": 0.0, "upper": 0.0}
        return metric

    flags = flags[np.isfinite(flags)]
    if flags.size == 0:
        metric["ler_ci"] = {"lower": 0.0, "upper": 0.0}
        return metric

    flags = np.clip(flags, 0.0, 1.0)
    total = float(flags.size)
    errors = float(np.sum(1.0 - flags))
    lower, upper = _wilson_interval(errors, total)
    metric["ler_ci"] = {"lower": float(lower), "upper": float(upper)}
    return metric


def _has_iqm_convergence(
    values: list[float],
    *,
    window: int = 10,
    relative_tolerance: float = 0.05,
) -> bool:
    """Detect convergence by checking IQM stability across adjacent windows."""
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0:
        return False
    if window <= 0:
        return False
    if len(values) < window * 2:
        return False

    values_array = np.asarray(values, dtype=float)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size < window * 2:
        return False

    previous = values_array[-2 * window : -window]
    recent = values_array[-window:]
    if len(previous) == 0 or len(recent) == 0:
        return False

    previous_iqm = _interquartile_mean(previous)
    recent_iqm = _interquartile_mean(recent)
    if not np.isfinite(previous_iqm) or not np.isfinite(recent_iqm):
        return False

    delta = abs(recent_iqm - previous_iqm)
    scale = max(abs(previous_iqm), abs(recent_iqm))
    q1_prev = np.quantile(previous, 0.25)
    q3_prev = np.quantile(previous, 0.75)
    q1_recent = np.quantile(recent, 0.25)
    q3_recent = np.quantile(recent, 0.75)
    window_iqr = float(max(q3_prev - q1_prev, q3_recent - q1_recent))

    if scale <= 1e-12:
        return delta <= relative_tolerance and window_iqr <= relative_tolerance

    required_tolerance = relative_tolerance * max(scale, 1e-12)
    if delta > required_tolerance:
        return False

    stability_tolerance = relative_tolerance * max(scale, 1.0)
    return window_iqr <= stability_tolerance


def _interquartile_mean(values: list[float]) -> float:
    """Return an interquartile mean for robust CI-like center estimates."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    if arr.size <= 2:
        return float(np.mean(arr))

    q1 = np.quantile(arr, 0.25)
    q3 = np.quantile(arr, 0.75)
    trimmed = arr[(arr >= q1) & (arr <= q3)]
    if trimmed.size == 0:
        return float(np.mean(arr))
    return float(np.mean(trimmed))


def _linear_schedule(start: float, stop: float, progress: float) -> float:
    progress = float(max(0.0, min(1.0, progress)))
    return start + (stop - start) * progress


def _resolve_curriculum_value(
    *,
    enabled: bool,
    start: float,
    stop: float,
    current_episode: int,
    ramp_episodes: int,
) -> float:
    if not enabled:
        return float(start)
    if ramp_episodes <= 0:
        return float(stop)
    if start == stop:
        return float(start)

    ratio = (current_episode + 1) / max(1, ramp_episodes)
    if ratio >= 1.0:
        return float(stop)
    return float(_linear_schedule(start, stop, ratio))


def _append_sampling_fields(metric: dict[str, Any], env_info: dict[str, Any]) -> dict[str, Any]:
    metric.update(_sampling_trace_payload(env_info))
    return metric


@dataclass(frozen=True)
class StrategyRuntime:
    """Runtime context passed into a training strategy."""

    episodes: int
    distance: int
    rounds: int
    p: float
    batch_size: int
    use_diffusion: bool
    syndrome_emit_every: int
    emit_metric: EmitMetric
    emit_syndrome: EmitSyndrome
    emit_done: EmitDone
    emit_error: EmitError
    should_stop: ShouldStop
    env_builder_registry: EnvBuilderRegistry
    use_accelerated_sampling: bool = False
    protocol: str = "surface"
    sampling_backend: str | None = None
    decoder_name: str | None = None
    enable_profile_traces: bool = False
    benchmark_probe_token: str | None = None
    protocol_metadata: dict[str, Any] = field(default_factory=dict)
    protocol_registry: ProtocolRegistry | None = None
    seed: int = 0
    curriculum_enabled: bool = False
    curriculum_distance_start: int | None = None
    curriculum_distance_end: int | None = None
    curriculum_p_start: float | None = None
    curriculum_p_end: float | None = None
    curriculum_ramp_episodes: int = 0
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    max_gradient_norm: float = 1.0
    pepg_population_size: int = 32
    pepg_learning_rate: float = 0.05
    pepg_sigma_learning_rate: float = 0.02

    use_mwpm_baseline: bool = True
    use_soft_information: bool = False
    parameter_dim: int = 4
    batch_shots: int = 64
    base_error_rate: float | None = None
    circuit_type: str = "tri"
    use_superdense: bool = False
    max_distance: int = 13
    min_distance: int = 3
    max_rounds: int = 10
    target_threshold: float = 0.005
    max_steps: int = 20


class TrainingStrategy(Protocol):
    """Strategy contract for RL training workflows."""

    name: str
    environment_name: str

    def run(self, runtime: StrategyRuntime) -> None:
        ...


class _BaseTrainingStrategy:
    """Shared behaviour for concrete RL strategies."""

    name: str
    environment_name: str

    def build_env(
        self,
        runtime: StrategyRuntime,
        *,
        distance: int | None = None,
        physical_error_rate: float | None = None,
        seed: int | None = None,
        base_error_rate: float | None = None,
    ):
        if distance is None:
            distance = runtime.distance
        if physical_error_rate is None:
            physical_error_rate = runtime.p
        if base_error_rate is None:
            base_error_rate = runtime.base_error_rate if runtime.base_error_rate is not None else physical_error_rate
        if seed is None:
            seed = runtime.seed

        context = EnvBuildContext(
            distance=int(distance),
            rounds=runtime.rounds,
            physical_error_rate=float(physical_error_rate),
            seed=seed,
            use_mwpm_baseline=runtime.use_mwpm_baseline,
            use_soft_information=runtime.use_soft_information,
            use_accelerated_sampling=runtime.use_accelerated_sampling,
            sampling_backend=runtime.sampling_backend,
            parameter_dim=runtime.parameter_dim,
            batch_shots=runtime.batch_shots,
            base_error_rate=base_error_rate,
            circuit_type=runtime.circuit_type,
            use_superdense=runtime.use_superdense,
            max_distance=runtime.max_distance,
            min_distance=runtime.min_distance,
            max_rounds=runtime.max_rounds,
            target_threshold=runtime.target_threshold,
            max_steps=runtime.max_steps,
            protocol=runtime.protocol,
            enable_profile_traces=runtime.enable_profile_traces,
            benchmark_probe_token=runtime.benchmark_probe_token,
            protocol_metadata=runtime.protocol_metadata,
            decoder_name=runtime.decoder_name,
        )
        protocol_registry = runtime.protocol_registry or DEFAULT_PROTOCOL_REGISTRY
        protocol_adapter = protocol_registry.get(context.protocol)
        context = protocol_adapter.normalize_context(context)
        protocol_adapter.validate_context(context)
        builder = runtime.env_builder_registry.get(self.environment_name)
        return builder.build(context)

    def _resolve_curriculum(self, runtime: StrategyRuntime, episode: int) -> tuple[int, float]:
        start_distance = (
            runtime.curriculum_distance_start
            if runtime.curriculum_distance_start is not None
            else runtime.distance
        )
        stop_distance = (
            runtime.curriculum_distance_end
            if runtime.curriculum_distance_end is not None
            else runtime.distance
        )
        start_p = (
            runtime.curriculum_p_start
            if runtime.curriculum_p_start is not None
            else runtime.p
        )
        stop_p = (
            runtime.curriculum_p_end
            if runtime.curriculum_p_end is not None
            else runtime.p
        )
        distance = int(
            round(
                _resolve_curriculum_value(
                    enabled=runtime.curriculum_enabled,
                    start=float(start_distance),
                    stop=float(stop_distance),
                    current_episode=episode,
                    ramp_episodes=runtime.curriculum_ramp_episodes,
                )
            )
        )
        physical_error_rate = _resolve_curriculum_value(
            enabled=runtime.curriculum_enabled,
            start=float(start_p),
            stop=float(stop_p),
            current_episode=episode,
            ramp_episodes=runtime.curriculum_ramp_episodes,
        )
        return max(1, distance), max(1e-12, min(0.5, physical_error_rate))

    def _should_stop_early(
        self,
        *,
        runtime: StrategyRuntime,
        window_success: deque[float],
        best_success: float,
        no_improve_count: int,
    ) -> tuple[float, int, bool]:
        if runtime.early_stopping_patience <= 0:
            return best_success, no_improve_count, False
        if len(window_success) < runtime.batch_size:
            return best_success, no_improve_count, False
        current = float(np.mean(window_success))
        if current > best_success + runtime.early_stopping_min_delta:
            return current, 0, False
        new_no_improve_count = no_improve_count + 1
        if new_no_improve_count >= runtime.early_stopping_patience:
            return current, new_no_improve_count, True
        return best_success, new_no_improve_count, False


class PPOTrainingStrategy(_BaseTrainingStrategy):
    name = "ppo"
    environment_name = "qec"

    def run(self, runtime: StrategyRuntime) -> None:
        try:
            import torch
            from surface_code_in_stem.rl_control.sota_agents import PPOAgent

            torch.manual_seed(runtime.seed)
            np.random.seed(runtime.seed)

            env_distance, env_p = self._resolve_curriculum(runtime, 0)
            env = self.build_env(runtime, distance=env_distance, physical_error_rate=env_p, seed=runtime.seed)
            state_dim = env.observation_space.shape[0]
            action_dim = len(env.action_space.nvec)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                device=device,
                max_grad_norm=runtime.max_gradient_norm,
            )

            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            log_probs: list[float] = []
            rewards: list[float] = []
            values: list[float] = []
            success_hist: list[float] = []
            mwpm_hist: list[float] = []
            episode_success_window: deque[float] = deque(maxlen=max(1, runtime.batch_size))
            best_success = -1.0
            no_improve_streak = 0
            completed_episodes = 0
            current_distance = env_distance
            current_p = env_p

            for ep in range(runtime.episodes):
                if runtime.should_stop():
                    break
                completed_episodes = ep + 1

                target_distance, target_p = self._resolve_curriculum(runtime, ep)
                if target_distance != current_distance or target_p != current_p:
                    current_distance = target_distance
                    current_p = target_p
                    env = self.build_env(
                        runtime,
                        distance=current_distance,
                        physical_error_rate=current_p,
                        seed=int(runtime.seed + completed_episodes),
                    )

                state, info = env.reset(seed=int(runtime.seed + completed_episodes))
                action, log_prob, value = agent.select_action(state)
                _, reward, _, _, env_info = env.step(action)

                correct = bool(env_info.get("is_correct", False))
                sampling_env_info = info if isinstance(info, dict) else {}
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(float(reward))
                values.append(float(value))
                success_hist.append(1.0 if correct else 0.0)
                mwpm_hist.append(1.0 if info.get("mwpm_correct", False) else 0.0)
                episode_success_window.append(1.0 if correct else 0.0)

                if ep % runtime.syndrome_emit_every == 0:
                    syndrome_arr = np.asarray(info.get("binary_syndrome", state), dtype=np.int8)
                    runtime.emit_syndrome(syndrome_arr, np.asarray(action), correct, ep + 1)

                if (ep + 1) % runtime.batch_size == 0 or (ep + 1) == runtime.episodes:
                    ret_t = torch.FloatTensor(rewards)
                    val_t = torch.FloatTensor(values)
                    adv_t = ret_t - val_t
                    s_t = torch.FloatTensor(np.array(states))
                    a_t = torch.FloatTensor(np.array(actions))
                    lp_t = torch.FloatTensor(log_probs)
                    loss_d = agent.update(s_t, a_t, lp_t, ret_t, adv_t)
                    states, actions, log_probs, rewards, values = [], [], [], [], []

                    window = min(runtime.batch_size, len(success_hist))
                    batch_success = list(success_hist[-window:])
                    batch_rewards = np.asarray(ret_t.detach().cpu().numpy(), dtype=float)
                    avg_reward = float(np.mean(batch_rewards)) if batch_rewards.size > 0 else 0.0
                    metric = {
                        "episode": ep + 1,
                        "reward": avg_reward,
                        "rl_success": float(np.mean(batch_success)),
                        "rl_success_iqm": _interquartile_mean(batch_success),
                        "rl_success_iqm_converged": _has_iqm_convergence(success_hist),
                        "mwpm_success": float(np.mean(mwpm_hist[-window:])),
                        "success_window": window,
                        "policy_loss": float(loss_d.get("policy_loss", 0)),
                        "value_loss": float(loss_d.get("value_loss", 0)),
                        "logical_error_rate": 0.0,
                        "effective_p": 0.0,
                        "curriculum_distance": int(current_distance),
                        "curriculum_p": float(current_p),
                        "curriculum_episode": ep + 1,
                        "seed": int(runtime.seed),
                        "learning_steps": 1 + (ep // runtime.batch_size),
                    }
                    _append_ler_ci(metric, success_hist)
                    _append_sampling_fields(metric, sampling_env_info)
                    runtime.emit_metric(metric)

                    best_success, no_improve_streak, stop_early = self._should_stop_early(
                        runtime=runtime,
                        window_success=episode_success_window,
                        best_success=best_success,
                        no_improve_count=no_improve_streak,
                    )
                    if stop_early:
                        break

            runtime.emit_done(completed_episodes)

        except Exception as exc:  # pragma: no cover - runtime errors surfaced via queue
            runtime.emit_error(str(exc))


class SACTrainingStrategy(_BaseTrainingStrategy):
    name = "sac"
    environment_name = "qec_continuous"

    def run(self, runtime: StrategyRuntime) -> None:
        try:
            import torch
            from surface_code_in_stem.rl_control.replay_buffer import Experience, ReplayBuffer
            from surface_code_in_stem.rl_control.sota_agents import ContinuousSACAgent

            torch.manual_seed(runtime.seed)
            np.random.seed(runtime.seed)

            env_distance, env_p = self._resolve_curriculum(runtime, 0)
            env = self.build_env(runtime, distance=env_distance, physical_error_rate=env_p, seed=runtime.seed)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            agent = ContinuousSACAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                action_space=env.action_space,
                use_diffusion=runtime.use_diffusion,
                device=device,
                max_grad_norm=runtime.max_gradient_norm,
            )
            buffer = ReplayBuffer(capacity=5000, prioritized=False)
            completed_episodes = 0
            episode_success_window: deque[float] = deque(maxlen=max(1, runtime.batch_size))
            best_success = -1.0
            no_improve_streak = 0
            current_distance = env_distance
            current_p = env_p
            policy_updates = 0
            last_policy_loss = 0.0
            last_q_loss = 0.0
            last_alpha_loss = 0.0
            last_alpha = float(agent.alpha)

            for ep in range(runtime.episodes):
                if runtime.should_stop():
                    break
                completed_episodes = ep + 1

                target_distance, target_p = self._resolve_curriculum(runtime, ep)
                if target_distance != current_distance or target_p != current_p:
                    current_distance = target_distance
                    current_p = target_p
                    env = self.build_env(
                        runtime,
                        distance=current_distance,
                        physical_error_rate=current_p,
                        seed=int(runtime.seed + completed_episodes),
                    )

                state, reset_info = env.reset(seed=int(runtime.seed + completed_episodes))
                ep_reward = 0.0
                ep_info: dict = {}
                sampling_env_info: dict = reset_info if isinstance(reset_info, dict) else {}
                done = False
                terminated_by_env = False

                while not done:
                    if runtime.should_stop():
                        break
                    if len(buffer) > runtime.batch_size:
                        action = agent.select_action(state, evaluate=False)
                    else:
                        action = env.action_space.sample()

                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    ep_reward += float(reward)
                    ep_info = info

                    buffer.push(Experience(
                        state=torch.FloatTensor(state),
                        action=torch.FloatTensor(action),
                        reward=float(reward),
                        next_state=torch.FloatTensor(next_state),
                        done=bool(done),
                    ))
                    state = next_state

                    if len(buffer) > runtime.batch_size:
                        s_b, a_b, r_b, ns_b, d_b, _, _ = buffer.sample(runtime.batch_size)
                        mask_b = 1.0 - d_b.float()
                        losses = agent.update_parameters(s_b, a_b, r_b, ns_b, mask_b)
                        last_policy_loss = float(losses.get("policy_loss", 0.0))
                        last_q_loss = float(losses.get("q_loss", 0.0))
                        last_alpha_loss = float(losses.get("alpha_loss", 0.0))
                        last_alpha = float(losses.get("alpha", agent.alpha))
                        policy_updates += 1
                    terminated_by_env = terminated_by_env or terminated
                    if terminated:
                        break

                if runtime.should_stop():
                    break

                ler = float(ep_info.get("logical_error_rate", 0.0))
                eff_p = float(ep_info.get("effective_p", 0.0))
                base_p = eff_p if eff_p > 0 else runtime.p
                if base_p <= 0:
                    base_p = 1e-12
                episode_success = 1.0 - min(1.0, ler / base_p)
                episode_success = float(max(0.0, episode_success))
                episode_success_window.append(episode_success)
                metric = {
                    "episode": ep + 1,
                    "reward": ep_reward,
                    "rl_success": float(episode_success),
                    "rl_success_iqm": _interquartile_mean(list(episode_success_window)),
                    "rl_success_iqm_converged": _has_iqm_convergence(list(episode_success_window)),
                    "mwpm_success": 0.0,
                    "logical_error_rate": ler,
                    "effective_p": eff_p,
                    "policy_loss": float(last_policy_loss),
                    "value_loss": float(last_q_loss),
                    "alpha_loss": float(last_alpha_loss),
                    "alpha": float(last_alpha),
                    "policy_updates": int(policy_updates),
                    "curriculum_distance": int(current_distance),
                    "curriculum_p": float(current_p),
                    "curriculum_episode": ep + 1,
                    "seed": int(runtime.seed),
                    "terminated_by_env": bool(terminated_by_env),
                    "done": bool(done),
                    "learning_steps": 1 + (ep // max(1, runtime.batch_size)),
                }
                _append_ler_ci(metric, list(episode_success_window))
                _append_sampling_fields(metric, sampling_env_info)
                runtime.emit_metric(metric)

                best_success, no_improve_streak, stop_early = self._should_stop_early(
                    runtime=runtime,
                    window_success=episode_success_window,
                    best_success=best_success,
                    no_improve_count=no_improve_streak,
                )

                if ep % runtime.syndrome_emit_every == 0:
                    dummy = np.zeros(max(1, state_dim), dtype=np.int8)
                    runtime.emit_syndrome(dummy, np.array(action), ler < runtime.p, ep + 1)
                if stop_early:
                    break

            runtime.emit_done(completed_episodes)

        except Exception as exc:  # pragma: no cover
            runtime.emit_error(str(exc))


class PEPGTrainingStrategy(_BaseTrainingStrategy):
    name = "pepg"
    environment_name = "qec_continuous"

    def run(self, runtime: StrategyRuntime) -> None:
        try:
            from surface_code_in_stem.rl_control.optimizer import PEPGOptimizer

            np.random.seed(runtime.seed)
            env_distance, env_p = self._resolve_curriculum(runtime, 0)
            env = self.build_env(
                runtime,
                distance=env_distance,
                physical_error_rate=env_p,
                seed=runtime.seed,
            )
            action_dim = env.action_space.shape[0]
            population_size = int(runtime.pepg_population_size)
            if population_size < 2:
                population_size = 2
            if population_size % 2 == 1:
                population_size += 1

            optimizer = PEPGOptimizer(
                parameter_dim=int(action_dim),
                seed=runtime.seed,
                learning_rate=runtime.pepg_learning_rate,
                sigma_learning_rate=runtime.pepg_sigma_learning_rate,
            )
            completed_episodes = 0
            episode_success_window: deque[float] = deque(maxlen=max(1, runtime.batch_size))
            best_success = -1.0
            no_improve_streak = 0
            current_distance = env_distance
            current_p = env_p
            learning_steps = 0

            for ep in range(runtime.episodes):
                if runtime.should_stop():
                    break
                completed_episodes = ep + 1

                target_distance, target_p = self._resolve_curriculum(runtime, ep)
                if target_distance != current_distance or target_p != current_p:
                    current_distance = target_distance
                    current_p = target_p
                    env = self.build_env(
                        runtime,
                        distance=current_distance,
                        physical_error_rate=current_p,
                        seed=int(runtime.seed + completed_episodes),
                    )

                candidates, perturbations = optimizer.ask(population_size)
                rewards = np.zeros(population_size, dtype=float)
                candidate_lers: list[float] = []
                candidate_infos: list[dict] = []
                for idx, candidate in enumerate(candidates):
                    obs_reset = env.reset(seed=int(runtime.seed + completed_episodes + idx))
                    state = np.asarray(obs_reset[0], dtype=float)
                    reset_info = obs_reset[1] if isinstance(obs_reset, tuple) and len(obs_reset) > 1 else {}
                    _, reward, _, _, step_info = env.step(np.asarray(candidate, dtype=float))
                    rewards[idx] = float(reward)
                    if not isinstance(step_info, Mapping):
                        step_info = {}
                    lER = float(step_info.get("logical_error_rate", 0.0))
                    candidate_lers.append(max(0.0, lER))
                    candidate_info: dict[str, Any] = {}
                    if isinstance(reset_info, Mapping):
                        candidate_info.update(reset_info)
                    candidate_info.update(step_info)
                    candidate_infos.append(candidate_info)

                # Keep behavior compatible with population-based optimizers.
                optimizer.tell(perturbations, rewards)
                learning_steps += 1

                best_idx = int(np.argmax(rewards))
                best_reward = float(rewards[best_idx])
                best_ler = float(candidate_lers[best_idx]) if candidate_lers else 0.0
                best_info = candidate_infos[best_idx] if candidate_infos else {}
                base_p = float(best_info.get("effective_p", current_p))
                if base_p <= 0.0:
                    base_p = runtime.p if runtime.p > 0 else 1e-12
                episode_success = 1.0 - min(1.0, best_ler / base_p)
                episode_success = float(max(0.0, episode_success))
                episode_success_window.append(episode_success)

                metric = {
                    "episode": ep + 1,
                    "reward": float(best_reward),
                    "rl_success": float(episode_success),
                    "rl_success_iqm": _interquartile_mean(list(episode_success_window)),
                    "rl_success_iqm_converged": _has_iqm_convergence(list(episode_success_window)),
                    "mwpm_success": 0.0,
                    "logical_error_rate": best_ler,
                    "effective_p": base_p,
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "alpha_loss": 0.0,
                    "alpha": 0.0,
                    "policy_updates": int(learning_steps),
                    "sigma_mean": float(np.mean(optimizer.sigma)),
                    "curriculum_distance": int(current_distance),
                    "curriculum_p": float(current_p),
                    "curriculum_episode": ep + 1,
                    "seed": int(runtime.seed),
                    "learning_steps": learning_steps,
                }
                _append_ler_ci(metric, list(episode_success_window))
                _append_sampling_fields(metric, best_info if isinstance(best_info, dict) else {})
                runtime.emit_metric(metric)

                best_success, no_improve_streak, stop_early = self._should_stop_early(
                    runtime=runtime,
                    window_success=episode_success_window,
                    best_success=best_success,
                    no_improve_count=no_improve_streak,
                )
                if stop_early:
                    break

                if ep % runtime.syndrome_emit_every == 0:
                    runtime.emit_syndrome(state.astype(np.int8), np.zeros_like(state), best_ler < base_p, ep + 1)

            runtime.emit_done(completed_episodes)
        except Exception as exc:  # pragma: no cover
            runtime.emit_error(str(exc))


class PPOColourTrainingStrategy(_BaseTrainingStrategy):
    name = "ppo_colour"
    environment_name = "colour_gym"

    def run(self, runtime: StrategyRuntime) -> None:
        try:
            import torch
            from surface_code_in_stem.rl_control.sota_agents import PPOAgent

            env = self.build_env(runtime)
            state_dim = env.observation_space.shape[0]
            action_dim = len(env.action_space.nvec)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)

            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            log_probs: list[float] = []
            rewards: list[float] = []
            values: list[float] = []
            success_hist: list[float] = []
            mwpm_hist: list[float] = []

            for ep in range(runtime.episodes):
                if runtime.should_stop():
                    break

                state, info = env.reset()
                action, log_prob, value = agent.select_action(state)
                _, reward, _, _, env_info = env.step(action)

                correct = bool(env_info.get("is_correct", False))
                sampling_env_info = info if isinstance(info, dict) else {}
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(float(reward))
                values.append(float(value))
                success_hist.append(1.0 if correct else 0.0)
                mwpm_hist.append(1.0 if info.get("mwpm_correct", False) else 0.0)

                if ep % runtime.syndrome_emit_every == 0:
                    syndrome_arr = np.asarray(info.get("binary_syndrome", state), dtype=np.int8)
                    runtime.emit_syndrome(syndrome_arr, np.asarray(action), correct, ep + 1)

                if (ep + 1) % runtime.batch_size == 0:
                    ret_t = torch.FloatTensor(rewards)
                    val_t = torch.FloatTensor(values)
                    adv_t = ret_t - val_t
                    s_t = torch.FloatTensor(np.array(states))
                    a_t = torch.FloatTensor(np.array(actions))
                    lp_t = torch.FloatTensor(log_probs)
                    loss_d = agent.update(s_t, a_t, lp_t, ret_t, adv_t)
                    states, actions, log_probs, rewards, values = [], [], [], [], []

                    window = min(runtime.batch_size, len(success_hist))
                    metric = {
                        "episode": ep + 1,
                        "reward": float(reward),
                        "rl_success": float(np.mean(success_hist[-window:])),
                        "mwpm_success": float(np.mean(mwpm_hist[-window:])),
                        "policy_loss": float(loss_d.get("policy_loss", 0)),
                        "value_loss": float(loss_d.get("value_loss", 0)),
                        "logical_error_rate": 0.0,
                        "effective_p": 0.0,
                    }
                    _append_ler_ci(metric, success_hist)
                    _append_sampling_fields(metric, sampling_env_info)
                    runtime.emit_metric(metric)

            runtime.emit_done(runtime.episodes)

        except Exception as exc:  # pragma: no cover
            runtime.emit_error(str(exc))


class SACColourTrainingStrategy(_BaseTrainingStrategy):
    name = "sac_colour"
    environment_name = "colour_calibration"

    def run(self, runtime: StrategyRuntime) -> None:
        try:
            import torch
            from surface_code_in_stem.rl_control.replay_buffer import Experience, ReplayBuffer
            from surface_code_in_stem.rl_control.sota_agents import ContinuousSACAgent

            env = self.build_env(runtime)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            agent = ContinuousSACAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                action_space=env.action_space,
                use_diffusion=runtime.use_diffusion,
                device=device,
            )
            buffer = ReplayBuffer(capacity=5000, prioritized=False)

            for ep in range(runtime.episodes):
                if runtime.should_stop():
                    break

                state, reset_info = env.reset()
                ep_reward = 0.0
                ep_info: dict = {}
                sampling_env_info = reset_info if isinstance(reset_info, dict) else {}
                done = False

                while not done:
                    if runtime.should_stop():
                        break
                    if len(buffer) > runtime.batch_size:
                        action = agent.select_action(state, evaluate=False)
                    else:
                        action = env.action_space.sample()

                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    ep_reward += float(reward)
                    ep_info = info

                    buffer.push(Experience(
                        state=torch.FloatTensor(state),
                        action=torch.FloatTensor(action),
                        reward=float(reward),
                        next_state=torch.FloatTensor(next_state),
                        done=bool(done),
                    ))
                    state = next_state

                    if len(buffer) > runtime.batch_size:
                        s_b, a_b, r_b, ns_b, d_b, _, _ = buffer.sample(runtime.batch_size)
                        mask_b = 1.0 - d_b.float()
                        agent.update_parameters(s_b, a_b, r_b, ns_b, mask_b)

                ler = float(ep_info.get("logical_error_rate", 0.0))
                metric = {
                    "episode": ep + 1,
                    "reward": ep_reward,
                    "rl_success": 0.0,
                    "mwpm_success": 0.0,
                    "logical_error_rate": ler,
                    "effective_p": float(ep_info.get("mean_defect_rate", 0.0)),
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                }
                _append_ler_ci(metric, [1.0 if ler <= 0.0 else 0.0])
                _append_sampling_fields(metric, sampling_env_info)
                runtime.emit_metric(metric)

                if ep % runtime.syndrome_emit_every == 0:
                    dummy = np.zeros(max(1, state_dim), dtype=np.int8)
                    runtime.emit_syndrome(dummy, np.array(action), ler < runtime.p, ep + 1)

            runtime.emit_done(runtime.episodes)

        except Exception as exc:  # pragma: no cover
            runtime.emit_error(str(exc))


class DiscoverColourTrainingStrategy(_BaseTrainingStrategy):
    name = "discover_colour"
    environment_name = "colour_discovery"

    def run(self, runtime: StrategyRuntime) -> None:
        try:
            env = self.build_env(runtime)
            info: dict = {}
            for ep in range(runtime.episodes):
                if runtime.should_stop():
                    break

                state, info = env.reset()
                state = np.asarray(state)
                done = False
                total_reward = 0.0
                while not done:
                    if runtime.should_stop():
                        break
                    action = np.random.randint(0, env.action_space.n)
                    state, reward, terminated, truncated, step_info = env.step(action)
                    done = terminated or truncated
                    total_reward += float(reward)
                    info = step_info

                metric = {
                    "episode": ep + 1,
                    "reward": total_reward,
                    "rl_success": 1.0 if total_reward > 0 else 0.0,
                    "mwpm_success": 0.0,
                    "logical_error_rate": 0.0,
                    "effective_p": float(info.get("p", runtime.p)),
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                }
                runtime.emit_metric(metric)

            runtime.emit_done(runtime.episodes)

        except Exception as exc:  # pragma: no cover
            runtime.emit_error(str(exc))


class TrainingStrategyRegistry:
    """Registry for available training strategies."""

    def __init__(self) -> None:
        self._strategies: dict[str, TrainingStrategy] = {}

    def register(self, strategy: TrainingStrategy) -> None:
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> TrainingStrategy:
        if name not in self._strategies:
            raise KeyError(f"Unknown training strategy '{name}'.")
        return self._strategies[name]

    def list(self) -> list[str]:
        return sorted(self._strategies.keys())


def default_training_strategy_registry() -> TrainingStrategyRegistry:
    """Create the default training strategy registry."""
    registry = TrainingStrategyRegistry()
    registry.register(PPOTrainingStrategy())
    registry.register(SACTrainingStrategy())
    registry.register(PEPGTrainingStrategy())
    registry.register(PPOColourTrainingStrategy())
    registry.register(SACColourTrainingStrategy())
    registry.register(DiscoverColourTrainingStrategy())
    return registry

