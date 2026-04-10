"""Threaded RL training runner that streams metrics into a shared queue.

Used by the Streamlit app to run training asynchronously while the UI polls
and updates live charts without blocking the event loop.
"""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from app.rl_strategies import (
    StrategyRuntime,
    default_training_strategy_registry,
)
from surface_code_in_stem.rl_control.envs import default_builder_registry

# Allow importing from the repo root
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class TrainingEvent:
    """Base event pushed from the training thread."""

    __slots__ = ("kind",)

    def __init__(self, kind: str):
        self.kind = kind


class MetricEvent(TrainingEvent):
    """Per-episode metric snapshot."""

    __slots__ = ("kind", "data")

    def __init__(self, data: dict[str, Any]):
        super().__init__("metric")
        self.data = data


class SyndromeEvent(TrainingEvent):
    """Current syndrome + action taken by the agent this step."""

    __slots__ = ("kind", "syndrome", "action", "correct", "episode")

    def __init__(self, syndrome: np.ndarray, action: np.ndarray, correct: bool, episode: int):
        super().__init__("syndrome")
        self.syndrome = syndrome
        self.action = action
        self.correct = correct
        self.episode = episode


class DoneEvent(TrainingEvent):
    """Training finished normally."""

    __slots__ = ("kind", "total_episodes")

    def __init__(self, total_episodes: int):
        super().__init__("done")
        self.total_episodes = total_episodes


class ErrorEvent(TrainingEvent):
    """Training crashed — carries the exception message."""

    __slots__ = ("kind", "message")

    def __init__(self, message: str):
        super().__init__("error")
        self.message = message


class RLRunner:
    """Runs RL training strategies in a background thread.

    Results are streamed to ``event_queue`` so the Streamlit UI can poll
    without blocking.

    Usage::

        runner = RLRunner(mode="ppo", distance=3, rounds=2, p=0.01, episodes=200)
        runner.start()

        while runner.is_running():
            try:
                event = runner.poll(timeout=0.05)
                if isinstance(event, MetricEvent):
                    ...
            except queue.Empty:
                pass
        runner.join()
    """

    def __init__(
        self,
        mode: str = "ppo",
        distance: int = 3,
        rounds: int = 2,
        physical_error_rate: float = 0.01,
        episodes: int = 200,
        batch_size: int = 32,
        pepg_population_size: int = 32,
        pepg_learning_rate: float = 0.05,
        pepg_sigma_learning_rate: float = 0.02,
        use_diffusion: bool = False,
        use_accelerated: bool | None = None,
        seed: int = 0,
        syndrome_emit_every: int = 5,
        protocol: str = "surface",
        sampling_backend: str | None = None,
        enable_profile_traces: bool = False,
        benchmark_probe_token: str | None = None,
        protocol_metadata: dict[str, Any] | None = None,
        decoder_name: str | None = None,
        curriculum_enabled: bool = False,
        curriculum_distance_start: int | None = None,
        curriculum_distance_end: int | None = None,
        curriculum_p_start: float | None = None,
        curriculum_p_end: float | None = None,
        curriculum_ramp_episodes: int = 0,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.0,
        max_gradient_norm: float = 1.0,
        on_metric: Optional[Callable[[dict], None]] = None,
        *,
        strategy_registry = None,
        env_builder_registry = None,
        protocol_registry = None,
    ):
        self.mode = mode
        self.distance = distance
        self.rounds = rounds
        self.p = physical_error_rate
        self.episodes = episodes
        self.batch_size = batch_size
        self.use_diffusion = use_diffusion
        self.seed = seed
        self.sampling_backend = sampling_backend
        if use_accelerated is None:
            if self.sampling_backend is not None:
                use_accelerated = str(self.sampling_backend).lower() != "stim"
            else:
                from surface_code_in_stem.accelerators import qhybrid_backend
                use_accelerated = bool(qhybrid_backend.probe_capability().get("enabled", False))
        self.use_accelerated = bool(use_accelerated)
        self.syndrome_emit_every = syndrome_emit_every
        self.enable_profile_traces = enable_profile_traces
        self.benchmark_probe_token = benchmark_probe_token
        self.protocol_metadata = dict(protocol_metadata or {})
        self.decoder_name = decoder_name
        self.curriculum_enabled = curriculum_enabled
        self.curriculum_distance_start = curriculum_distance_start
        self.curriculum_distance_end = curriculum_distance_end
        self.curriculum_p_start = curriculum_p_start
        self.curriculum_p_end = curriculum_p_end
        self.curriculum_ramp_episodes = curriculum_ramp_episodes
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.max_gradient_norm = max_gradient_norm
        self.pepg_population_size = pepg_population_size
        self.pepg_learning_rate = pepg_learning_rate
        self.pepg_sigma_learning_rate = pepg_sigma_learning_rate
        self._on_metric = on_metric

        self._strategy_registry = (
            strategy_registry
            if strategy_registry is not None
            else default_training_strategy_registry()
        )
        self._env_builder_registry = (
            env_builder_registry
            if env_builder_registry is not None
            else default_builder_registry()
        )
        self._protocol_registry = protocol_registry
        self.protocol = protocol

        self.event_queue: queue.Queue[TrainingEvent] = queue.Queue(maxsize=2000)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._dropped_events = 0

    def start(self) -> None:
        self._stop_event.clear()
        strategy = self._strategy_registry.get(self.mode)

        runtime = StrategyRuntime(
            episodes=self.episodes,
            distance=self.distance,
            rounds=self.rounds,
            p=self.p,
            batch_size=self.batch_size,
            use_diffusion=self.use_diffusion,
            syndrome_emit_every=self.syndrome_emit_every,
            emit_metric=self._emit_metric,
            emit_syndrome=self._emit_syndrome,
            emit_done=self._emit_done,
            emit_error=self._emit_error,
            should_stop=self._should_stop,
            env_builder_registry=self._env_builder_registry,
            use_accelerated_sampling=self.use_accelerated,
            protocol=self.protocol,
            sampling_backend=self.sampling_backend,
            decoder_name=self.decoder_name,
            enable_profile_traces=self.enable_profile_traces,
            benchmark_probe_token=self.benchmark_probe_token,
            protocol_metadata=self.protocol_metadata,
            protocol_registry=self._protocol_registry,
            seed=self.seed,
            curriculum_enabled=self.curriculum_enabled,
            curriculum_distance_start=self.curriculum_distance_start,
            curriculum_distance_end=self.curriculum_distance_end,
            curriculum_p_start=self.curriculum_p_start,
            curriculum_p_end=self.curriculum_p_end,
            curriculum_ramp_episodes=self.curriculum_ramp_episodes,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=self.early_stopping_min_delta,
            max_gradient_norm=self.max_gradient_norm,
            pepg_population_size=self.pepg_population_size,
            pepg_learning_rate=self.pepg_learning_rate,
            pepg_sigma_learning_rate=self.pepg_sigma_learning_rate,
        )

        self._thread = threading.Thread(target=lambda: strategy.run(runtime), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: float = 5.0) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def poll(self, timeout: float = 0.02) -> TrainingEvent:
        return self.event_queue.get(timeout=timeout)

    def drain(self, max_events: int | None = None) -> list[TrainingEvent]:
        """Drain queued events. Optionally cap to latest `max_events`."""
        events = []
        remaining = max_events if max_events is not None else None
        while True:
            if remaining is not None and remaining <= 0:
                break
            try:
                events.append(self.event_queue.get_nowait())
                if remaining is not None:
                    remaining -= 1
            except queue.Empty:
                break
        return events

    def drain_latest(self, max_events: int = 1, coalesce: bool = False) -> list[TrainingEvent]:
        """Drain newest events quickly.

        If `coalesce` is True, this returns only the most recent metric and syndrome
        event of each kind, plus at most one done/error event.
        """
        if max_events is None or max_events <= 0:
            events = self.drain()
        elif coalesce:
            events = self.drain()
        else:
            events = self.drain(max_events)
        if coalesce and max_events and max_events > 0:
            events = events[-max_events:]
        if not coalesce:
            return events

        latest: dict[str, TrainingEvent] = {}
        ordered_kinds = []
        for event in events:
            kind = event.kind
            if kind not in latest:
                ordered_kinds.append(kind)
            latest[kind] = event

        result: list[TrainingEvent] = []
        for kind in ordered_kinds:
            result.append(latest[kind])
        return result

    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    def _emit_metric(self, metric: dict[str, Any]) -> None:
        self._push(MetricEvent(metric))
        if self._on_metric:
            self._on_metric(metric)

    def _emit_syndrome(
        self,
        syndrome: np.ndarray,
        action: np.ndarray,
        correct: bool,
        episode: int,
    ) -> None:
        self._push(SyndromeEvent(syndrome, action, correct, episode))

    def _emit_done(self, total_episodes: int) -> None:
        self._push(DoneEvent(total_episodes))

    def _emit_error(self, message: str) -> None:
        self._push(ErrorEvent(message))

    def _push(self, event: TrainingEvent) -> None:
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            self._dropped_events += 1
            pass

    def dropped_events(self) -> int:
        """Return the number of events dropped due to queue overflow."""
        return self._dropped_events
