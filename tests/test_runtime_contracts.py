"""Regression tests for runtime orchestration contracts."""

from __future__ import annotations

import numpy as np

from app.rl_runner import RLRunner
from app.rl_strategies import _BaseTrainingStrategy, TrainingStrategyRegistry
from surface_code_in_stem.protocols import DEFAULT_PROTOCOL_REGISTRY
from surface_code_in_stem.rl_control.envs import EnvBuildContext, EnvBuilderRegistry


class BurstStrategy(_BaseTrainingStrategy):
    name = "bench_burst"
    environment_name = "bench"

    def run(self, runtime) -> None:
        for idx in range(1, 51):
            runtime.emit_metric({"episode": idx, "reward": float(idx % 5)})
            runtime.emit_syndrome(np.array([idx % 2], dtype=np.int8), np.array([idx % 3], dtype=np.int8), True, idx)
        runtime.emit_done(runtime.episodes)


class CaptureContextStrategy(_BaseTrainingStrategy):
    name = "bench_capture"
    environment_name = "capture"

    def run(self, runtime) -> None:
        self._captured = self.build_env(runtime)
        runtime.emit_done(runtime.episodes)


class BenchBuilder:
    name = "bench"

    def build(self, context: EnvBuildContext):
        return context


class CaptureBuilder:
    name = "capture"

    def __init__(self) -> None:
        self.context = None

    def build(self, context: EnvBuildContext):
        self.context = context
        return context


def _coalesce_events(events: list[object]) -> list[object]:
    latest: dict[str, object] = {}
    ordered_kinds: list[str] = []
    for event in events:
        kind = getattr(event, "kind", "")
        if kind not in latest:
            ordered_kinds.append(kind)
        latest[kind] = event
    return [latest[kind] for kind in ordered_kinds]


def test_runner_drains_and_coalesces_contract_events():
    strategy_registry = TrainingStrategyRegistry()
    strategy_registry.register(BurstStrategy())

    env_builder_registry = EnvBuilderRegistry()
    env_builder_registry.register(BenchBuilder())

    runner = RLRunner(
        mode="bench_burst",
        episodes=1,
        distance=3,
        rounds=2,
        physical_error_rate=0.01,
        batch_size=32,
        strategy_registry=strategy_registry,
        env_builder_registry=env_builder_registry,
        protocol_registry=DEFAULT_PROTOCOL_REGISTRY,
        protocol="surface",
    )
    runner.start()
    runner.join(timeout=5.0)
    events = runner.drain()
    kinds = [e.kind for e in events]

    assert "metric" in kinds
    assert "syndrome" in kinds
    assert kinds[-1] == "done"
    coalesced = _coalesce_events(events)
    coalesced_kinds = [e.kind for e in coalesced]
    assert coalesced_kinds.count("metric") == 1
    assert coalesced_kinds.count("syndrome") == 1


def test_surface_protocol_normalization_happens_in_runner_context():
    strategy_registry = TrainingStrategyRegistry()
    strategy_registry.register(CaptureContextStrategy())

    capture_builder = CaptureBuilder()
    env_builder_registry = EnvBuilderRegistry()
    env_builder_registry.register(capture_builder)

    runner = RLRunner(
        mode="bench_capture",
        episodes=1,
        distance=4,
        rounds=2,
        physical_error_rate=0.01,
        batch_size=8,
        strategy_registry=strategy_registry,
        env_builder_registry=env_builder_registry,
        protocol_registry=DEFAULT_PROTOCOL_REGISTRY,
        protocol="surface",
    )
    runner.start()
    runner.join(timeout=5.0)
    _events = runner.drain()

    assert capture_builder.context is not None
    assert capture_builder.context.protocol == "surface"
    assert capture_builder.context.distance % 2 == 1

