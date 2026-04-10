"""Boundary tests for registry and strategy dispatch seams."""

import pytest

from app.rl_runner import DoneEvent, RLRunner
from app.rl_strategies import (
    _BaseTrainingStrategy,
    TrainingStrategyRegistry,
    default_training_strategy_registry,
)
from surface_code_in_stem.rl_control.envs import (
    EnvBuildContext,
    EnvBuilderRegistry,
    default_builder_registry,
)
from surface_code_in_stem.protocols import DEFAULT_PROTOCOL_REGISTRY
import surface_code_in_stem.protocols as protocols_module
import surface_code_in_stem.rl_control.envs as envs_module
from syndrome_net.container import DIContainer
from surface_code_in_stem.protocols.base import ProtocolContract


class FakeEntryPoint:
    """Simple, test-only object matching the EntryPoint.load() contract."""

    def __init__(self, name: str, factory):
        self.name = name
        self._factory = factory

    def load(self):
        return self._factory


class DummyDynamicProtocol:
    """Minimal protocol implementation used for entry-point discovery tests."""

    def __init__(self):
        self.contract = ProtocolContract(
            name="dynamic_protocol",
            family="test",
            description="dynamic protocol used by tests",
            capabilities=["discover"],
        )

    def supports(self, context: EnvBuildContext) -> bool:
        return True

    def normalize_context(self, context: EnvBuildContext) -> EnvBuildContext:
        return context

    def validate_context(self, context: EnvBuildContext) -> None:
        if context.distance <= 0:
            raise ValueError("distance must be positive")


class DummyDynamicEnvBuilder:
    """Minimal env builder used for entry-point discovery tests."""

    name = "dynamic_env"

    def build(self, context: EnvBuildContext):
        return object()


class DummyDynamicCircuitBuilder:
    """Minimal circuit builder used for entry-point discovery tests."""

    name = "dynamic_builder"
    supported_distances = [3]

    def build(self, spec):
        return "DEADBEEF"


def test_default_builder_registry_discovers_dynamic_discovery_entrypoint(monkeypatch):
    monkeypatch.setattr(
        envs_module,
        "_iter_entry_points",
        lambda group: [FakeEntryPoint("dynamic_env", DummyDynamicEnvBuilder)],
    )

    registry = envs_module.default_builder_registry()
    assert "dynamic_env" in registry.list()


def test_protocol_registry_discovers_dynamic_backends(monkeypatch):
    monkeypatch.setattr(
        protocols_module,
        "_iter_entry_points",
        lambda group: [FakeEntryPoint("dynamic_protocol", DummyDynamicProtocol)],
    )

    registry = protocols_module.create_default_protocol_registry()
    assert "dynamic_protocol" in registry.list()
    assert isinstance(registry.get("dynamic_protocol"), DummyDynamicProtocol)


def test_protocol_registry_recovers_from_entrypoint_failure(monkeypatch):
    monkeypatch.setattr(protocols_module, "_iter_entry_points", lambda _: (_ for _ in ()).throw(RuntimeError("entrypoint error")))
    registry = protocols_module.create_default_protocol_registry()
    assert set(registry.list()) == {"nisq", "sqkd", "surface"}


def test_container_discovers_dynamic_circuit_builders(monkeypatch):
    monkeypatch.setattr(
        "syndrome_net.container._iter_entry_points",
        lambda group: [FakeEntryPoint("dynamic_builder", DummyDynamicCircuitBuilder)],
    )

    container = DIContainer()
    container.register_defaults()
    assert "dynamic_builder" in container.circuit_builders.list()


def test_builder_registry_recovers_from_entrypoint_failure(monkeypatch):
    monkeypatch.setattr(envs_module, "_iter_entry_points", lambda _: (_ for _ in ()).throw(RuntimeError("entrypoint error")))
    registry = envs_module.default_builder_registry()
    assert set(registry.list()) == {
        "qec",
        "qec_continuous",
        "colour_gym",
        "colour_calibration",
        "colour_discovery",
    }


def test_container_logs_warning_when_loom_builder_unavailable(monkeypatch, caplog):
    monkeypatch.setattr(
        "syndrome_net.codes.LoomColorCodeBuilder.is_available",
        lambda: False,
    )
    caplog.set_level("WARNING")

    container = DIContainer()
    container.register_defaults()

    assert "el-loom builder unavailable because dependency is missing" in caplog.text
    assert "loom_color_code" not in container.circuit_builders.list()


class DummyEnvBuilder:
    """Minimal env builder that records the build context."""

    name = "dummy"

    def __init__(self) -> None:
        self.context: EnvBuildContext | None = None

    def build(self, context: EnvBuildContext):
        self.context = context
        return object()


class DummyTrainingStrategy:
    """Minimal strategy that emits exactly one done event."""

    name = "dummy"
    environment_name = "dummy"

    def run(self, runtime) -> None:
        runtime.emit_done(runtime.episodes)


class DiscoverBuildStrategy:
    """Strategy that only resolves an environment."""

    name = "discover"
    environment_name = "dummy"

    def __init__(self) -> None:
        self.received = False

    def run(self, runtime) -> None:
        runtime.env_builder_registry.get("dummy").build(
            EnvBuildContext(
                distance=runtime.distance,
                rounds=runtime.rounds,
                physical_error_rate=runtime.p,
            )
        )
        self.received = True
        runtime.emit_done(runtime.episodes)


class DiscoverProtocolBuildStrategy(_BaseTrainingStrategy):
    """Strategy that only verifies protocol-adjusted context propagation."""

    name = "discover_protocol"
    environment_name = "dummy"

    def run(self, runtime) -> None:
        self.build_env(runtime)
        runtime.emit_done(runtime.episodes)


def test_default_builder_registry_is_populated():
    registry = default_builder_registry()
    names = set(registry.list())
    assert names == {
        "qec",
        "qec_continuous",
        "colour_gym",
        "colour_calibration",
        "colour_discovery",
    }


def test_default_training_strategy_registry_includes_pepg():
    registry = default_training_strategy_registry()
    names = set(registry.list())
    assert {"ppo", "sac", "pepg", "ppo_colour", "sac_colour", "discover_colour"} <= names


def test_env_builder_registry_register_get_and_duplicates():
    registry = EnvBuilderRegistry()
    builder = DummyEnvBuilder()
    registry.register(builder)

    assert registry.list() == ["dummy"]
    assert registry.get("dummy") is builder

    with pytest.raises(ValueError, match="already registered"):
        registry.register(builder)


def test_env_builder_registry_unknown_name_is_explicit():
    registry = EnvBuilderRegistry()
    with pytest.raises(KeyError, match="Unknown environment builder"):
        registry.get("missing")


def test_training_strategy_registry_tracks_and_dispatches():
    registry = TrainingStrategyRegistry()
    strategy = DummyTrainingStrategy()
    registry.register(strategy)

    assert registry.list() == ["dummy"]
    assert registry.get("dummy") is strategy

    with pytest.raises(KeyError, match="Unknown training strategy"):
        registry.get("missing")


def test_default_protocol_registry_is_populated():
    assert DEFAULT_PROTOCOL_REGISTRY.list() == ["nisq", "sqkd", "surface"]


def test_rlrunner_uses_injected_strategy_registry():
    strategy = DiscoverBuildStrategy()
    builder = DummyEnvBuilder()
    strategy_registry = TrainingStrategyRegistry()
    strategy_registry.register(strategy)

    env_builder_registry = EnvBuilderRegistry()
    env_builder_registry.register(builder)

    runner = RLRunner(
        mode="discover",
        episodes=3,
        strategy_registry=strategy_registry,
        env_builder_registry=env_builder_registry,
    )

    runner.start()
    runner.join(timeout=1.0)

    events = runner.drain()
    assert any(isinstance(event, DoneEvent) for event in events)
    assert strategy.received
    assert isinstance(builder.context, EnvBuildContext)


def test_rlrunner_protocol_normalizes_surface_context():
    strategy = DiscoverProtocolBuildStrategy()
    builder = DummyEnvBuilder()
    strategy_registry = TrainingStrategyRegistry()
    strategy_registry.register(strategy)

    env_builder_registry = EnvBuilderRegistry()
    env_builder_registry.register(builder)

    runner = RLRunner(
        mode="discover_protocol",
        episodes=1,
        distance=4,
        strategy_registry=strategy_registry,
        env_builder_registry=env_builder_registry,
        protocol="surface",
    )
    runner.start()
    runner.join(timeout=1.0)
    assert runner.drain()  # done event emitted
    assert builder.context is not None
    assert builder.context.protocol == "surface"
    assert builder.context.distance >= 3
    assert builder.context.distance % 2 == 1


def test_rlrunner_raises_for_unknown_strategy():
    with pytest.raises(KeyError, match="Unknown training strategy"):
        RLRunner(mode="not_a_real_strategy").start()
