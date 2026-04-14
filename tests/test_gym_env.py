"""Tests for the OpenAI Gym QEC environment."""

import pytest
import numpy as np
from typing import Any

pytest.importorskip("gymnasium")
stim = pytest.importorskip("stim")

from surface_code_in_stem.rl_control.gym_env import QECGymEnv, QECContinuousControlEnv
from surface_code_in_stem.decoders import DecoderOutput, DecoderMetadata


def test_qec_gym_env_initialization():
    env = QECGymEnv(distance=3, rounds=2, physical_error_rate=0.01)
    
    assert env.observation_space.shape[0] > 0
    assert env.action_space.shape[0] > 0
    
    # Check that it resets correctly
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert isinstance(obs, np.ndarray)
    assert "mwpm_prediction" in info


def test_qec_gym_env_step():
    env = QECGymEnv(distance=3, rounds=2, physical_error_rate=0.01)
    env.reset(seed=42)
    
    # Take an arbitrary action (predict no logical errors)
    action = np.zeros(env.action_space.shape[0])
    
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(reward, float)
    assert terminated is True
    assert truncated is False
    assert "actual_logical" in info
    assert "is_correct" in info


def test_qec_gym_env_uses_requested_decoder(monkeypatch):
    class FakeDecoder:
        name = "mock_decoder"

        def decode(self, detector_events: Any, metadata: Any) -> DecoderOutput:
            predictions = np.zeros((1, metadata.num_observables), dtype=np.int8)
            return DecoderOutput(
                logical_predictions=predictions,
                decoder_name=self.name,
                diagnostics={"source": "mock"},
            )

    class FakeContainer:
        def get_decoder(self, name: str) -> object:
            assert name == "mock_decoder"
            return FakeDecoder()

    monkeypatch.setattr(
        "syndrome_net.container.get_container",
        lambda: FakeContainer(),
    )
    env = QECGymEnv(distance=3, rounds=2, physical_error_rate=0.01, decoder_name="mock_decoder")
    _, info = env.reset(seed=4)
    assert info["baseline_decoder_requested"] == "mock_decoder"
    assert info["baseline_decoder"] == "mock_decoder"
    assert isinstance(info["mwpm_prediction"], np.ndarray)
    assert info["mwpm_prediction"].shape[0] == env.num_observables


def test_qec_gym_env_falls_back_to_mwpm_when_decoder_missing(monkeypatch):
    class FakeContainer:
        def get_decoder(self, name: str) -> object:
            raise KeyError(name)

    monkeypatch.setattr(
        "syndrome_net.container.get_container",
        lambda: FakeContainer(),
    )
    env = QECGymEnv(distance=3, rounds=2, physical_error_rate=0.01, decoder_name="missing_decoder")
    _, info = env.reset(seed=5)
    assert info["baseline_decoder_requested"] == "missing_decoder"
    assert info["baseline_decoder"] == "mwpm"
    assert str(info.get("baseline_decoder_fallback_reason", "")).startswith("fallback_to_mwpm_decoder")


def test_qec_gym_env_exposes_baseline_diagnostics(monkeypatch):
    class FakeDecoder:
        name = "mock_diagnostics_decoder"

        def decode(self, detector_events: Any, metadata: Any) -> DecoderOutput:
            predictions = np.zeros((1, metadata.num_observables), dtype=np.int8)
            return DecoderOutput(
                logical_predictions=predictions,
                decoder_name=self.name,
                diagnostics={
                    "predecoder_backend": "fake_cnn",
                    "predecoder_latency_ms": 1.5,
                    "predecoder_fallback_reason": "none",
                    "contract_flags": "backend_enabled,contract_met",
                },
            )

    class FakeContainer:
        def get_decoder(self, name: str) -> object:
            assert name == "mock_diagnostics_decoder"
            return FakeDecoder()

    monkeypatch.setattr(
        "syndrome_net.container.get_container",
        lambda: FakeContainer(),
    )
    env = QECGymEnv(distance=3, rounds=2, physical_error_rate=0.01, decoder_name="mock_diagnostics_decoder")
    _, info = env.reset(seed=6)
    diagnostics = info["baseline_decoder_diagnostics"]
    assert diagnostics["predecoder_backend"] == "fake_cnn"
    assert diagnostics["predecoder_latency_ms"] == 1.5
    assert info["baseline_predecode_backend"] == "fake_cnn"
    assert info["baseline_predecode_latency_ms"] == 1.5
    assert info["baseline_predecode_fallback_reason"] == "none"
    assert info["baseline_contract_flags"] == "backend_enabled,contract_met"


def test_qec_gym_env_passes_predecoder_protocol_metadata(monkeypatch):
    captured_extra: dict[str, object] = {}

    class FakeDecoder:
        name = "fake_predecoder"

        def decode(self, detector_events: Any, metadata: DecoderMetadata) -> DecoderOutput:
            captured_extra.update(dict(metadata.extra))
            predictions = np.zeros((1, metadata.num_observables), dtype=np.int8)
            return DecoderOutput(logical_predictions=predictions, decoder_name=self.name, diagnostics={})

    class FakeContainer:
        def get_decoder(self, name: str) -> object:
            assert name == "fake_predecoder"
            return FakeDecoder()

    monkeypatch.setattr(
        "syndrome_net.container.get_container",
        lambda: FakeContainer(),
    )
    env = QECGymEnv(
        distance=3,
        rounds=2,
        physical_error_rate=0.01,
        decoder_name="fake_predecoder",
        protocol_metadata={
            "predecoder_backend": "torch",
            "predecoder_artifact": "/tmp/does-not-matter.pt",
            "predecoder_seed": 42,
        },
    )
    _, info = env.reset(seed=7)
    assert info["baseline_decoder"] == "fake_predecoder"
    assert captured_extra["predecoder_backend"] == "torch"
    assert captured_extra["predecoder_artifact"] == "/tmp/does-not-matter.pt"
    assert captured_extra["predecoder_seed"] == 42
    assert isinstance(info["baseline_decoder_diagnostics"], dict)


def test_qec_continuous_control_env():
    env = QECContinuousControlEnv(distance=3, rounds=2, parameter_dim=2, batch_shots=10)
    
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    
    action = np.array([0.01, -0.01])
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(reward, float)
    assert reward <= 0.0  # Reward is negative logical error rate
    assert "logical_error_rate" in info
    assert "effective_p" in info
